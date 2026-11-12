"""A small multilayer perceptron whose layers are either digital (float) or memristor crossbars.

The same training loop drives both. For an analog layer:

* forward   y = x @ W        -> one read of the crossbar (row inputs, column outputs)
* backward  dx = delta @ W.T -> one *transposed* read of the same crossbar
* update    W += -lr * x.T @ delta -> the periphery turns the update into programming pulses
  (stochastically rounded to whole pulses); the devices' nonlinearity, asymmetry and noise decide
  what change actually happens.

Biases are an extra crossbar row driven by a constant input of 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crossbar import Crossbar, Periphery
from .device import DeviceSpec


def _with_bias(x: np.ndarray) -> np.ndarray:
    return np.hstack([x, np.ones((x.shape[0], 1))])


class DenseLayer:
    """Digital float layer. Optional hardware-aware training: weight clipping plus noise injection."""

    analog = False

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 w_clip: float | None = None, noise: float = 0.0):
        self.rng = rng
        self.w = np.vstack([rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_out)), np.zeros((1, n_out))])
        self.w_clip, self.noise = w_clip, noise
        self._x = None

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        self._x = _with_bias(x)
        w = self.w
        if train and self.noise > 0 and self.w_clip:
            w = w + self.noise * self.w_clip * self.rng.standard_normal(w.shape)
        return self._x @ w

    def backward(self, delta: np.ndarray) -> np.ndarray:
        self._grad = self._x.T @ delta
        return delta @ self.w[:-1].T

    def step(self, lr: float) -> int:
        self.w -= lr * self._grad
        if self.w_clip:
            np.clip(self.w, -self.w_clip, self.w_clip, out=self.w)
        return 0


class AnalogLayer:
    """Layer stored in a memristor crossbar (weights + bias row)."""

    analog = True

    def __init__(self, n_in: int, n_out: int, device: DeviceSpec, periphery: Periphery, w_max: float,
                 mapping: str, rng: np.random.Generator):
        self.xbar = Crossbar(n_in + 1, n_out, device, periphery, w_max=w_max, mapping=mapping, rng=rng)
        self._x = None
        self.pulses = 0
        self.out_scale = 1.0      # digital gain set by global drift compensation
        self._drift_ref = None

    @property
    def w(self) -> np.ndarray:
        return self.xbar.weights()

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        self._x = _with_bias(x)
        return self.xbar.matvec(self._x) * self.out_scale

    def backward(self, delta: np.ndarray) -> np.ndarray:
        self._delta = delta
        return self.xbar.matvec_t(delta)[:, :-1]

    def step(self, lr: float) -> int:
        sent = self.xbar.update(-lr * (self._x.T @ self._delta))
        self.pulses += sent
        return sent

    def calibrate(self, x: np.ndarray) -> None:
        self.xbar.calibrate(_with_bias(x))


class MLP:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray, train: bool = False) -> np.ndarray:
        self._h = []
        for i, layer in enumerate(self.layers):
            x = layer.forward(x, train)
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)
                self._h.append(x)
        return x

    def backward(self, delta: np.ndarray) -> None:
        for i in range(len(self.layers) - 1, -1, -1):
            delta = self.layers[i].backward(delta)
            if i > 0:
                delta = delta * (self._h[i - 1] > 0)

    def step(self, lr: float) -> int:
        return sum(layer.step(lr) for layer in self.layers)

    def predict(self, x: np.ndarray, batch: int = 500) -> np.ndarray:
        return np.concatenate([self.forward(x[i:i + batch]).argmax(1) for i in range(0, len(x), batch)])

    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(x) == y).mean())

    def calibrate(self, x: np.ndarray) -> None:
        """Calibrate every analog layer's ADC range on representative data (layer by layer)."""
        for i, layer in enumerate(self.layers):
            if getattr(layer, "analog", False):
                layer.xbar.adc_range = None
                layer.calibrate(x)
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)

    @property
    def pulses(self) -> int:
        return sum(layer.xbar.pulses for layer in self.layers if layer.analog)

    def set_age(self, seconds: float) -> None:
        """Pretend `seconds` have passed since the weights were programmed (conductance drift)."""
        for layer in self.layers:
            if layer.analog:
                layer.xbar.set_age(seconds)

    def drift_compensate(self, x_cal: np.ndarray, reference: bool = False) -> None:
        """Global drift compensation: one digital gain per layer.

        With ``reference=True`` (right after programming) it records the mean absolute output of each
        analog layer for a fixed calibration batch. Later calls re-measure it and set the layer's
        output gain so the average magnitude matches again. Drift mostly shrinks all conductances
        together, so a single gain recovers most of the accuracy. It needs no retraining and no
        reprogramming, just one extra read of a calibration batch.
        """
        x = x_cal
        for i, layer in enumerate(self.layers):
            if layer.analog:
                layer.out_scale = 1.0
                level = float(np.abs(layer.forward(x)).mean())
                if reference or layer._drift_ref is None:
                    layer._drift_ref = level
                else:
                    layer.out_scale = layer._drift_ref / max(level, 1e-12)
            x = layer.forward(x)
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0.0)


def softmax_xent_grad(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    z = logits - logits.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    n = len(y)
    loss = float(-np.log(p[np.arange(n), y] + 1e-12).mean())
    p[np.arange(n), y] -= 1.0
    return loss, p / n


@dataclass
class TrainConfig:
    epochs: int = 12
    lr: float = 0.1
    lr_decay: float = 0.85     # per epoch
    batch: int = 32
    seed: int = 0


def train(model: MLP, x: np.ndarray, y: np.ndarray, cfg: TrainConfig = TrainConfig(),
          x_val: np.ndarray | None = None, y_val: np.ndarray | None = None, log=None) -> list[dict]:
    rng = np.random.default_rng(cfg.seed)
    history = []
    lr = cfg.lr
    for epoch in range(1, cfg.epochs + 1):
        order = rng.permutation(len(x))
        losses = []
        for i in range(0, len(x), cfg.batch):
            idx = order[i:i + cfg.batch]
            loss, delta = softmax_xent_grad(model.forward(x[idx], train=True), y[idx])
            model.backward(delta)
            model.step(lr)
            losses.append(loss)
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "lr": lr}
        if x_val is not None:
            rec["val_acc"] = model.accuracy(x_val, y_val)
        history.append(rec)
        if log:
            log(rec)
        lr *= cfg.lr_decay
    return history


def float_mlp(sizes: list[int], rng: np.random.Generator, w_clip: float | None = None, noise: float = 0.0) -> MLP:
    return MLP([DenseLayer(a, b, rng, w_clip, noise) for a, b in zip(sizes[:-1], sizes[1:])])


def default_w_max(n_in: int, factor: float = 4.0) -> float:
    """Weight range of a crossbar layer trained from scratch: `factor` x the He-init std."""
    return factor * np.sqrt(2.0 / n_in)


def analog_mlp(sizes: list[int], device: DeviceSpec, periphery: Periphery, rng: np.random.Generator,
               mapping: str = "dual", w_max: list[float] | None = None) -> MLP:
    """A crossbar network initialised with He-random weights (for in-situ training)."""
    layers = []
    for k, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        wm = w_max[k] if w_max else default_w_max(a)
        layer = AnalogLayer(a, b, device, periphery, wm, mapping, rng)
        init = np.vstack([rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros((1, b))])
        layer.xbar.program(init)
        layers.append(layer)
    return MLP(layers)


def to_analog(model: MLP, device: DeviceSpec, periphery: Periphery, rng: np.random.Generator,
              mapping: str = "dual", method: str = "write_verify", w_max: list[float] | None = None,
              clip_percentile: float = 99.9, compensate: bool = True) -> tuple[MLP, dict]:
    """Program a trained float network into crossbars (ex-situ / "train then deploy")."""
    layers, stats = [], []
    for k, layer in enumerate(model.layers):
        w = layer.w
        wm = w_max[k] if w_max else (layer.w_clip or float(np.percentile(np.abs(w), clip_percentile)))
        al = AnalogLayer(w.shape[0] - 1, w.shape[1], device, periphery, wm, mapping, rng)
        stats.append(al.xbar.program(w, method=method, compensate=compensate))
        layers.append(al)
    return MLP(layers), {
        "pulses": int(sum(s["pulses"] for s in stats)),
        "weight_rms_error": float(np.mean([s["weight_rms_error"] for s in stats])),
        "bad_weights": float(np.mean([s["bad_weights"] for s in stats])),
    }
