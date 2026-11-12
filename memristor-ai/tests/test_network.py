import numpy as np
import pytest

from membrain import experiments as ex
from membrain.cli import main
from membrain.crossbar import IDEAL_PERIPHERY, Periphery
from membrain.data import GLYPHS, make_digits
from membrain.device import PRESETS
from membrain.energy import inference_energy
from membrain.network import TrainConfig, analog_mlp, float_mlp, to_analog, train


@pytest.fixture(scope="module")
def data():
    xtr, ytr = make_digits(1500, seed=1)
    xte, yte = make_digits(500, seed=2)
    return xtr, ytr, xte, yte


@pytest.fixture(scope="module")
def trained(data):
    xtr, ytr, xte, yte = data
    m = float_mlp([256, 32, 10], np.random.default_rng(0))
    train(m, xtr, ytr, TrainConfig(epochs=8))
    return m


def test_digits_are_deterministic_balanced_and_in_range():
    x1, y1 = make_digits(200, seed=5)
    x2, y2 = make_digits(200, seed=5)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)
    assert x1.shape == (200, 256) and x1.min() >= 0 and x1.max() <= 1
    assert np.bincount(y1).tolist() == [20] * 10
    assert set(GLYPHS) == set(range(10))


def test_float_network_learns(trained, data):
    assert trained.accuracy(data[2], data[3]) > 0.75


def test_ideal_deployment_preserves_accuracy(trained, data):
    xtr, _, xte, yte = data
    am, stats = to_analog(trained, PRESETS["ideal"], IDEAL_PERIPHERY, np.random.default_rng(0))
    assert stats["weight_rms_error"] < 0.005
    assert np.mean(am.predict(xte) == trained.predict(xte)) > 0.98


def test_realistic_deployment_ordering(trained, data):
    """Write-verify with fault compensation must beat open-loop programming on RRAM."""
    xtr, _, xte, yte = data
    accs = {}
    for method, comp in [("open_loop", False), ("write_verify", True)]:
        am, _ = to_analog(trained, PRESETS["rram"], Periphery(), np.random.default_rng(3), method=method, compensate=comp)
        am.calibrate(xtr[:300])
        accs[method] = am.accuracy(xte, yte)
    assert accs["write_verify"] > accs["open_loop"]
    assert accs["write_verify"] > trained.accuracy(xte, yte) - 0.05


def test_on_chip_training_learns_on_ecram(data):
    xtr, ytr, xte, yte = data
    am = analog_mlp([256, 32, 10], PRESETS["ecram"], Periphery(), np.random.default_rng(0), mapping="ref")
    hist = train(am, xtr, ytr, TrainConfig(epochs=5), xte, yte)
    assert hist[-1]["val_acc"] > 0.7
    assert am.pulses > 0


def test_drift_compensation_restores_output_scale(trained, data):
    xtr, _, xte, _ = data
    dev = PRESETS["pcm"].with_(drift_nu_std=0.0, stuck_on=0, stuck_off=0, read_noise=0.0)
    am, _ = to_analog(trained, dev, Periphery(adc_bits=0, dac_bits=0), np.random.default_rng(0))
    cal = xtr[:200]
    fresh = am.forward(xte[:100])
    am.drift_compensate(cal, reference=True)
    am.set_age(3.15e7)
    drifted = am.forward(xte[:100])
    am.drift_compensate(cal)
    fixed = am.forward(xte[:100])
    assert np.abs(drifted).mean() < 0.5 * np.abs(fresh).mean()
    assert np.abs(fixed).mean() == pytest.approx(np.abs(fresh).mean(), rel=0.05)


def test_energy_model_reports_analog_advantage(trained, data):
    am, _ = to_analog(trained, PRESETS["rram"], Periphery(), np.random.default_rng(0))
    e = inference_energy(am, data[2][:50])
    assert e["macs"] == 257 * 32 + 33 * 10
    assert 0 < e["analog_pj"] < e["digital_pj"]
    assert e["breakdown_pj"]["array"] > 0


def test_demo_runs_end_to_end(tmp_path):
    res = ex.run_demo(PRESETS["ecram"], epochs=1, n_train=400, n_test=200, log=lambda *_: None,
                      chart=str(tmp_path / "c.svg"))
    names = [r["method"] for r in res["rows"]]
    assert len(names) == 8 and all(0 <= r["accuracy"] <= 1 for r in res["rows"])
    assert (tmp_path / "c.svg").read_text().startswith("<svg")


def test_cli_devices(tmp_path, capsys):
    main(["devices", "--out", str(tmp_path)])
    assert "rram" in capsys.readouterr().out
    assert (tmp_path / "device_curves.svg").exists()
