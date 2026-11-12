import numpy as np
import pytest

from membrain.crossbar import IDEAL_PERIPHERY, Crossbar, Periphery, ir_drop_currents
from membrain.device import PRESETS, DeviceSpec


def make(n_in=60, n_out=7, device=PRESETS["ideal"], periphery=IDEAL_PERIPHERY, mapping="dual", w_max=1.0, seed=0):
    return Crossbar(n_in, n_out, device, periphery, w_max=w_max, mapping=mapping, rng=np.random.default_rng(seed))


@pytest.mark.parametrize("mapping", ["dual", "ref"])
def test_ideal_crossbar_computes_matrix_product_with_signed_inputs(mapping):
    rng = np.random.default_rng(1)
    w = rng.uniform(-0.9, 0.9, (60, 7))
    x = rng.normal(size=(25, 60))
    xb = make(mapping=mapping)
    xb.program(w)
    assert np.allclose(xb.matvec(x), x @ w, atol=5e-3 * np.abs(x @ w).max())
    d = rng.normal(size=(25, 7))
    assert np.allclose(xb.matvec_t(d), d @ w.T, atol=5e-3 * np.abs(d @ w.T).max())


def test_weights_are_clipped_to_range():
    xb = make()
    xb.program(np.full((60, 7), 3.0))
    assert np.allclose(xb.weights(), 1.0, atol=1e-2)


def test_adc_quantisation_error_is_bounded():
    rng = np.random.default_rng(2)
    w = rng.uniform(-1, 1, (60, 7))
    x = rng.random((40, 60))
    xb = make(periphery=Periphery(dac_bits=0, adc_bits=6, tile_rows=0))
    xb.program(w)
    xb.calibrate(x)
    full_scale = np.abs(x @ w).max()
    assert np.abs(xb.matvec(x) - x @ w).max() <= full_scale / (2 ** 5 - 1) * 0.51 + 1e-2 * full_scale


def test_tiling_matches_untiled_without_quantisation():
    rng = np.random.default_rng(3)
    w = rng.uniform(-1, 1, (300, 9))
    x = rng.random((10, 300))
    tiled = make(300, 9, periphery=IDEAL_PERIPHERY.with_(tile_rows=64))
    assert len(tiled.tiles) == 5
    tiled.program(w)
    assert np.allclose(tiled.matvec(x), x @ w, atol=1e-2 * np.abs(x @ w).max())


def test_ir_drop_solver_zero_resistance_is_ohm_law_and_losses_grow_with_resistance():
    rng = np.random.default_rng(4)
    g = rng.uniform(1, 40, (64, 16))
    v = rng.uniform(0, 0.2, (5, 64))
    assert np.allclose(ir_drop_currents(v, g, 0.0), v @ g)
    losses = [1 - ir_drop_currents(v, g, r).sum() / (v @ g).sum() for r in (0.5, 2.0, 8.0)]
    assert 0 < losses[0] < losses[1] < losses[2] < 1


def test_ir_drop_solver_matches_exact_circuit_solution():
    """Compare against a direct nodal-analysis solve of a small array."""
    rng = np.random.default_rng(5)
    n, m, r = 4, 3, 50.0
    g = rng.uniform(10, 100, (n, m)) * 1e-6           # siemens
    v = rng.uniform(0.1, 0.2, n)
    # Unknowns: row node voltages (n*m) then column node voltages (n*m).
    size = 2 * n * m
    a, b = np.zeros((size, size)), np.zeros(size)
    rid = lambda i, k: i * m + k
    cid = lambda i, k: n * m + i * m + k
    gw = 1 / r
    for i in range(n):
        for k in range(m):
            # Row node (i, k): left neighbour (or driver), right neighbour, cell down to column node.
            e = rid(i, k)
            a[e, e] += g[i, k]
            a[e, cid(i, k)] -= g[i, k]
            if k == 0:
                a[e, e] += gw
                b[e] += gw * v[i]
            else:
                a[e, e] += gw
                a[e, rid(i, k - 1)] -= gw
            if k < m - 1:
                a[e, e] += gw
                a[e, rid(i, k + 1)] -= gw
            # Column node (i, k): cell from row node, up neighbour, down neighbour (or sense at ground).
            e = cid(i, k)
            a[e, e] += g[i, k]
            a[e, rid(i, k)] -= g[i, k]
            if i > 0:
                a[e, e] += gw
                a[e, cid(i - 1, k)] -= gw
            a[e, e] += gw
            if i < n - 1:
                a[e, cid(i + 1, k)] -= gw
    x = np.linalg.solve(a, b)
    exact = np.array([x[cid(n - 1, k)] * gw for k in range(m)]) * 1e6   # uA into the sense amp
    approx = ir_drop_currents(v[None], g * 1e6, r, iters=30)[0]
    assert np.allclose(approx, exact, rtol=1e-6)


def test_dual_update_only_potentiates_and_refreshes_full_pairs():
    dev = DeviceSpec(levels=20)
    xb = make(10, 4, device=dev, periphery=Periphery(refresh_at=0.8))
    xb.program(np.zeros((10, 4)))
    for _ in range(30):
        xb.update(np.full((10, 4), 0.1))
        xb.update(np.full((10, 4), -0.1))
    assert xb.refreshes > 0
    assert np.abs(xb.weights()).max() < 0.25
    assert xb.gp.normalized().max() <= 0.8 + 1.0 / dev.levels + 1e-9


def test_stochastic_rounding_of_tiny_updates_is_unbiased():
    dev = DeviceSpec(levels=100)
    xb = make(50, 50, device=dev, mapping="ref")
    xb.program(np.zeros((50, 50)))
    step_w = xb.w_per_g * dev.step
    for _ in range(40):
        xb.update(np.full((50, 50), 0.25 * step_w))
    assert xb.weights().mean() == pytest.approx(40 * 0.25 * step_w, rel=0.05)


def test_fault_compensation_reduces_weight_error():
    dev = PRESETS["rram"].with_(stuck_on=0.03, stuck_off=0.0)
    w = np.random.default_rng(6).uniform(-0.3, 0.3, (80, 20))
    plain, comp = make(80, 20, device=dev, seed=7), make(80, 20, device=dev, seed=7)
    e_plain = plain.program(w, compensate=False)["weight_rms_error"]
    e_comp = comp.program(w, compensate=True)["weight_rms_error"]
    assert e_comp < 0.6 * e_plain
