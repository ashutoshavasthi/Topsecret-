import numpy as np
import pytest

from membrain.device import PRESETS, DeviceArray, DeviceSpec, pulse_response


def rng():
    return np.random.default_rng(0)


def test_linear_device_takes_exactly_levels_pulses_to_span_range():
    spec = DeviceSpec(levels=50)
    arr = DeviceArray(spec, (3,), rng())
    for _ in range(50):
        arr.apply_pulses(np.ones(3))
    assert np.allclose(arr.g, spec.g_max)
    arr.apply_pulses(-25 * np.ones(3))
    assert np.allclose(arr.g, 0.5 * (spec.g_min + spec.g_max))


@pytest.mark.parametrize("nu", [0.5, 2.0, 5.0])
def test_nonlinear_closed_form_matches_single_steps_and_saturates(nu):
    spec = DeviceSpec(levels=40, nl_pot=nu, nl_dep=nu)
    one, many = DeviceArray(spec, (1,), rng()), DeviceArray(spec, (1,), rng())
    for _ in range(17):
        one.apply_pulses(np.array([1]))
    many.apply_pulses(np.array([17]))
    assert one.g[0] == pytest.approx(many.g[0], rel=1e-9)
    many.apply_pulses(np.array([23]))
    assert many.g[0] == pytest.approx(spec.g_max, rel=1e-9)   # exactly `levels` pulses span the range


def test_nonlinear_steps_shrink_towards_the_top():
    spec = PRESETS["rram"].with_(c2c=0.0)
    trace = pulse_response(spec)
    up = np.diff(trace[: spec.levels + 1])
    assert np.all(up > 0)
    assert up[0] > 5 * up[-1]


def test_stuck_devices_ignore_pulses():
    spec = DeviceSpec(levels=10, stuck_off=0.3, stuck_on=0.3)
    arr = DeviceArray(spec, (400,), rng())
    off, on = arr.stuck == -1, arr.stuck == 1
    assert 0.2 < off.mean() < 0.4 and 0.2 < on.mean() < 0.4
    arr.apply_pulses(np.full(400, 5))
    assert np.allclose(arr.g[off], spec.g_min)
    assert np.allclose(arr.g[on], spec.g_max)
    arr.reset()
    assert np.allclose(arr.g[on], spec.g_max)


def test_write_verify_beats_open_loop_on_nonlinear_device():
    spec = PRESETS["rram"].with_(stuck_off=0, stuck_on=0)
    target = np.random.default_rng(1).uniform(spec.g_min, spec.g_max, (40, 40))
    wv, ol = DeviceArray(spec, target.shape, rng()), DeviceArray(spec, target.shape, rng())
    wv.reset()
    wv.program(target)
    ol.program_open_loop(target)
    err_wv = np.abs(wv.g - target).mean() / spec.g_range
    err_ol = np.abs(ol.g - target).mean() / spec.g_range
    assert err_wv < 0.03
    assert err_wv < err_ol / 3


def test_write_verify_handles_abrupt_reset_devices():
    spec = PRESETS["pcm"].with_(stuck_off=0, stuck_on=0, read_noise=0.0)
    target = np.random.default_rng(2).uniform(spec.g_min, spec.g_max, (30, 30))
    arr = DeviceArray(spec, target.shape, rng())
    arr.program(target)
    assert np.abs(arr.g - target).mean() / spec.g_range < 0.03


def test_drift_lowers_conductance_by_power_law():
    spec = DeviceSpec(levels=10, drift_nu=0.05, drift_t0=1.0)
    arr = DeviceArray(spec, (5,), rng())
    arr.apply_pulses(np.full(5, 5))
    g0 = arr.conductance().copy()
    arr.age = 3600.0
    assert np.allclose(arr.conductance(), g0 * 3601.0 ** -0.05)
    arr.apply_pulses(np.full(5, 1))   # reprogramming restarts the drift clock
    assert arr.age == 0.0


def test_read_noise_is_relative():
    spec = DeviceSpec(levels=10, read_noise=0.05)
    arr = DeviceArray(spec, (20000,), rng())
    arr.apply_pulses(np.full(20000, 10))
    reads = arr.read()
    assert np.std(reads / arr.g) == pytest.approx(0.05, rel=0.05)
