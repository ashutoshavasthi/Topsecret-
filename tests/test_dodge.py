import ctypes
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from skymind.config import load_config
from skymind.dodge import SENSORS, DodgeReflex, Projectile, ReflexConfig, ThreatSensorSpec, run_throws, sense

ROOT = Path(__file__).resolve().parents[1]


def test_projectile_ballistics():
    p = Projectile(np.array([0.0, 0.0, 2.0]), np.array([3.0, 0.0, 0.0]), 0.04)
    for _ in range(500):
        p.step(0.001)
    assert math.isclose(p.pos[0], 1.5, rel_tol=1e-6)
    assert abs(p.pos[2] - (2.0 - 0.5 * 9.81 * 0.25)) < 0.01  # semi-implicit Euler, 1 ms steps


def test_sensors_see_non_conducting_objects_within_their_limits():
    rng = np.random.default_rng(0)
    tof = ThreatSensorSpec("tof", "tof", 60, 45, 45)
    cam = ThreatSensorSpec("cam", "camera", 40, 66, 50)
    ball = Projectile(np.array([1.0, 0.0, 1.0]), np.zeros(3), 0.04)  # 8 cm paper ball, 1 m ahead
    for spec in (tof, cam):
        m = sense(spec, np.array([0.0, 0.0, 1.0]), 0.0, ball, rng)
        assert m is not None and abs(np.linalg.norm(m[0]) - 1.0) < 0.3
    far = Projectile(np.array([3.0, 0.0, 1.0]), np.zeros(3), 0.04)
    assert sense(tof, np.array([0.0, 0.0, 1.0]), 0.0, far, rng) is None  # small target: ToF range is short
    assert sense(cam, np.array([0.0, 0.0, 1.0]), 0.0, far, rng) is not None  # camera still sees it
    behind = Projectile(np.array([-1.0, 0.0, 1.0]), np.zeros(3), 0.04)
    assert sense(cam, np.array([0.0, 0.0, 1.0]), 0.0, behind, rng) is None


def _feed(reflex, start, vel, t0=0.0, n=8, rate=40.0, noise=0.0, rng=None):
    g = np.array([0, 0, -9.81])
    for i in range(n):
        t = t0 + i / rate
        rel = start + vel * t + 0.5 * g * t * t
        if noise:
            rel = rel + rng.normal(0, noise, 3)
        reflex.measure(t, rel, 0.05)
    return t


def _best_alternative(r, t, alt):
    """Predicted miss for the chosen escape vs. every simple alternative (and staying put)."""
    rel, vel = r._propagate(r.track, t)
    ts = np.linspace(0, r.cfg.horizon, 71)
    traj = rel + vel * ts[:, None] + 0.5 * np.array([0, 0, -9.81]) * ts[:, None] ** 2
    chosen = r.predicted_miss(r.escape_dir, traj, ts, alt)
    alts = [np.array(v, float) for v in ([0, 0, -1], [0, 0, 1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0])]
    others = [m for m in (r.predicted_miss(d, traj, ts, alt) for d in alts) if m is not None]
    stay = float(np.min(np.linalg.norm(traj, axis=1)))
    return chosen, others, stay


def test_reflex_drops_under_a_throw_passing_just_above():
    r = DodgeReflex(ReflexConfig(), 0.1, 2.0)
    t = _feed(r, np.array([2.0, 0.0, 0.3]), np.array([-6.0, 0.0, 0.885]))  # flat, passes ~5 cm above centre
    assert r.check(t, 1.1)
    assert r.escape_dir[2] < -0.5  # gravity-assisted drop under it


def test_reflex_ignores_a_miss():
    r = DodgeReflex(ReflexConfig(), 0.1, 2.0)
    t = _feed(r, np.array([2.0, 1.0, 0.0]), np.array([-6.0, 0.0, 1.0]))  # passes 1 m to the side
    assert not r.check(t, 1.1)


def test_reflex_predicts_a_lob_after_losing_sight_and_picks_the_best_escape():
    r = DodgeReflex(ReflexConfig(), 0.1, 2.5)
    # Arcs up and lands on us at 45 deg after 0.5 s; only seen for the first 0.2 s.
    t = _feed(r, np.array([1.5, 0.0, 0.3]), np.array([-3.0, 0.0, 1.85]), n=9)
    fired = [tt for tt in np.arange(t, 0.5, 0.01) if r.check(tt, 1.1)]
    assert fired, "should predict the arc after losing sight of it"
    chosen, others, stay = _best_alternative(r, fired[0], 1.1)
    assert chosen >= max(others) - 1e-9 and chosen > stay + 0.1


def test_reflex_respects_the_floor():
    r = DodgeReflex(ReflexConfig(), 0.1, 2.0)
    t = _feed(r, np.array([2.0, 0.0, 0.3]), np.array([-6.0, 0.0, 0.885]))
    assert r.check(t, 0.45)  # hovering low: dropping 0.5 m would hit the floor
    assert r.escape_dir[2] > -0.5


def test_reflex_beats_no_reflex_on_real_throws():
    cfg = load_config("config/nano_room.yaml")
    base = run_throws(cfg, SENSORS["none"], 5.0, throws=6, seed=3)
    dodge = run_throws(cfg, SENSORS["tof8x8+looming_wide"], 5.0, throws=6, seed=3)
    assert dodge.hits < base.hits
    assert dodge.dodge_rate >= 0.6


@pytest.fixture(scope="module")
def c_reflex(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("c") / "libdodge.so"
    src = ROOT / "firmware" / "dodge_reflex" / "dodge_reflex.c"
    subprocess.run([cc, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", str(src), "-o", str(out), "-lm"],
                   check=True, capture_output=True)
    lib = ctypes.CDLL(str(out))

    class Meas(ctypes.Structure):
        _fields_ = [("t", ctypes.c_float), ("rel", ctypes.c_float * 3), ("sigma", ctypes.c_float)]

    class Reflex(ctypes.Structure):
        _fields_ = [("body_radius", ctypes.c_float), ("ceiling", ctypes.c_float), ("min_altitude", ctypes.c_float),
                    ("horizon", ctypes.c_float), ("margin", ctypes.c_float),
                    ("accel_up", ctypes.c_float), ("accel_down", ctypes.c_float), ("accel_side", ctypes.c_float),
                    ("lag_vertical", ctypes.c_float), ("lag_side", ctypes.c_float),
                    ("escape_time", ctypes.c_float), ("escape_speed", ctypes.c_float), ("active", ctypes.c_int),
                    ("hist", Meas * 10), ("n_hist", ctypes.c_int), ("n_fit", ctypes.c_int), ("t_last", ctypes.c_float),
                    ("has_vel", ctypes.c_int), ("rel", ctypes.c_float * 3), ("vel", ctypes.c_float * 3),
                    ("escape_dir", ctypes.c_float * 3), ("triggers", ctypes.c_int)]

    lib.dr_init.argtypes = [ctypes.POINTER(Reflex), ctypes.c_float, ctypes.c_float, ctypes.c_float]
    lib.dr_measure.argtypes = [ctypes.POINTER(Reflex), ctypes.c_float, ctypes.c_float * 3, ctypes.c_float]
    lib.dr_check.argtypes = [ctypes.POINTER(Reflex), ctypes.c_float, ctypes.c_float]
    lib.dr_check.restype = ctypes.c_int
    u16x16 = ctypes.c_ushort * 16
    f3 = ctypes.c_float * 3
    lib.dr_tof_zones_to_threat.argtypes = [u16x16, u16x16, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                           ctypes.c_float, ctypes.c_float, f3, ctypes.c_float, f3,
                                           ctypes.POINTER(ctypes.c_float)]
    lib.dr_tof_zones_to_threat.restype = ctypes.c_int
    return lib, Reflex


def test_c_firmware_core_matches_python(c_reflex):
    lib, Reflex = c_reflex
    rng = np.random.default_rng(42)
    agree = total = 0
    dir_err = []
    for _ in range(300):
        cr = Reflex()
        lib.dr_init(ctypes.byref(cr), 0.1, 2.0, 0.35)
        pr = DodgeReflex(ReflexConfig(), 0.1, 2.0)
        start = np.array([rng.uniform(1.0, 3.0), rng.uniform(-1, 1), rng.uniform(-0.4, 0.4)])
        aim = rng.normal(0, 0.15, 3)
        tf = rng.uniform(0.25, 0.8)
        vel = (aim - start) / tf - 0.5 * np.array([0, 0, -9.81]) * tf
        alt = rng.uniform(0.4, 1.8)
        for i in range(12):
            t = i / 40.0
            rel = start + vel * t + 0.5 * np.array([0, 0, -9.81]) * t * t + rng.normal(0, 0.03, 3)
            pr.measure(t, rel, 0.05)
            lib.dr_measure(ctypes.byref(cr), t, (ctypes.c_float * 3)(*rel), 0.05)
            p_hit = pr.check(t, alt)
            c_hit = lib.dr_check(ctypes.byref(cr), t, alt)
            total += 1
            agree += p_hit == bool(c_hit)
            if p_hit and c_hit:
                dir_err.append(float(np.linalg.norm(np.array(cr.escape_dir[:]) - pr.escape_dir)))
    assert agree / total > 0.995  # float32 vs float64 can flip a borderline case, nothing more
    assert dir_err and max(dir_err) < 1e-2


def test_c_tof_zones_separate_thrown_objects_from_walls(c_reflex):
    lib, _ = c_reflex
    u16x16, f3 = ctypes.c_ushort * 16, ctypes.c_float * 3
    fov = math.radians(45)
    wall_prev, wall_cur = [1500] * 16, [1500] * 16
    rel, sig = f3(), ctypes.c_float()
    # Flying toward a wall at 1 m/s: ranges shrink, but our own motion explains it -> no threat.
    moving = [1500 - 17] * 16  # 1 m/s over 1/60 s
    assert lib.dr_tof_zones_to_threat(u16x16(*wall_prev), u16x16(*moving), 1 / 60, fov, fov, 0.0, 0.0,
                                      f3(1.0, 0, 0), 1.5, rel, ctypes.byref(sig)) == 0
    # Hovering; a ball in zone 6 (row 1, col 2: slightly right, slightly up) closes at 6 m/s.
    cur = list(wall_cur)
    prev = list(wall_prev)
    prev[6], cur[6] = 1200, 1100
    assert lib.dr_tof_zones_to_threat(u16x16(*prev), u16x16(*cur), 1 / 60, fov, fov, 0.0, 0.0,
                                      f3(0, 0, 0), 1.5, rel, ctypes.byref(sig)) == 1
    v = np.array(rel[:])
    assert abs(np.linalg.norm(v) - 1.1) < 1e-3
    assert v[0] > 0 and v[1] < 0 and v[2] > 0  # ahead, to the right, above centre
    assert 0.03 < sig.value < 0.2
