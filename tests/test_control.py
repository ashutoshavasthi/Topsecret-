import math

import numpy as np

from skymind.config import SkyMindConfig
from skymind.control.estimator import MahonyAHRS, StateEstimate, StateEstimator
from skymind.control.flight_controller import FlightController, Setpoint
from skymind.control.mixer import Mixer, allocation_matrix
from skymind.control.pid import PID
from skymind.math3d import quat_from_euler, quat_to_euler
from skymind.sim.physics import QuadrotorSim


def test_pid_anti_windup():
    pid = PID(1.0, 10.0, 0.0, output_limit=1.0)
    for _ in range(1000):
        pid.update(100.0, 0.0, 0.01)  # saturated the whole time
    assert abs(pid.integral) < 1e-9
    # Once the error flips the output must respond immediately, not after unwinding.
    assert pid.update(-1.0, 0.0, 0.01) < 0


def test_mixer_roundtrip_and_priority():
    m = Mixer(0.18, 0.016, 8.0)
    motors = m.mix(12.0, np.array([0.1, -0.2, 0.01]))
    wrench = allocation_matrix(0.18, 0.016) @ motors
    assert np.allclose(wrench, [12.0, 0.1, -0.2, 0.01], atol=1e-9)
    # Asking for more collective than available keeps the roll torque intact.
    motors = m.mix(40.0, np.array([0.3, 0.0, 0.0]))
    assert np.isclose((allocation_matrix(0.18, 0.016) @ motors)[1], 0.3, atol=1e-9)
    assert motors.max() <= 8.0


def test_mahony_converges_and_learns_bias():
    ahrs = MahonyAHRS(kp=1.0, ki=0.1)
    ahrs.q = quat_from_euler(0.4, -0.3, 0.0)  # wrong initial guess
    bias = np.array([0.02, -0.01, 0.0])
    up = np.array([0.0, 0.0, 9.81])  # level and still
    for _ in range(int(40 / 0.004)):
        ahrs.update(bias, up, 0.004, heading=0.0)
    roll, pitch, yaw = quat_to_euler(ahrs.q)
    assert abs(roll) < math.radians(0.5) and abs(pitch) < math.radians(0.5) and abs(yaw) < math.radians(0.5)
    assert np.allclose(ahrs.bias_integral, -bias, atol=3e-3)


def _fly(cfg, sim, est, fc, sp, seconds, dt=0.004, truth=False):
    for k in range(int(seconds / dt)):
        imu = sim.imu()
        est.update_imu(imu.gyro, imu.accel, dt, sim.heading_measurement())
        if k % 12 == 0:
            est.update_position(sim.position_measurement(), 12 * dt)
        e = est.estimate()
        if truth:
            st = sim.state
            e = StateEstimate(st.position.copy(), st.velocity.copy(), st.quaternion.copy(), st.omega.copy())
        sim.step(fc.update(e, sp, dt).motors, dt)


def _setup(seed=3):
    cfg = SkyMindConfig()
    sim = QuadrotorSim(cfg.vehicle, cfg.sensors, np.random.default_rng(seed))
    est = StateEstimator(cfg.estimator)
    fc = FlightController(cfg.vehicle, cfg.control)
    return cfg, sim, est, fc


def test_self_balancing_recovers_from_large_tilt():
    """Drop the drone at 35/30 degrees of tilt mid-air: it must level out and hold position."""
    cfg, sim, est, fc = _setup()
    sim.reset(position=[0, 0, 3.0], euler=[math.radians(35), math.radians(-30), 0.5])
    est.reset(sim.state.position, sim.state.quaternion)
    sp = Setpoint(position=np.array([0.0, 0.0, 3.0]), yaw=0.5)
    _fly(cfg, sim, est, fc, sp, 1.5)
    roll, pitch, _ = sim.state.euler
    assert abs(roll) < math.radians(8) and abs(pitch) < math.radians(8)
    _fly(cfg, sim, est, fc, sp, 6.0)
    roll, pitch, yaw = sim.state.euler
    assert abs(roll) < math.radians(4) and abs(pitch) < math.radians(4)
    assert abs(yaw - 0.5) < math.radians(5)
    assert np.linalg.norm(sim.state.position - [0, 0, 3.0]) < 0.35


def test_rejects_torque_disturbance():
    """A sudden 0.3 N m roll kick (like a bird strike or payload shift) is cancelled."""
    cfg, sim, est, fc = _setup()
    sim.reset(position=[0, 0, 2.0])
    est.reset(sim.state.position, sim.state.quaternion)
    sp = Setpoint(position=np.array([0.0, 0.0, 2.0]))
    _fly(cfg, sim, est, fc, sp, 2.0, truth=True)
    sim.external_torque = np.array([0.3, 0.0, 0.0])
    _fly(cfg, sim, est, fc, sp, 0.15, truth=True)
    sim.external_torque = np.zeros(3)
    _fly(cfg, sim, est, fc, sp, 2.0, truth=True)
    roll, pitch, _ = sim.state.euler
    assert abs(roll) < math.radians(2) and abs(pitch) < math.radians(2)


def test_tilt_limit_respected():
    cfg, sim, est, fc = _setup()
    sim.reset(position=[0, 0, 5.0])
    est.reset(sim.state.position, sim.state.quaternion)
    sp = Setpoint(position=np.array([50.0, 0.0, 5.0]))  # huge step
    max_tilt = 0.0
    for _ in range(int(3 / 0.004)):
        st = sim.state
        e = StateEstimate(st.position.copy(), st.velocity.copy(), st.quaternion.copy(), st.omega.copy())
        sim.step(fc.update(e, sp, 0.004).motors, 0.004)
        r, p, _ = sim.state.euler
        max_tilt = max(max_tilt, math.degrees(math.acos(math.cos(r) * math.cos(p))))
    assert max_tilt < cfg.control.max_tilt_deg + 5.0
