import numpy as np

from skymind.math3d import quat_from_euler, quat_integrate, quat_to_euler, quat_to_rot, wrap_angle


def test_euler_roundtrip():
    for rpy in [(0.1, -0.2, 0.3), (-1.0, 0.5, 3.0), (0.0, 0.0, -2.5)]:
        assert np.allclose(quat_to_euler(quat_from_euler(*rpy)), rpy, atol=1e-9)


def test_rotation_is_orthonormal():
    r = quat_to_rot(quat_from_euler(0.4, -0.7, 1.9))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(r), 1.0)


def test_integrate_constant_yaw_rate():
    q = np.array([1.0, 0, 0, 0])
    for _ in range(100):
        q = quat_integrate(q, np.array([0, 0, 1.0]), 0.01)
    assert np.isclose(quat_to_euler(q)[2], 1.0, atol=1e-9)


def test_wrap_angle():
    assert np.isclose(wrap_angle(3 * np.pi), -np.pi)
    assert np.isclose(wrap_angle(-0.5), -0.5)
