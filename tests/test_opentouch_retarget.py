import math

import numpy as np

from fastwam.datasets.opentouch.retarget import retarget_hand_sequence


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _base_hand() -> np.ndarray:
    hand = np.zeros((21, 3), dtype=np.float64)
    hand[0] = [0.0, 0.0, 0.0]
    hand[5] = [0.05, 0.10, 0.0]
    hand[9] = [0.0, 0.11, 0.0]
    hand[17] = [-0.05, 0.10, 0.0]
    hand[4] = [-0.025, 0.16, 0.0]
    hand[8] = [0.025, 0.16, 0.0]
    # Nonzero filler points keep the synthetic frame representative.
    for index in range(21):
        if not hand[index].any():
            hand[index] = [0.001 * index, 0.05 + 0.002 * index, 0.001]
    hand[0] = [0.0, 0.0, 0.0]
    return hand


def test_action_and_proprio_share_initial_palm_canonical_frame():
    base = _base_hand()
    initial_rotation = _rotation(np.array([0.0, 0.0, 1.0]), math.pi / 2)
    canonical_delta_rotation = _rotation(np.array([1.0, 0.0, 0.0]), 0.2)
    next_rotation = initial_rotation @ canonical_delta_rotation
    canonical_translation = np.array([0.01, -0.02, 0.005])
    world_translation = initial_rotation @ canonical_translation

    frame0 = base @ initial_rotation.T
    frame1 = base @ next_rotation.T + world_translation
    result = retarget_hand_sequence(
        np.stack([frame0, frame1]),
        workspace_scale_m=0.10,
        smoothing_window=1,
    )

    np.testing.assert_allclose(result["proprio"][0, :3], 0.0, atol=1e-6)
    np.testing.assert_allclose(result["proprio"][0, 3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-6)
    # Synthetic palm width is 0.1 m and workspace_scale_m is also 0.1,
    # therefore the canonical translation is unchanged by scaling.
    np.testing.assert_allclose(result["action"][0, :3], canonical_translation, atol=1e-6)
    np.testing.assert_allclose(result["action"][0, 3:6], [0.2, 0.0, 0.0], atol=1e-6)
    assert result["action_valid"][0]


def test_action_dimension_seven_is_next_absolute_gripper_target():
    base = _base_hand()
    second = base.copy()
    second[4] = [-0.06, 0.16, 0.0]
    second[8] = [0.06, 0.16, 0.0]
    result = retarget_hand_sequence(
        np.stack([base, second]),
        workspace_scale_m=0.10,
        smoothing_window=1,
    )
    np.testing.assert_allclose(result["action"][0, 6], result["gripper"][1], atol=1e-7)
    assert not np.isclose(
        result["action"][0, 6], result["gripper"][1] - result["gripper"][0]
    )
