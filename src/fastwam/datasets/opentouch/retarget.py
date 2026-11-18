from __future__ import annotations

import math

import numpy as np

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17


def _normalize(v: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= eps:
        return np.zeros(3, dtype=np.float64), False
    return v / norm, True


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float64, copy=True)
    radius = window // 2
    out = np.full_like(x, np.nan, dtype=np.float64)
    for i in range(len(x)):
        part = x[max(0, i - radius) : min(len(x), i + radius + 1)]
        finite = np.isfinite(part)
        count = finite.sum(axis=0)
        np.divide(
            np.where(finite, part, 0.0).sum(axis=0),
            count,
            out=out[i],
            where=count > 0,
        )
    return out


def _matrix_to_rotvec(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    u, _, vh = np.linalg.svd(matrix)
    matrix = u @ vh
    if np.linalg.det(matrix) < 0:
        u[:, -1] *= -1
        matrix = u @ vh
    angle = math.acos(float(np.clip((np.trace(matrix) - 1) * 0.5, -1, 1)))
    skew = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    )
    if angle < eps:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        values, vectors = np.linalg.eig(matrix)
        axis = vectors[:, int(np.argmin(np.abs(values.real - 1.0)))].real
        axis, ok = _normalize(axis, eps)
        return axis * angle if ok else np.zeros(3, dtype=np.float64)
    return skew * (angle / (2 * math.sin(angle)))


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Return a normalized [qw,qx,qy,qz] quaternion."""
    m = matrix
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.asarray([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = math.sqrt(max(1 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2
            q = np.asarray([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif axis == 1:
            s = math.sqrt(max(1 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2
            q = np.asarray([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = math.sqrt(max(1 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2
            q = np.asarray([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q if q[0] >= 0 else -q


def retarget_hand_sequence(
    landmarks: np.ndarray,
    external_position: np.ndarray | None = None,
    *,
    workspace_scale_m: float = 0.30,
    smoothing_window: int = 5,
    gripper_closed_ratio: float = 0.18,
    gripper_open_ratio: float = 1.35,
) -> dict[str, np.ndarray]:
    """Create 7D task-space deltas and 8D task-space pseudo-proprio."""
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise ValueError(f"Expected landmarks [T,21,3], got {landmarks.shape}")
    frames = len(landmarks)
    finite = np.isfinite(landmarks).all(axis=(1, 2))
    nonzero = np.max(np.abs(np.nan_to_num(landmarks)), axis=(1, 2)) > 1e-8
    valid = finite & nonzero
    filtered = landmarks.copy()
    filtered[~valid] = np.nan
    filtered = _moving_average(filtered, smoothing_window)

    if external_position is not None and np.asarray(external_position).shape == (frames, 3):
        position = np.asarray(external_position, dtype=np.float64).copy()
        position[~valid] = np.nan
        position = _moving_average(position, smoothing_window)
    else:
        position = filtered[:, WRIST].copy()

    rotations = np.tile(np.eye(3), (frames, 1, 1))
    palm_width = np.full(frames, np.nan)
    for i in range(frames):
        if not valid[i] or not np.isfinite(position[i]).all():
            valid[i] = False
            continue
        x, ok_x = _normalize(filtered[i, INDEX_MCP] - filtered[i, PINKY_MCP])
        y_hint, ok_y = _normalize(filtered[i, MIDDLE_MCP] - filtered[i, WRIST])
        z, ok_z = _normalize(np.cross(x, y_hint))
        y, ok_y2 = _normalize(np.cross(z, x))
        if not (ok_x and ok_y and ok_z and ok_y2):
            valid[i] = False
            continue
        rotations[i] = np.stack([x, y, z], axis=-1)
        palm_width[i] = np.linalg.norm(filtered[i, INDEX_MCP] - filtered[i, PINKY_MCP])
    valid &= np.isfinite(palm_width) & (palm_width > 1e-8)
    scale = float(np.nanmedian(palm_width[valid])) if valid.any() else 1.0
    scale = max(scale, 1e-8)

    pinch = np.linalg.norm(filtered[:, THUMB_TIP] - filtered[:, INDEX_TIP], axis=-1) / np.maximum(palm_width, 1e-8)
    gripper = np.clip((pinch - gripper_closed_ratio) / (gripper_open_ratio - gripper_closed_ratio), 0, 1)
    gripper[~valid] = 0

    first_valid = int(np.flatnonzero(valid)[0]) if valid.any() else 0
    origin = position[first_valid] if valid.any() else np.zeros(3)
    base_rotation = rotations[first_valid] if valid.any() else np.eye(3)
    task_position = ((position - origin) @ base_rotation) / scale * workspace_scale_m
    task_rotation = np.einsum("ij,tjk->tik", base_rotation.T, rotations)

    proprio = np.zeros((frames, 8), dtype=np.float32)
    proprio[:, :3] = np.nan_to_num(task_position).astype(np.float32)
    for i in range(frames):
        proprio[i, 3:7] = _matrix_to_quaternion(task_rotation[i]) if valid[i] else np.asarray([1, 0, 0, 0])
    proprio[:, 7] = gripper.astype(np.float32)

    action = np.zeros((max(frames - 1, 0), 7), dtype=np.float32)
    action_valid = np.zeros(max(frames - 1, 0), dtype=bool)
    for i in range(frames - 1):
        if not (valid[i] and valid[i + 1]):
            continue
        delta_position = rotations[i].T @ (position[i + 1] - position[i]) / scale * workspace_scale_m
        delta_rotation = rotations[i].T @ rotations[i + 1]
        value = np.concatenate([delta_position, _matrix_to_rotvec(delta_rotation), [gripper[i + 1]]])
        if np.isfinite(value).all():
            action[i] = value.astype(np.float32)
            action_valid[i] = True
    return {
        "action": action,
        "action_valid": action_valid,
        "proprio": proprio,
        "pose_valid": valid,
        "gripper": gripper.astype(np.float32),
    }
