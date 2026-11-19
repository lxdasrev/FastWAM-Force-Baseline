from __future__ import annotations

import numpy as np


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError(f"Metric value/mask shape mismatch: {values.shape} vs {mask.shape}")
    selected = values[mask]
    if selected.size == 0:
        return float("nan")
    return float(selected.mean())


def compute_opentouch_action_metrics(
    *,
    pred_action: np.ndarray,
    gt_action: np.ndarray,
    action_valid: np.ndarray,
) -> dict[str, float]:
    """Compute denormalized action errors for one OpenTouch window."""
    pred_action = np.asarray(pred_action, dtype=np.float64)
    gt_action = np.asarray(gt_action, dtype=np.float64)
    action_valid = np.asarray(action_valid, dtype=bool)
    if pred_action.shape != gt_action.shape or pred_action.ndim != 2 or pred_action.shape[1] != 7:
        raise ValueError(
            "OpenTouch action prediction/GT must both be [T,7], got "
            f"{pred_action.shape} and {gt_action.shape}"
        )
    if action_valid.shape != (pred_action.shape[0],):
        raise ValueError(
            f"OpenTouch action_valid must be [T], got {action_valid.shape} for T={pred_action.shape[0]}"
        )
    action_abs = np.abs(pred_action - gt_action)
    action_sq = np.square(pred_action - gt_action)
    action_mask = np.broadcast_to(action_valid[:, None], action_abs.shape)
    return {
        "action_mae": _masked_mean(action_abs, action_mask),
        "action_rmse": float(np.sqrt(_masked_mean(action_sq, action_mask))),
        "action_translation_mae_m": _masked_mean(action_abs[:, :3], action_mask[:, :3]),
        "action_rotation_mae_rad": _masked_mean(action_abs[:, 3:6], action_mask[:, 3:6]),
        "action_gripper_mae": _masked_mean(action_abs[:, 6:7], action_mask[:, 6:7]),
    }


def compute_opentouch_inference_metrics(
    *,
    pred_action: np.ndarray,
    gt_action: np.ndarray,
    action_valid: np.ndarray,
    pred_force: np.ndarray,
    gt_force: np.ndarray,
    force_valid: np.ndarray,
) -> dict[str, float]:
    """Compute denormalized action/force errors for an OpenTouch window."""
    pred_action = np.asarray(pred_action, dtype=np.float64)
    gt_action = np.asarray(gt_action, dtype=np.float64)
    pred_force = np.asarray(pred_force, dtype=np.float64)
    gt_force = np.asarray(gt_force, dtype=np.float64)
    action_valid = np.asarray(action_valid, dtype=bool)
    force_valid = np.asarray(force_valid, dtype=bool)

    if pred_action.shape != gt_action.shape or pred_action.ndim != 2 or pred_action.shape[1] != 7:
        raise ValueError(
            "OpenTouch action prediction/GT must both be [T,7], got "
            f"{pred_action.shape} and {gt_action.shape}"
        )
    if pred_force.shape != gt_force.shape or pred_force.ndim != 2 or pred_force.shape[1] != 2:
        raise ValueError(
            "OpenTouch force prediction/GT must both be [T,2], got "
            f"{pred_force.shape} and {gt_force.shape}"
        )
    if action_valid.shape != (pred_action.shape[0],):
        raise ValueError(
            f"OpenTouch action_valid must be [T], got {action_valid.shape} for T={pred_action.shape[0]}"
        )
    if force_valid.shape != pred_force.shape:
        raise ValueError(
            f"OpenTouch force_valid must be {pred_force.shape}, got {force_valid.shape}"
        )

    action_metrics = compute_opentouch_action_metrics(
        pred_action=pred_action,
        gt_action=gt_action,
        action_valid=action_valid,
    )
    force_abs = np.abs(pred_force - gt_force)
    force_sq = np.square(pred_force - gt_force)
    force_mae_per_finger = [
        _masked_mean(force_abs[:, finger], force_valid[:, finger])
        for finger in range(2)
    ]

    return {
        **action_metrics,
        "force_mae": _masked_mean(force_abs, force_valid),
        "force_rmse": float(np.sqrt(_masked_mean(force_sq, force_valid))),
        "force_thumb_mae": force_mae_per_finger[0],
        "force_index_mae": force_mae_per_finger[1],
    }
