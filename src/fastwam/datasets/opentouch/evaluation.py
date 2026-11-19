from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .metrics import compute_opentouch_action_metrics, compute_opentouch_inference_metrics


def _first(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1:
        return value[0]
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


@torch.no_grad()
def infer_opentouch_sample(
    *,
    model,
    dataset,
    sample: dict[str, Any],
    num_inference_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run one OpenTouch action or action+force sample with shared semantics."""
    video = sample["video"]
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"OpenTouch video must be [3,T,H,W], got {tuple(video.shape)}")
    action = sample["action"]
    if action.ndim == 3:
        action = action[0]
    proprio = sample["proprio"]
    if proprio.ndim == 3:
        proprio = proprio[0]
    proprio = proprio[0]

    prompt = _first(sample["prompt"])
    context = sample.get("context")
    context_mask = sample.get("context_mask")
    if isinstance(context, torch.Tensor) and context.ndim == 3:
        context = context[0]
    if isinstance(context_mask, torch.Tensor) and context_mask.ndim == 2:
        context_mask = context_mask[0]

    common_kwargs = {
        "input_image": video[:, 0],
        "action_horizon": int(action.shape[0]),
        "proprio": proprio,
        "num_inference_steps": int(num_inference_steps),
        "seed": int(seed),
        "tiled": False,
        "prompt": prompt if context is None else None,
    }
    if context is not None:
        common_kwargs["context"] = context
        common_kwargs["context_mask"] = context_mask

    force = sample.get("force")
    if force is not None:
        if getattr(model, "force_expert", None) is None:
            raise ValueError("Force sample was provided to a two-expert model.")
        if force.ndim == 3:
            force = force[0]
        observed_steps = int(model.force_expert.observed_steps)
        force_valid_tensor = sample.get("force_valid")
        if force_valid_tensor is None:
            force_valid_tensor = torch.ones_like(force, dtype=torch.bool)
        elif force_valid_tensor.ndim == 3:
            force_valid_tensor = force_valid_tensor[0]
        force_pad = sample.get("force_is_pad")
        if isinstance(force_pad, torch.Tensor) and force_pad.ndim == 2:
            force_pad = force_pad[0]
        infer_kwargs = {
            **common_kwargs,
            "force_history": force[:observed_steps],
            "force_history_valid": force_valid_tensor[:observed_steps],
            "force_history_is_pad": (
                None if force_pad is None else force_pad[:observed_steps]
            ),
        }
        pred = model.infer_action_force(**infer_kwargs)
        pred_force = dataset.denormalize_force(
            pred["future_force"].detach().cpu().numpy()
        )
        gt_force = dataset.denormalize_force(
            force[observed_steps:].detach().cpu().numpy()
        )
        force_valid = force_valid_tensor[observed_steps:].detach().cpu().numpy().astype(bool)
        if force_pad is not None:
            force_valid &= ~force_pad[observed_steps:].detach().cpu().numpy().astype(bool)[:, None]
    else:
        if getattr(model, "force_expert", None) is not None:
            raise ValueError("Two-expert OpenTouch sample was provided to a force-enabled model.")
        infer_kwargs = common_kwargs
        pred = model.infer_action(**infer_kwargs)
        pred_force = gt_force = force_valid = None

    pred_action = dataset.denormalize_action(pred["action"].detach().cpu().numpy())
    gt_action = dataset.denormalize_action(action.detach().cpu().numpy())
    action_pad = sample.get("action_is_pad")
    if isinstance(action_pad, torch.Tensor) and action_pad.ndim == 2:
        action_pad = action_pad[0]
    action_valid = (
        np.ones((gt_action.shape[0],), dtype=bool)
        if action_pad is None
        else ~action_pad.detach().cpu().numpy().astype(bool)
    )
    if pred_force is None:
        metrics = compute_opentouch_action_metrics(
            pred_action=pred_action,
            gt_action=gt_action,
            action_valid=action_valid,
        )
        mode = "baseline"
    else:
        metrics = compute_opentouch_inference_metrics(
            pred_action=pred_action,
            gt_action=gt_action,
            action_valid=action_valid,
            pred_force=pred_force,
            gt_force=gt_force,
            force_valid=force_valid,
        )
        mode = "force"
    return {
        "mode": mode,
        "metrics": metrics,
        "pred_action": pred_action,
        "gt_action": gt_action,
        "action_valid": action_valid,
        "pred_force": pred_force,
        "gt_force": gt_force,
        "force_valid": force_valid,
        "inference_kwargs": infer_kwargs,
    }
