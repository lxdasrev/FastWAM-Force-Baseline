from __future__ import annotations

import json
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.datasets.opentouch.evaluation import infer_opentouch_sample
from fastwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)


def _aggregate(records: list[dict]) -> dict[str, dict[str, float | int]]:
    metric_names = sorted({key for record in records for key in record["metrics"]})
    result = {}
    for name in metric_names:
        values = np.asarray(
            [record["metrics"].get(name, float("nan")) for record in records],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        result[name] = {
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "std": float(finite.std()) if finite.size else float("nan"),
            "count": int(finite.size),
        }
    return result


@hydra.main(config_path="../configs", config_name="eval_opentouch", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logging()
    if not cfg.get("checkpoint_path"):
        raise ValueError("checkpoint_path is required.")
    checkpoint_path = Path(str(cfg.checkpoint_path))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    split = str(cfg.get("split", "val"))
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    data_cfg = cfg.data.val if split == "val" and cfg.data.get("val") is not None else cfg.data.train
    dataset = instantiate(data_cfg, split=split)
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    dtype = _mixed_precision_to_model_dtype(
        _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    )
    model = instantiate(
        cfg.model,
        model_dtype=dtype,
        device=device,
        skip_dit_load_from_pretrain=True,
    )
    if bool(dataset.include_force) != (getattr(model, "force_expert", None) is not None):
        raise ValueError("OpenTouch dataset/model expert modes do not match.")
    artifact_dir = cfg.get("data_artifact_dir") or str(checkpoint_path.with_suffix(""))
    dataset.validate_data_artifacts(artifact_dir)
    payload = model.load_checkpoint(
        checkpoint_path,
        data_artifact_dir=artifact_dir,
        strict_experiment_mode=True,
    )
    model.eval()

    total = len(dataset)
    requested = cfg.get("num_samples")
    if requested is None:
        indices = np.arange(total, dtype=np.int64)
    else:
        count = min(int(requested), total)
        indices = np.random.default_rng(int(cfg.get("sample_seed", 42))).choice(
            total, size=count, replace=False
        )
    records = []
    for ordinal, index in enumerate(indices.tolist()):
        source, demo, anchor = dataset.windows[index]
        result = infer_opentouch_sample(
            model=model,
            dataset=dataset,
            sample=dataset[index],
            num_inference_steps=int(cfg.get("num_inference_steps", 10)),
            seed=int(cfg.get("sample_seed", 42)) + int(index),
        )
        records.append(
            {
                "dataset_index": int(index),
                "source": source.name,
                "demo": demo,
                "anchor": int(anchor),
                "metrics": {key: float(value) for key, value in result["metrics"].items()},
            }
        )
        logger.info("Evaluated %d/%d: index=%d", ordinal + 1, len(indices), index)

    output = {
        "experiment_mode": getattr(model, "experiment_mode", None),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_step": payload.get("global_step", payload.get("step")),
        "joint_step": payload.get("joint_step"),
        "initialization_mode": getattr(model, "initialization_mode", "unknown_legacy"),
        "initialization_checkpoint": getattr(model, "initialization_checkpoint", None),
        "split": split,
        "num_inference_steps": int(cfg.get("num_inference_steps", 10)),
        "sample_seed": int(cfg.get("sample_seed", 42)),
        "dataset_indices": [int(index) for index in indices.tolist()],
        "config": OmegaConf.to_container(cfg, resolve=True),
        "samples": records,
        "aggregate": _aggregate(records),
    }
    output_dir = Path(str(cfg.get("output_dir", "./eval/opentouch")))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / str(cfg.get("output_json", "metrics.json"))
    output_path.write_text(json.dumps(output, indent=2, allow_nan=True), encoding="utf-8")
    logger.info("Saved OpenTouch evaluation to %s", output_path)


if __name__ == "__main__":
    main()
