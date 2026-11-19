from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn

from fastwam.datasets.opentouch import OpenTouchDataset
from fastwam.datasets.opentouch.metrics import compute_opentouch_action_metrics
from fastwam.models.wan22.fastwam import FastWAM


def _base_hand() -> np.ndarray:
    hand = np.zeros((21, 3), dtype=np.float32)
    hand[5] = [0.05, 0.10, 0.0]
    hand[9] = [0.0, 0.11, 0.0]
    hand[17] = [-0.05, 0.10, 0.0]
    hand[4] = [-0.025, 0.16, 0.0]
    hand[8] = [0.025, 0.16, 0.0]
    for index in range(1, 21):
        if not hand[index].any():
            hand[index] = [0.001 * index, 0.05 + 0.002 * index, 0.001]
    return hand


def _write_episode(path: Path, *, include_pressure: bool) -> None:
    frames = 50
    hands = np.repeat(_base_hand()[None], frames, axis=0)
    hands[:, :, 0] += np.arange(frames, dtype=np.float32)[:, None] * 1e-4
    images = np.zeros((frames, 16, 16, 3), dtype=np.uint8)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.create_dataset("right_hand_landmarks", data=hands)
        demo.create_dataset("rgb_images", data=images)
        if include_pressure:
            demo.create_dataset(
                "right_pressure", data=np.full((frames, 16, 16), 3000.0, dtype=np.float32)
            )


def _dataset(root: Path, cache: Path, *, include_force: bool) -> OpenTouchDataset:
    return OpenTouchDataset(
        raw_data_root=str(root),
        cache_root=str(cache),
        split="train",
        include_force=include_force,
        compute_force_stats=False,
        normalize_action_proprio=False,
        smoothing_window=1,
    )


def test_baseline_does_not_require_or_emit_pressure(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_episode(data_root / "task.hdf5", include_pressure=False)
    dataset = _dataset(data_root, tmp_path / "cache", include_force=False)
    sample = dataset[0]
    assert dataset.taxel_mask_path is None
    assert set(sample).isdisjoint({"force", "force_valid", "force_is_pad"})
    assert set(dataset.normalization_payload()).isdisjoint({"force_median", "force_scale"})
    assert "force" not in dataset.data_contract_payload()
    assert sample["prompt"].startswith("Manipulate the object:")
    assert not list((tmp_path / "cache").glob("force_stats_*.json"))
    episode_cache = list((tmp_path / "cache" / "episodes").glob("*.npz"))
    assert episode_cache
    with np.load(episode_cache[0], allow_pickle=False) as payload:
        assert set(payload.files).isdisjoint({"force", "force_valid"})


def test_force_and_baseline_share_windows_but_keep_mode_specific_payloads(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_episode(data_root / "task.hdf5", include_pressure=True)
    baseline = _dataset(data_root, tmp_path / "cache", include_force=False)
    force = _dataset(data_root, tmp_path / "cache", include_force=True)
    assert baseline.windows == force.windows
    assert baseline.split_manifest_payload() == force.split_manifest_payload()
    assert "force" not in baseline[0]
    assert force[0]["force"].shape == (49, 2)
    assert "force_median" in force.normalization_payload()


def test_force_mode_still_requires_pressure(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_episode(data_root / "task.hdf5", include_pressure=False)
    with pytest.raises(KeyError, match="right_pressure"):
        _dataset(data_root, tmp_path / "cache", include_force=True)


class _TinyMoT(nn.Module):
    def __init__(self, force_expert: nn.Module | None = None):
        super().__init__()
        self.mixtures = nn.ModuleDict(
            {"video": nn.Linear(2, 2), "action": nn.Linear(2, 2)}
        )
        self.force_expert = force_expert
        self.force_va_gates = nn.ParameterDict(
            {} if force_expert is None else {"0": nn.Parameter(torch.zeros(()))}
        )
        self.compile_training_layers = False


def _tiny_model(*, force: bool) -> FastWAM:
    force_expert = nn.Linear(2, 2) if force else None
    mot = _TinyMoT(force_expert)
    return FastWAM(
        video_expert=mot.mixtures["video"],
        action_expert=mot.mixtures["action"],
        force_expert=force_expert,
        mot=mot,
        vae=nn.Identity(),
        text_dim=4,
        proprio_dim=8,
    )


def test_baseline_model_has_no_force_scheduler():
    model = _tiny_model(force=False)
    assert model.experiment_mode == "baseline"
    assert model.train_force_scheduler is None
    assert model.infer_force_scheduler is None


def test_official_initialization_loads_only_video_action(tmp_path):
    source = _tiny_model(force=False)
    target = _tiny_model(force=True)
    source_state = source.mot.state_dict()
    proprio_before = {key: value.clone() for key, value in target.proprio_encoder.state_dict().items()}
    force_before = {key: value.clone() for key, value in target.force_expert.state_dict().items()}
    checkpoint = tmp_path / "official.pt"
    torch.save({"mot": source_state, "step": 12}, checkpoint)

    target.initialize_from_official_fastwam(checkpoint)
    target_state = target.mot.state_dict()
    for key, value in source_state.items():
        torch.testing.assert_close(target_state[key], value)
    for key, value in proprio_before.items():
        torch.testing.assert_close(target.proprio_encoder.state_dict()[key], value)
    for key, value in force_before.items():
        torch.testing.assert_close(target.force_expert.state_dict()[key], value)
    assert target.initialization_mode == "official_fastwam"
    assert target.initialization_source_step == 12


def test_strict_checkpoint_mode_rejects_cross_mode_load(tmp_path):
    baseline = _tiny_model(force=False)
    force = _tiny_model(force=True)
    checkpoint = tmp_path / "baseline.pt"
    baseline.save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="experiment_mode"):
        force.load_checkpoint(checkpoint, strict_experiment_mode=True)


def test_official_initialization_rejects_force_checkpoint(tmp_path):
    source = _tiny_model(force=True)
    target = _tiny_model(force=True)
    checkpoint = tmp_path / "force.pt"
    source.save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="force experiment"):
        target.initialize_from_official_fastwam(checkpoint)


def test_action_only_metrics_respect_step_mask():
    gt = np.zeros((2, 7), dtype=np.float32)
    pred = np.ones((2, 7), dtype=np.float32)
    pred[1] = 100.0
    metrics = compute_opentouch_action_metrics(
        pred_action=pred,
        gt_action=gt,
        action_valid=np.asarray([True, False]),
    )
    assert metrics["action_mae"] == pytest.approx(1.0)
    assert metrics["action_rmse"] == pytest.approx(1.0)
