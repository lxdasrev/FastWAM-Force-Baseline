from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any

# HDF5 file locks are unreliable for read-only files accessed through \\wsl$.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .retarget import retarget_hand_sequence
from .contract import write_artifacts, verify_artifact_payloads


class OpenTouchForceDataset(Dataset):
    """FastWAM windows built directly from OpenTouch raw HDF5 files.

    Numeric retargeting is cached per episode. RGB JPEG frames remain in the
    source HDF5 and are decoded only for the nine frames used by FastWAM.
    """

    PROCESSING_VERSION = 3
    DATA_CONTRACT_VERSION = 1

    def __init__(
        self,
        raw_data_root: str,
        cache_root: str,
        split: str = "train",
        split_seed: int = 42,
        split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        video_size: tuple[int, int] = (224, 224),
        history_steps: int = 16,
        future_steps: int = 32,
        video_stride: int = 4,
        pressure_baseline: float = 3072.0,
        force_deadzone: float = 0.0,
        force_asinh_scale: float = 256.0,
        force_clip: float = 5.0,
        workspace_scale_m: float = 0.30,
        smoothing_window: int = 5,
        max_delta_translation_m: float = 0.05,
        max_delta_rotation_rad: float = 0.50,
        compute_force_stats: bool = True,
        normalize_action_proprio: bool = True,
        strict_schema: bool = True,
        prompt_template: str = "Manipulate the object using visual and tactile feedback: {task}",
        sample_rate_hz: float = 30.0,
        gripper_closed_ratio: float = 0.18,
        gripper_open_ratio: float = 1.35,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {split!r}")
        if len(split_ratios) != 3 or not np.isclose(sum(split_ratios), 1.0):
            raise ValueError("`split_ratios` must contain train/val/test ratios summing to 1.")
        self.raw_data_root = Path(raw_data_root).expanduser()
        self.cache_root = Path(cache_root).expanduser()
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.split_seed = int(split_seed)
        self.split_ratios = tuple(float(v) for v in split_ratios)
        self.video_size = tuple(int(v) for v in video_size)
        self.history_steps = int(history_steps)
        self.future_steps = int(future_steps)
        self.total_force_steps = self.history_steps + 1 + self.future_steps
        self.video_stride = int(video_stride)
        self.video_indices = list(range(0, self.future_steps + 1, self.video_stride))
        if self.video_indices[-1] != self.future_steps:
            raise ValueError("future_steps must be divisible by video_stride.")
        self.pressure_baseline = float(pressure_baseline)
        self.force_deadzone = float(force_deadzone)
        self.force_asinh_scale = float(force_asinh_scale)
        self.force_clip = float(force_clip)
        self.workspace_scale_m = float(workspace_scale_m)
        self.smoothing_window = int(smoothing_window)
        self.max_delta_translation_m = float(max_delta_translation_m)
        self.max_delta_rotation_rad = float(max_delta_rotation_rad)
        self.strict_schema = bool(strict_schema)
        self.normalize_action_proprio = bool(normalize_action_proprio)
        self.prompt_template = str(prompt_template)
        self.sample_rate_hz = float(sample_rate_hz)
        self.gripper_closed_ratio = float(gripper_closed_ratio)
        self.gripper_open_ratio = float(gripper_open_ratio)

        asset = Path(__file__).with_name("assets") / "fingertip_taxels.json"
        self.taxel_mask_path = asset
        self.taxel_mask_sha256 = hashlib.sha256(asset.read_bytes()).hexdigest()
        spec = json.loads(asset.read_text(encoding="utf-8"))
        names = list(spec["finger_names"])
        if names[:2] != ["thumb", "index"]:
            raise ValueError(f"Unexpected finger order in {asset}: {names}")
        weights = np.zeros((len(names), 16, 16), dtype=np.float32)
        for finger_idx, name in enumerate(names):
            for row, col in spec["right_pressure_taxels"][name]:
                weights[finger_idx, row, col] = 1.0
        counts = weights.sum(axis=(1, 2))
        self.finger_weights = (weights / counts[:, None, None]).reshape(len(names), 256)

        files = sorted(self.raw_data_root.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No .hdf5 files found in {self.raw_data_root}")
        self.files = files
        groups = np.asarray(sorted(path.stem for path in files), dtype=object)
        np.random.default_rng(self.split_seed).shuffle(groups)
        num_groups = len(groups)
        num_train = int(round(num_groups * self.split_ratios[0]))
        num_val = int(round(num_groups * self.split_ratios[1]))
        num_train = min(max(num_train, 1), num_groups)
        num_val = min(max(num_val, 1), max(num_groups - num_train, 0))
        self.group_to_split = {
            group: (
                "train"
                if idx < num_train
                else ("val" if idx < num_train + num_val else "test")
            )
            for idx, group in enumerate(groups.tolist())
        }
        self.episodes: list[tuple[Path, str, int]] = []
        all_episodes: list[tuple[Path, str, int]] = []
        required = {"right_pressure", "right_hand_landmarks"}
        for path in files:
            with h5py.File(path, "r") as handle:
                if "data" not in handle:
                    if self.strict_schema:
                        raise KeyError(f"{path} has no `data` group")
                    continue
                for demo_name in sorted(handle["data"].keys()):
                    demo = handle["data"][demo_name]
                    missing = required - set(demo.keys())
                    image_key = self._find_image_key(demo)
                    if missing or image_key is None:
                        if self.strict_schema:
                            raise KeyError(
                                f"{path.name}/{demo_name} missing {sorted(missing)} or RGB image sequence"
                            )
                        continue
                    frames = int(demo["right_hand_landmarks"].shape[0])
                    aligned_keys = ["right_pressure", image_key]
                    mismatched = [key for key in aligned_keys if int(demo[key].shape[0]) != frames]
                    if mismatched:
                        if self.strict_schema:
                            raise ValueError(
                                f"{path.name}/{demo_name} has non-aligned frame counts for {mismatched}"
                            )
                        continue
                    if frames >= self.history_steps + self.future_steps + 1:
                        all_episodes.append((path, demo_name, frames))
        self._all_episodes = all_episodes
        self.episodes = [ep for ep in all_episodes if self._group_split(ep[0].stem) == self.split]
        self.force_median, self.force_scale = self._load_or_compute_force_stats(compute_force_stats)
        (
            self.action_mean,
            self.action_scale,
            self.proprio_mean,
            self.proprio_scale,
        ) = self._load_or_compute_robot_stats()

        self.windows: list[tuple[Path, str, int]] = []
        for path, demo_name, frames in self.episodes:
            # anchor is t0; t0-16 ... t0+32 must all exist.
            for anchor in range(self.history_steps, frames - self.future_steps):
                self.windows.append((path, demo_name, anchor))
        if not self.windows:
            raise RuntimeError(f"OpenTouch split {split!r} contains no valid windows")
        self._contact_window_cache: dict[int, int] = {}

    def _group_split(self, group: str) -> str:
        return self.group_to_split[group]

    @staticmethod
    def _find_image_key(demo: h5py.Group) -> str | None:
        for key in ("rgb_images_jpeg", "rgb_images", "images", "image"):
            if key in demo:
                return key
        return None

    def _pressure_to_two_finger(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 3 or raw.shape[1:] != (16, 16):
            raise ValueError(f"Expected pressure [T,16,16], got {raw.shape}")
        cleaned = np.nan_to_num(
            raw, nan=self.pressure_baseline, posinf=self.pressure_baseline, neginf=0.0
        )
        corrected = np.maximum(self.pressure_baseline - cleaned, 0.0)
        corrected[cleaned <= 0] = 0.0
        five = corrected.reshape(len(raw), 256) @ self.finger_weights.T
        two = five[:, :2]
        transformed = np.arcsinh(np.maximum(two - self.force_deadzone, 0.0) / self.force_asinh_scale)
        raw_valid = np.isfinite(raw) & (raw > 0)
        valid = np.zeros((len(raw), 2), dtype=bool)
        for finger_idx in range(2):
            mask = self.finger_weights[finger_idx].reshape(16, 16) > 0
            valid[:, finger_idx] = raw_valid[:, mask].any(axis=1)
        return transformed.astype(np.float32), valid.astype(bool)

    def _stats_path(self) -> Path:
        signature = {
            "files": [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in self.files],
            "seed": self.split_seed,
            "ratios": self.split_ratios,
            "split_version": 2,
            "processing_version": self.PROCESSING_VERSION,
            "baseline": self.pressure_baseline,
            "deadzone": self.force_deadzone,
            "asinh_scale": self.force_asinh_scale,
            "taxel_mask_sha256": self.taxel_mask_sha256,
        }
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_root / f"force_stats_{digest}.json"

    def _load_or_compute_force_stats(self, enabled: bool) -> tuple[np.ndarray, np.ndarray]:
        if not enabled:
            return np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32)
        path = self._stats_path()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return np.asarray(payload["median"], dtype=np.float32), np.asarray(payload["scale"], dtype=np.float32)
        chunks = []
        train_episodes = [ep for ep in self._all_episodes if self._group_split(ep[0].stem) == "train"]
        for source, demo_name, _ in train_episodes:
            with h5py.File(source, "r") as handle:
                values, valid = self._pressure_to_two_finger(handle["data"][demo_name]["right_pressure"][()])
            chunks.append(np.where(valid, values, np.nan))
        merged = np.concatenate(chunks, axis=0)
        median = np.nanmedian(merged, axis=0).astype(np.float32)
        mad = np.nanmedian(np.abs(merged - median), axis=0)
        scale = np.maximum(1.4826 * mad, 1e-4).astype(np.float32)
        path.write_text(
            json.dumps({"median": median.tolist(), "scale": scale.tolist()}, indent=2),
            encoding="utf-8",
        )
        return median, scale

    def _episode_cache_path(self, source: Path, demo_name: str) -> Path:
        signature = (
            f"{source.resolve()}:{source.stat().st_mtime_ns}:{demo_name}:"
            f"{self.workspace_scale_m}:{self.smoothing_window}:"
            f"{self.max_delta_translation_m}:{self.max_delta_rotation_rad}:"
            f"{self.PROCESSING_VERSION}:{self.gripper_closed_ratio}:{self.gripper_open_ratio}:"
            f"{self.pressure_baseline}:{self.force_deadzone}:{self.force_asinh_scale}:{self.force_clip}:"
            f"{self.taxel_mask_sha256}:"
            f"{self.force_median.tolist()}:{self.force_scale.tolist()}"
        )
        digest = hashlib.sha256(signature.encode()).hexdigest()
        return self.cache_root / "episodes" / f"{digest}.npz"

    def _robot_stats_path(self) -> Path:
        signature = {
            "episodes": [(p.name, d, n) for p, d, n in self._all_episodes],
            "seed": self.split_seed,
            "ratios": self.split_ratios,
            "split_version": 2,
            "processing_version": self.PROCESSING_VERSION,
            "workspace_scale_m": self.workspace_scale_m,
            "smoothing_window": self.smoothing_window,
            "max_delta_translation_m": self.max_delta_translation_m,
            "max_delta_rotation_rad": self.max_delta_rotation_rad,
            "gripper_closed_ratio": self.gripper_closed_ratio,
            "gripper_open_ratio": self.gripper_open_ratio,
        }
        digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache_root / f"robot_stats_{digest}.json"

    def _load_or_compute_robot_stats(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.normalize_action_proprio:
            return (
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
                np.zeros(8, dtype=np.float32),
                np.ones(8, dtype=np.float32),
            )
        path = self._robot_stats_path()
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return tuple(
                np.asarray(payload[key], dtype=np.float32)
                for key in ("action_mean", "action_scale", "proprio_mean", "proprio_scale")
            )
        actions = []
        proprios = []
        for source, demo_name, _ in self._all_episodes:
            if self._group_split(source.stem) != "train":
                continue
            episode = self._load_numeric_episode(source, demo_name)
            actions.append(episode["action"][episode["action_valid"]])
            proprios.append(episode["proprio"][episode["pose_valid"]])
        action_values = np.concatenate(actions, axis=0)
        proprio_values = np.concatenate(proprios, axis=0)
        action_mean = action_values.mean(axis=0).astype(np.float32)
        action_scale = np.maximum(action_values.std(axis=0), 1e-4).astype(np.float32)
        proprio_mean = proprio_values.mean(axis=0).astype(np.float32)
        proprio_scale = np.maximum(proprio_values.std(axis=0), 1e-4).astype(np.float32)
        payload = {
            "action_mean": action_mean.tolist(),
            "action_scale": action_scale.tolist(),
            "proprio_mean": proprio_mean.tolist(),
            "proprio_scale": proprio_scale.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return action_mean, action_scale, proprio_mean, proprio_scale

    def _load_numeric_episode(self, source: Path, demo_name: str) -> dict[str, np.ndarray]:
        cache_path = self._episode_cache_path(source, demo_name)
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as payload:
                return {key: payload[key] for key in payload.files}
        with h5py.File(source, "r") as handle:
            demo = handle["data"][demo_name]
            landmarks = demo["right_hand_landmarks"][()]
            external = None
            for key in ("right_palm_pos", "right_wrist_pos"):
                if key in demo and tuple(demo[key].shape) == (len(landmarks), 3):
                    external = demo[key][()]
                    break
            retargeted = retarget_hand_sequence(
                landmarks,
                external,
                workspace_scale_m=self.workspace_scale_m,
                smoothing_window=self.smoothing_window,
                gripper_closed_ratio=self.gripper_closed_ratio,
                gripper_open_ratio=self.gripper_open_ratio,
            )
            if "timestamps" in demo and len(demo["timestamps"]) == len(landmarks):
                timestamps = np.asarray(demo["timestamps"][()], dtype=np.float64)
                dt = np.diff(timestamps)
                positive = dt[np.isfinite(dt) & (dt > 0)]
                if positive.size:
                    max_dt = 3.0 * float(np.median(positive))
                    retargeted["action_valid"] &= np.isfinite(dt) & (dt > 0) & (dt <= max_dt)
            retargeted["action_valid"] &= (
                np.linalg.norm(retargeted["action"][:, :3], axis=1) <= self.max_delta_translation_m
            )
            retargeted["action_valid"] &= (
                np.linalg.norm(retargeted["action"][:, 3:6], axis=1) <= self.max_delta_rotation_rad
            )
            force, force_valid = self._pressure_to_two_finger(demo["right_pressure"][()])
        force = np.clip(
            (force - self.force_median[None]) / self.force_scale[None],
            -self.force_clip,
            self.force_clip,
        ).astype(np.float32)
        payload = {
            "action": retargeted["action"],
            "action_valid": retargeted["action_valid"],
            "proprio": retargeted["proprio"],
            "pose_valid": retargeted["pose_valid"],
            "force": force,
            "force_valid": force_valid,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.npz")
        np.savez_compressed(tmp_path, **payload)
        try:
            os.replace(tmp_path, cache_path)
        except FileExistsError:
            tmp_path.unlink(missing_ok=True)
        return payload

    def _decode_image(self, item: Any) -> Image.Image:
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        array = np.asarray(item)
        if array.ndim == 3:
            if array.shape[0] == 3 and array.shape[-1] != 3:
                array = np.moveaxis(array, 0, -1)
            return Image.fromarray(array.astype(np.uint8)).convert("RGB")
        if isinstance(item, (bytes, bytearray, np.bytes_, np.void)):
            encoded = bytes(item)
        else:
            encoded = array.tobytes()
        return Image.open(io.BytesIO(encoded)).convert("RGB")

    def _load_video(self, source: Path, demo_name: str, indices: list[int]) -> torch.Tensor:
        frames = []
        with h5py.File(source, "r") as handle:
            demo = handle["data"][demo_name]
            key = self._find_image_key(demo)
            dataset = demo[key]
            for index in indices:
                image = self._decode_image(dataset[index])
                image = image.resize((self.video_size[1], self.video_size[0]), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
                frames.append(torch.from_numpy(array).permute(2, 0, 1))
        return torch.stack(frames, dim=1)  # [C,T,H,W]

    def __len__(self) -> int:
        return len(self.windows)

    def find_contact_window(self, start_index: int = 0) -> int:
        """Return a deterministic validation window with observed valid contact."""
        start_index = int(start_index) % len(self.windows)
        if start_index in self._contact_window_cache:
            return self._contact_window_cache[start_index]
        for offset in range(len(self.windows)):
            index = (start_index + offset) % len(self.windows)
            source, demo_name, anchor = self.windows[index]
            episode = self._load_numeric_episode(source, demo_name)
            begin = anchor - self.history_steps
            stop = anchor + 1
            force = episode["force"][begin:stop]
            valid = episode["force_valid"][begin:stop]
            physical_force = self.denormalize_force(force)
            if np.any(valid & (physical_force > self.force_deadzone + 1e-6)):
                self._contact_window_cache[start_index] = index
                return index
        self._contact_window_cache[start_index] = start_index
        return start_index

    def normalize_force(self, force: np.ndarray) -> np.ndarray:
        transformed = np.arcsinh(
            np.maximum(np.asarray(force, dtype=np.float32) - self.force_deadzone, 0.0)
            / self.force_asinh_scale
        )
        return np.clip(
            (transformed - self.force_median) / self.force_scale,
            -self.force_clip,
            self.force_clip,
        ).astype(np.float32)

    def denormalize_force(self, normalized_force: np.ndarray) -> np.ndarray:
        transformed = np.asarray(normalized_force, dtype=np.float32) * self.force_scale + self.force_median
        relative = np.maximum(np.sinh(transformed) * self.force_asinh_scale, 0.0)
        return (relative + self.force_deadzone).astype(np.float32)

    def normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        return ((np.asarray(proprio) - self.proprio_mean) / self.proprio_scale).astype(np.float32)

    def denormalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        return (np.asarray(proprio) * self.proprio_scale + self.proprio_mean).astype(np.float32)

    def denormalize_action(self, action: np.ndarray) -> np.ndarray:
        return (np.asarray(action) * self.action_scale + self.action_mean).astype(np.float32)

    def _git_commit(self) -> str:
        repository = Path(__file__).resolve()
        for parent in repository.parents:
            if (parent / ".git").exists():
                try:
                    result = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=parent,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    return result.stdout.strip()
                except (OSError, subprocess.SubprocessError):
                    break
        return "unknown"

    def _source_file_records(self) -> list[dict[str, Any]]:
        hash_cache_path = self.cache_root / "source_file_hashes.json"
        if hash_cache_path.exists():
            try:
                cached = json.loads(hash_cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = {}
        else:
            cached = {}
        records = []
        updated_cache: dict[str, Any] = {}
        for path in self.files:
            stat = path.stat()
            key = str(path.resolve())
            entry = cached.get(key, {})
            if (
                entry.get("size_bytes") == stat.st_size
                and entry.get("mtime_ns") == stat.st_mtime_ns
                and isinstance(entry.get("sha256"), str)
            ):
                digest = entry["sha256"]
            else:
                hasher = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
            updated_cache[key] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": digest,
            }
            records.append(
                {
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "sha256": digest,
                    "split": self._group_split(path.stem),
                }
            )
        temporary = hash_cache_path.with_name(
            f"{hash_cache_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(
            json.dumps(updated_cache, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, hash_cache_path)
        return records

    def normalization_payload(self) -> dict[str, np.ndarray]:
        return {
            "action_mean": self.action_mean.astype(np.float32),
            "action_scale": self.action_scale.astype(np.float32),
            "proprio_mean": self.proprio_mean.astype(np.float32),
            "proprio_scale": self.proprio_scale.astype(np.float32),
            "force_median": self.force_median.astype(np.float32),
            "force_scale": self.force_scale.astype(np.float32),
        }

    def data_contract_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.DATA_CONTRACT_VERSION,
            "processing_version": self.PROCESSING_VERSION,
            "code_commit": self._git_commit(),
            "sample_rate_hz": self.sample_rate_hz,
            "timeline": {
                "video_offsets": self.video_indices,
                "force_offsets": list(range(-self.history_steps, self.future_steps + 1)),
                "action_offsets": list(range(1, self.future_steps + 1)),
            },
            "coordinate_system": {
                "name": "sequence_initial_palm_canonical_frame",
                "position": "p_C[t] = R_0^T @ (p[t] - p[0]) * workspace_scale / hand_scale",
                "orientation": "R_C[t] = R_0^T @ R[t]",
            },
            "proprio": {
                "shape": [1, 8],
                "dimensions": [
                    "position_x_m_canonical",
                    "position_y_m_canonical",
                    "position_z_m_canonical",
                    "orientation_qw_canonical",
                    "orientation_qx_canonical",
                    "orientation_qy_canonical",
                    "orientation_qz_canonical",
                    "absolute_gripper_opening_0_1",
                ],
            },
            "action_chunk": {
                "shape": [self.future_steps, 7],
                "dimensions": [
                    "delta_position_x_m_canonical",
                    "delta_position_y_m_canonical",
                    "delta_position_z_m_canonical",
                    "delta_rotation_x_rad_canonical_spatial_rotvec",
                    "delta_rotation_y_rad_canonical_spatial_rotvec",
                    "delta_rotation_z_rad_canonical_spatial_rotvec",
                    "next_absolute_gripper_opening_0_1",
                ],
                "note": "Only dimensions 0:6 are deltas; dimension 6 is an absolute next target.",
            },
            "force": {
                "shape": [self.total_force_steps, 2],
                "finger_order": ["thumb", "index"],
                "pressure_baseline": self.pressure_baseline,
                "deadzone": self.force_deadzone,
                "asinh_scale": self.force_asinh_scale,
                "normalized_clip": self.force_clip,
                "taxel_mask_sha256": self.taxel_mask_sha256,
            },
            "retarget": {
                "workspace_scale_m": self.workspace_scale_m,
                "smoothing_window": self.smoothing_window,
                "gripper_closed_ratio": self.gripper_closed_ratio,
                "gripper_open_ratio": self.gripper_open_ratio,
                "max_delta_translation_m": self.max_delta_translation_m,
                "max_delta_rotation_rad": self.max_delta_rotation_rad,
            },
        }

    def split_manifest_payload(self) -> dict[str, Any]:
        episode_records = [
            {
                "source": source.name,
                "episode": demo_name,
                "frames": frames,
                "split": self._group_split(source.stem),
            }
            for source, demo_name, frames in self._all_episodes
        ]
        return {
            "schema_version": 1,
            "split_seed": self.split_seed,
            "split_ratios": list(self.split_ratios),
            "source_files": self._source_file_records(),
            "episodes": episode_records,
        }

    def export_data_artifacts(self, directory: str | Path) -> dict[str, str]:
        return write_artifacts(
            directory,
            normalization=self.normalization_payload(),
            data_contract=self.data_contract_payload(),
            split_manifest=self.split_manifest_payload(),
        )

    def validate_data_artifacts(self, directory: str | Path) -> None:
        verify_artifact_payloads(
            directory,
            normalization=self.normalization_payload(),
            data_contract=self.data_contract_payload(),
            split_manifest=self.split_manifest_payload(),
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, demo_name, anchor = self.windows[index]
        episode = self._load_numeric_episode(source, demo_name)
        force_start = anchor - self.history_steps
        force_stop = anchor + self.future_steps + 1
        video_indices = [anchor + offset for offset in self.video_indices]
        action = episode["action"][anchor : anchor + self.future_steps]
        action_valid = episode["action_valid"][anchor : anchor + self.future_steps]
        force = episode["force"][force_start:force_stop]
        force_valid = episode["force_valid"][force_start:force_stop]
        if action.shape != (self.future_steps, 7) or force.shape != (self.total_force_steps, 2):
            raise RuntimeError("Internal OpenTouch window alignment failure.")
        task = f"{source.stem} {demo_name}".replace("_", " ")
        action = ((action - self.action_mean) / self.action_scale).astype(np.float32)
        proprio = (
            (episode["proprio"][anchor : anchor + 1] - self.proprio_mean) / self.proprio_scale
        ).astype(np.float32)
        return {
            "video": self._load_video(source, demo_name, video_indices),
            "action": torch.from_numpy(action),
            "proprio": torch.from_numpy(proprio),
            "force": torch.from_numpy(force),
            "force_valid": torch.from_numpy(force_valid),
            "force_is_pad": torch.zeros(self.total_force_steps, dtype=torch.bool),
            "image_is_pad": torch.zeros(len(self.video_indices), dtype=torch.bool),
            "action_is_pad": torch.from_numpy(~action_valid),
            "proprio_is_pad": torch.tensor([not bool(episode["pose_valid"][anchor])]),
            "prompt": self.prompt_template.format(task=task),
        }
