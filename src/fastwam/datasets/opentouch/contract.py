from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ARTIFACT_FILENAMES = (
    "normalization.npz",
    "data_contract.json",
    "split_manifest.json",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes(directory: str | Path) -> dict[str, str]:
    directory = Path(directory)
    missing = [name for name in ARTIFACT_FILENAMES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Data artifact directory {directory} is missing required files: {missing}"
        )
    return {name: sha256_file(directory / name) for name in ARTIFACT_FILENAMES}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_artifacts(
    directory: str | Path,
    *,
    normalization: Mapping[str, np.ndarray],
    data_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> dict[str, str]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    normalization_path = directory / "normalization.npz"
    temporary = normalization_path.with_name(
        f"{normalization_path.stem}.{os.getpid()}.npz"
    )
    np.savez_compressed(
        temporary,
        **{key: np.asarray(value) for key, value in normalization.items()},
    )
    os.replace(temporary, normalization_path)
    write_json(directory / "data_contract.json", data_contract)
    write_json(directory / "split_manifest.json", split_manifest)
    return artifact_hashes(directory)


def verify_artifact_hashes(
    directory: str | Path,
    expected_hashes: Mapping[str, str],
) -> dict[str, str]:
    actual = artifact_hashes(directory)
    mismatches = {
        name: {"expected": expected_hashes.get(name), "actual": digest}
        for name, digest in actual.items()
        if expected_hashes.get(name) != digest
    }
    if mismatches:
        raise ValueError(f"Checkpoint data artifact hash mismatch: {mismatches}")
    return actual


def _first_difference(expected: Any, actual: Any, path: str = "root") -> str | None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return f"{path}.{key}: unexpected field"
            if key not in actual:
                return f"{path}.{key}: missing field"
            difference = _first_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"{path}: length {len(actual)} != expected {len(expected)}"
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            difference = _first_difference(
                expected_item, actual_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{path}: {actual!r} != expected {expected!r}"
    return None


def verify_artifact_payloads(
    directory: str | Path,
    *,
    normalization: Mapping[str, np.ndarray],
    data_contract: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
) -> None:
    directory = Path(directory)
    with np.load(directory / "normalization.npz", allow_pickle=False) as payload:
        expected_keys = set(normalization)
        actual_keys = set(payload.files)
        if actual_keys != expected_keys:
            raise ValueError(
                "normalization.npz keys differ: "
                f"actual={sorted(actual_keys)}, expected={sorted(expected_keys)}"
            )
        for key, expected in normalization.items():
            actual = payload[key]
            if actual.shape != np.asarray(expected).shape or not np.array_equal(
                actual, np.asarray(expected)
            ):
                raise ValueError(f"normalization.npz field {key!r} differs from current dataset")

    actual_contract = json.loads((directory / "data_contract.json").read_text(encoding="utf-8"))
    difference = _first_difference(dict(data_contract), actual_contract, "data_contract")
    if difference is not None:
        raise ValueError(f"Checkpoint data contract differs from current dataset at {difference}")

    actual_split = json.loads((directory / "split_manifest.json").read_text(encoding="utf-8"))
    difference = _first_difference(dict(split_manifest), actual_split, "split_manifest")
    if difference is not None:
        raise ValueError(f"Checkpoint split manifest differs from current dataset at {difference}")
