import json

import numpy as np

from fastwam.datasets.opentouch.contract import (
    verify_artifact_hashes,
    verify_artifact_payloads,
    write_artifacts,
)


def test_data_artifact_round_trip_and_hash_binding(tmp_path):
    normalization = {
        "action_mean": np.zeros(7, dtype=np.float32),
        "action_scale": np.ones(7, dtype=np.float32),
    }
    contract = {"schema_version": 1, "frame": "canonical"}
    split = {"split_seed": 42, "source_files": []}
    hashes = write_artifacts(
        tmp_path,
        normalization=normalization,
        data_contract=contract,
        split_manifest=split,
    )
    verify_artifact_hashes(tmp_path, hashes)
    verify_artifact_payloads(
        tmp_path,
        normalization=normalization,
        data_contract=contract,
        split_manifest=split,
    )

    (tmp_path / "data_contract.json").write_text(
        json.dumps({"schema_version": 1, "frame": "body"}), encoding="utf-8"
    )
    try:
        verify_artifact_hashes(tmp_path, hashes)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("Corrupted data contract should fail hash verification")
