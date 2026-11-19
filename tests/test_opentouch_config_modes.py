from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_DIR = str((Path(__file__).resolve().parents[1] / "configs").resolve())


def _compose(config_name: str, *overrides: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))


def test_four_training_mode_initialization_combinations_compose():
    for task, has_force in (
        ("opentouch_baseline_224_1e-4", False),
        ("opentouch_force_224_1e-4", True),
    ):
        for initialization in ("wan_actiondit", "official_fastwam"):
            cfg = _compose(
                "train",
                f"task={task}",
                f"initialization={initialization}",
            )
            assert bool(cfg.data.train.include_force) is has_force
            assert (cfg.model.force_dit_config is not None) is has_force
            assert cfg.initialization.mode == initialization
            assert cfg.max_steps == cfg.force_warmup_steps + cfg.joint_training_steps


def test_two_eval_modes_compose_with_shared_interface():
    for task, has_force in (
        ("opentouch_baseline_224_1e-4", False),
        ("opentouch_force_224_1e-4", True),
    ):
        cfg = _compose("eval_opentouch", f"task={task}")
        assert bool(cfg.data.val.include_force) is has_force
        assert (cfg.model.force_dit_config is not None) is has_force
        assert cfg.split == "val"
        assert cfg.num_samples == 100
