"""Acceptance tests for TASK-03 — ICIP 2026 camera-ready config grid.

These tests enforce the five invariants from
`docs/icip2026/rebuttal_plans/TASK_03_ORCH_training_orchestration.md`:

    Test 1: All 6 configs exist and parse.
    Test 2: 11 critical hyperparameters identical across all 6 configs.
    Test 3: For each fold, shared vs decoupled configs differ only in the
            allowed fields (model.bottleneck_mode and per-cell experiment
            metadata / wandb tags that encode the arch or fold label).
    Test 4: Every SLURM script passes `bash -n` and carries the executable bit.
    Test 5: The 6 configs have 6 distinct `experiment.output_dir` values.

Tests 1, 2, 3, 5 are file-level Python checks. Test 4 is exercised via a
subprocess syntax check on each `train_generate.sh`.

The tests deliberately do NOT require `slimdiff` to be importable; they
only exercise YAML parsing via OmegaConf.
"""

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
CR_DIR = REPO_ROOT / "slurm" / "camera_ready"

ARCHS = ("shared", "decoupled")
FOLDS = (0, 1, 2)


def _cell_dir(arch: str, fold: int) -> Path:
    return CR_DIR / f"{arch}_fold_{fold}"


def _load(arch: str, fold: int) -> DictConfig:
    cfg_path = _cell_dir(arch, fold) / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    assert isinstance(cfg, DictConfig)
    return cfg


def _get_nested(cfg: DictConfig, dotted: str):
    node = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


# -----------------------------------------------------------------------------
# Test 1 — all configs exist and parse
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("fold", FOLDS)
def test_config_exists_and_parses(arch: str, fold: int):
    cfg_path = _cell_dir(arch, fold) / "config.yaml"
    assert cfg_path.is_file(), f"Missing config: {cfg_path}"
    cfg = OmegaConf.load(cfg_path)
    assert isinstance(cfg, DictConfig), f"Config did not parse as mapping: {cfg_path}"


# -----------------------------------------------------------------------------
# Test 2 — hyperparameter consistency
# -----------------------------------------------------------------------------


MUST_MATCH_KEYS = (
    "scheduler.prediction_type",
    "scheduler.schedule",
    "scheduler.num_train_timesteps",
    "training.optimizer.lr",
    "training.optimizer.weight_decay",
    "training.early_stopping.patience",
    "training.ema.decay",
    "training.gradient_clip_val",
    "loss.lp_norm.p",
    "sampler.num_inference_steps",
    "sampler.eta",
)


@pytest.mark.parametrize("key", MUST_MATCH_KEYS)
def test_hyperparameter_consistency(key: str):
    """All 6 configs share identical training hyperparameters."""
    values = []
    for arch in ARCHS:
        for fold in FOLDS:
            cfg = _load(arch, fold)
            values.append(str(_get_nested(cfg, key)))
    assert len(set(values)) == 1, (
        f"Hyperparameter {key!r} differs across configs: {values}"
    )


# -----------------------------------------------------------------------------
# Test 3 — arch configs for the same fold differ only in allowed fields
# -----------------------------------------------------------------------------


# Fields allowed to differ between shared and decoupled cells for the same fold.
# Everything else must be byte-identical after these are stripped.
_ALLOWED_ARCH_DIFFS = (
    "model.bottleneck_mode",
    "experiment.name",
    "experiment.output_dir",
    "logging.logger.wandb.name",
    "logging.logger.wandb.tags",
    "logging.logger.wandb.notes",
)


def _pop_nested(cfg: DictConfig, dotted: str) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node:
            return
        node = node[part]
    if parts[-1] in node:
        del node[parts[-1]]


@pytest.mark.parametrize("fold", FOLDS)
def test_arch_configs_minimal_diff(fold: int):
    """Shared vs decoupled configs for the same fold differ only in the
    declared arch-specific fields (bottleneck_mode and human-readable
    run labels)."""
    shared = _load("shared", fold)
    decoupled = _load("decoupled", fold)

    assert shared.model.bottleneck_mode == "shared"
    assert decoupled.model.bottleneck_mode == "decoupled"

    for key in _ALLOWED_ARCH_DIFFS:
        _pop_nested(shared, key)
        _pop_nested(decoupled, key)

    assert OmegaConf.to_yaml(shared) == OmegaConf.to_yaml(decoupled), (
        f"Configs for shared_fold_{fold} and decoupled_fold_{fold} differ "
        "in fields outside the allowed arch-specific set."
    )


# -----------------------------------------------------------------------------
# Test 4 — SLURM scripts are executable and pass syntax check
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("fold", FOLDS)
def test_slurm_script_syntax(arch: str, fold: int):
    script_path = _cell_dir(arch, fold) / "train_generate.sh"
    assert script_path.is_file(), f"Missing SLURM script: {script_path}"
    # Executable bit
    mode = script_path.stat().st_mode
    assert mode & stat.S_IXUSR, f"Not executable: {script_path}"
    # Bash syntax check
    result = subprocess.run(
        ["bash", "-n", str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Syntax error in {script_path}:\n{result.stderr}"
    )


def test_launcher_script_syntax():
    launcher = CR_DIR / "launch_camera_ready.sh"
    assert launcher.is_file(), f"Missing launcher: {launcher}"
    mode = launcher.stat().st_mode
    assert mode & stat.S_IXUSR, f"Not executable: {launcher}"
    result = subprocess.run(
        ["bash", "-n", str(launcher)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Syntax error in {launcher}:\n{result.stderr}"
    )


# -----------------------------------------------------------------------------
# Test 5 — unique output_dir values across all 6 cells
# -----------------------------------------------------------------------------


def test_output_dirs_unique():
    """Each of the 6 configs points to a unique output directory."""
    dirs: list[str] = []
    for arch in ARCHS:
        for fold in FOLDS:
            cfg = _load(arch, fold)
            dirs.append(str(cfg.experiment.output_dir))
    assert len(set(dirs)) == len(dirs), (
        f"Duplicate output_dir values across cells: {dirs}"
    )


# -----------------------------------------------------------------------------
# Bonus — bottleneck_mode exists and is a valid enum value
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("fold", FOLDS)
def test_bottleneck_mode_present(arch: str, fold: int):
    cfg = _load(arch, fold)
    assert "bottleneck_mode" in cfg.model
    assert cfg.model.bottleneck_mode == arch


# -----------------------------------------------------------------------------
# Bonus — SEED_BASE value in each SLURM script is 42 (paired-sample guarantee)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("fold", FOLDS)
def test_seed_base_is_42(arch: str, fold: int):
    script = (_cell_dir(arch, fold) / "train_generate.sh").read_text()
    assert "SEED_BASE=42" in script, (
        f"{arch}_fold_{fold}/train_generate.sh must use SEED_BASE=42 "
        "so generation noise is paired across architectures."
    )
