"""Emit shell-safe ``export KEY='value'`` lines from picasso_paths.yaml.

Invoked by every SLURM/bash script under ``slurm/camera_ready/``:

    eval "$(python _load_paths.py picasso_paths.yaml)"

Values are quoted via :func:`shlex.quote`, making the ``eval`` safe even if
paths contain whitespace. All emitted names are upper-case env vars; empty
strings are emitted verbatim (the shell treats them as unset-but-present).

Keys emitted (stable contract for downstream scripts):

- Paths: ``REPO_SRC``, ``DATA_SRC``, ``RESULTS_ROOT``, ``CACHE_DIR``,
  ``EVAL_OUTPUT_DIR``, ``POSTHOC_OUTPUT_DIR``, ``TORCH_HOME``
- Env  : ``CONDA_ENV_NAME``
- Cache: ``CACHE_CONFIG_TEMPLATE``
- Kfold: ``KFOLD_N_FOLDS``, ``KFOLD_SEED``
- Eval : ``EVAL_CONFIG_TEMPLATE``, ``EVAL_DEVICE``
- Post-hoc: ``POSTHOC_TAU_VALUES``, ``POSTHOC_MIN_LESION_SIZE_PX``,
  ``POSTHOC_SUBSET_SIZE``, ``POSTHOC_NUM_SUBSETS``, ``POSTHOC_EARLY_STOPPING_CSV``
- SLURM (build / fold_eval / posthoc):
  ``SLURM_<SECTION>_{TIME,CPUS,MEM,PARTITION,CONSTRAINT,GRES}``
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

import yaml


def _emit(key: str, value: object) -> None:
    print(f"export {key}={shlex.quote(str(value))}")


def main(yaml_path: str) -> int:
    cfg = yaml.safe_load(Path(yaml_path).read_text()) or {}

    paths = cfg.get("paths", {}) or {}
    env = cfg.get("env", {}) or {}
    cache = cfg.get("cache", {}) or {}
    kfold = cfg.get("kfold", {}) or {}
    ev = cfg.get("eval", {}) or {}
    ph = cfg.get("posthoc", {}) or {}
    slurm = cfg.get("slurm", {}) or {}

    repo_src = paths.get("repo_src")
    data_src = paths.get("data_src")
    results_root = paths.get("results_root")
    if not repo_src or not data_src or not results_root:
        print(
            "ERROR: picasso_paths.yaml must set paths.repo_src / data_src / results_root",
            file=sys.stderr,
        )
        return 1

    cache_dir = paths.get("cache_dir") or f"{data_src}/slice_cache"
    eval_output_dir = paths.get("eval_output_dir") or f"{results_root}/eval_output"
    posthoc_output_dir = (
        paths.get("posthoc_output_dir") or f"{results_root}/posthoc_output"
    )

    _emit("REPO_SRC", repo_src)
    _emit("DATA_SRC", data_src)
    _emit("RESULTS_ROOT", results_root)
    _emit("CACHE_DIR", cache_dir)
    _emit("EVAL_OUTPUT_DIR", eval_output_dir)
    _emit("POSTHOC_OUTPUT_DIR", posthoc_output_dir)

    torch_home = paths.get("torch_home") or f"{results_root}/.torch"
    _emit("TORCH_HOME", torch_home)

    _emit("CONDA_ENV_NAME", env.get("conda_env", "jsddpm"))

    _emit(
        "CACHE_CONFIG_TEMPLATE",
        cache.get("config_template", "src/diffusion/config/cache/epilepsy.yaml"),
    )

    _emit("KFOLD_N_FOLDS", kfold.get("n_folds", 3))
    _emit("KFOLD_SEED", kfold.get("seed", 42))

    _emit(
        "EVAL_CONFIG_TEMPLATE",
        ev.get(
            "config_template",
            "src/diffusion/scripts/similarity_metrics/config/icip2026_camera_ready.yaml",
        ),
    )
    _emit("EVAL_DEVICE", ev.get("device", "cuda:0"))

    _emit("POSTHOC_TAU_VALUES", ph.get("tau_values", ""))
    _emit("POSTHOC_MIN_LESION_SIZE_PX", ph.get("min_lesion_size_px", 5))
    _emit("POSTHOC_SUBSET_SIZE", ph.get("subset_size", 500))
    _emit("POSTHOC_NUM_SUBSETS", ph.get("num_subsets", 100))
    _emit("POSTHOC_EARLY_STOPPING_CSV", ph.get("early_stopping_csv", ""))

    for section in ("build", "fold_eval", "posthoc"):
        s = slurm.get(section, {}) or {}
        prefix = f"SLURM_{section.upper()}"
        _emit(f"{prefix}_TIME", s.get("time", "06:00:00"))
        _emit(f"{prefix}_CPUS", s.get("cpus_per_task", 8))
        _emit(f"{prefix}_MEM", s.get("mem", "32G"))
        _emit(f"{prefix}_PARTITION", s.get("partition", ""))
        _emit(f"{prefix}_CONSTRAINT", s.get("constraint", ""))
        _emit(f"{prefix}_GRES", s.get("gres", ""))

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            "Usage: python _load_paths.py <path/to/picasso_paths.yaml>",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
