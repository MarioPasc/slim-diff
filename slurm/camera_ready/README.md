# ICIP 2026 Camera-Ready — 3 folds × 2 architectures grid

This directory holds the SLURM scaffolding for the 6-cell camera-ready
experiment grid committed to in the author's response (file
`authors_answer.txt`). It addresses reviewer concerns **R1.1** (missing
baseline), **R1.3** (stability across splits), **R2.2** (shared-bottleneck
ablation), and **R2.3** (within-framework comparison).

## Experiment Matrix

| Cell | Architecture | Fold | Job name | Train/val pool | Test |
|---|---|---|---|---|---|
| `shared_fold_0`     | shared    | 0 | `slimdiff_cr_shared_fold_0`     | fold-0 train+val | fixed |
| `shared_fold_1`     | shared    | 1 | `slimdiff_cr_shared_fold_1`     | fold-1 train+val | fixed |
| `shared_fold_2`     | shared    | 2 | `slimdiff_cr_shared_fold_2`     | fold-2 train+val | fixed |
| `decoupled_fold_0`  | decoupled | 0 | `slimdiff_cr_decoupled_fold_0`  | fold-0 train+val | fixed |
| `decoupled_fold_1`  | decoupled | 1 | `slimdiff_cr_decoupled_fold_1`  | fold-1 train+val | fixed |
| `decoupled_fold_2`  | decoupled | 2 | `slimdiff_cr_decoupled_fold_2`  | fold-2 train+val | fixed |

Test set is **fixed** across folds (TASK-02 design); only the train/val
partition rotates. See `src/diffusion/data/kfold.py`.

## Launching on Picasso

```bash
# From the repo root on Picasso
bash slurm/camera_ready/launch_camera_ready.sh
```

The launcher submits all 6 jobs independently — they run in parallel
subject to cluster availability. Each job does train → generate sequentially
within its own 3-day window (see `#SBATCH --time=3-00:00:00`).

## Inputs

Each cell reuses the shared slice cache at
`${DATA_SRC}/slice_cache/` and per-fold CSVs at
`${DATA_SRC}/slice_cache/folds/fold_${FOLD_ID}/{train,val,test}.csv`.

The first camera-ready job to run creates both (idempotently). Subsequent
jobs skip. Fold CSV generation uses:

```bash
slimdiff-kfold --cache-dir "${CACHE_DIR}" --n-folds 3 --seed 42
```

The `slimdiff-kfold` CLI writes `folds.tmp/` then atomically swaps via
`os.replace` — concurrent jobs racing on fold creation cannot corrupt the
tree.

## Outputs

Each cell writes to a unique directory under `${RESULTS_DST}`:

```
results/camera_ready/
├── slimdiff_cr_shared_fold_0/
│   ├── config.yaml              # sed-patched copy of the committed YAML
│   ├── checkpoints/*.ckpt
│   ├── logs/
│   └── replicas/                # 20 NPZ files, ~9 000 samples each
│       ├── replica_000.npz
│       └── ...
├── slimdiff_cr_shared_fold_1/
├── slimdiff_cr_shared_fold_2/
├── slimdiff_cr_decoupled_fold_0/
├── slimdiff_cr_decoupled_fold_1/
└── slimdiff_cr_decoupled_fold_2/
```

Total generated samples per cell: **20 replicas × 150 samples/mode × 30
z-bins × 2 domains = 180 000** (90 000 per domain), matching the figure
quoted in the paper.

## Reproducibility guarantees

- **Training seed**: `experiment.seed = 33` (inherited from the best ICIP
  ablation config; held constant across the 6 cells).
- **Fold seed**: `--seed 42` passed to `slimdiff-kfold`. Deterministic fold
  assignment (see TASK-02's `test_determinism`).
- **Generation seed**: `SEED_BASE = 42`, **identical across all 6 cells**.
  This is intentional — `generate_replicas.py` derives x_T via
  SHA256(seed_base, replica_id, zbin, lesion_present, domain_int,
  sample_index). Architecture is **not** part of the hash, so
  `shared_fold_k` and `decoupled_fold_k` produce byte-identical x_T for
  matched (replica, zbin, domain, sample) tuples. TASK-06's paired
  qualitative comparison relies on this.

## Configuration source of truth

Per-cell configs live at `slurm/camera_ready/{cell}/config.yaml`. They are
derived from the canonical templates at `configs/camera_ready/base_{arch}.yaml`
by substituting the literal string `fold_K` with the concrete fold id.

**Divergences from the ICIP ablation reference** (`slurm/icip2026/lp_ablation/x0/lp_1.5/x0_lp_1.5.yaml`):

| Field | Reference | Camera-ready |
|---|---|---|
| `training.devices`  | 1 | 2 |
| `training.strategy` | `"auto"` | `"ddp"` |
| `training.max_epochs` | 500 | 1000 (early stopping at patience=25 still caps it) |
| `model.bottleneck_mode` | absent | `"shared"` or `"decoupled"` |
| `model.decoupled_bottleneck` | absent | present (used only for decoupled) |

Everything else (scheduler, sampler, loss, optimizer, EMA, conditioning) is
held fixed. The only field that distinguishes shared-fold-k from
decoupled-fold-k is `model.bottleneck_mode` — this is enforced by
`src/diffusion/tests/test_camera_ready_configs.py` (acceptance Test 3).

## Expected runtime

- Training (per cell): ~24–48 h on 2× A100 (3-day SLURM cap).
- Generation (per cell): ~2–4 h for 20 replicas × 9 000 samples/replica.
- Total grid: ~36 GPU-days training + ~60–120 GPU-hours generation.

## Troubleshooting

- **`slimdiff-*` CLI missing**: run `pip install -e .` in the Picasso env
  (`conda activate jsddpm`) to pick up the entrypoints from `pyproject.toml`.
- **Fold CSVs not regenerating after config change**: delete
  `${CACHE_DIR}/folds/fold_${FOLD_ID}/train.csv` and resubmit — the guard
  only triggers generation when this file is missing.
- **Only 1 GPU visible inside the job**: check `#SBATCH --gres=gpu:2` and
  the DDP verification output in the `*.out` file. Falls back to
  single-GPU training but diverges from the committed setup.
