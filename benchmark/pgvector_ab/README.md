pgvector Insert A/B Sweeper
===========================

Overview
--------
This benchmark orchestrates pgvector single-thread insert runs across:
- Strategies: `hnsw`, `ivf`, `prefilter` (GIN-only)
- Durability: `durable` (LOGGED + synchronous_commit=on), `non_durable` (UNLOGGED)

Per-run results are written under:
- `output/pgvector/insert_ab/<dataset_key>/{hnsw|ivf|gin}/{durable|non_durable}/run.{json,csv}`

A summary across all runs is written to:
- `output/pgvector/insert_ab/<dataset_key>/summary.{json,csv}`

Prerequisites
-------------
- Conda env: `ann_bench2`
- Local Postgres with `pgvector` and an initialized DB; see `scripts/pgvector/README.md` for setup and schema commands.
- Dataset prepared (e.g., YFCC 1M subsample) using the dataset utilities.

Dry‑Run Preview
---------------
Preview exact commands and planned outputs without touching the DB:
- `eval "$(conda shell.bash hook)" && conda activate ann_bench2 && \
  python -m benchmark.pgvector_ab.insert_ab run \
    --dry_run true --dataset yfcc100m --dataset_key yfcc100m \
    --dim 192 --test_size 0.01 --m 32 --efc 64 --lists 200 --limit 10000`

Execute Runs
------------
Run all 6 combinations and aggregate summary (DB must be ready):
- `eval "$(conda shell.bash hook)" && conda activate ann_bench2 && \
  python -m benchmark.pgvector_ab.insert_ab run \
    --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
    --dataset yfcc100m --dataset_key yfcc100m \
    --dim 192 --test_size 0.01 --m 32 --efc 64 --lists 200 --limit 50000 --dry_run false`

Parameters
----------
- `m`, `efc`: HNSW construction parameters.
- `lists`: IVF lists; sweeper can `--build_ivf true` to build IVF first.
- `limit`: Cap rows for smoke runs.
- `dry_run`: Print only (no DB); set false to execute.

Notes
-----
- Insert path measures per‑row latency in autocommit mode (one transaction per row).
- Non‑durable mode uses UNLOGGED tables by default to reflect in‑process baselines.
- HNSW index creation is handled inside `insert_bench`; IVF can be built first by the sweeper.
- Prefilter ensures `GIN(tags)` presence.

Troubleshooting
---------------
- IVF build errors: ensure `CREATE EXTENSION vector;` and that the `items` table is present; rebuild IVF before inserts if needed.
- DSN/auth errors: verify `--dsn` and that the target database is accessible.
- Missing outputs: re‑run with `--dry_run true` to verify planned paths and commands.

