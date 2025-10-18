#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# End-to-end YFCC 1M pgvector insert A/B (durable vs non-durable)
# - Resets database to avoid interference
# - Runs six insert benchmarks: {hnsw, ivf, prefilter} × {durable, non_durable}
# - Writes per-run A/B artifacts under output/pgvector/insert_ab/
# - Aggregates summary.{json,csv}

########################################
# Config (override with env vars)
########################################
PG_CONTAINER_NAME=${PG_CONTAINER_NAME:-pgvector-dev}
PG_IMAGE=${PG_IMAGE:-pgvector/pgvector:0.8.1-pg18-trixie}
PG_PORT=${PG_PORT:-5432}

DSN=${DSN:-postgresql://postgres:postgres@localhost:${PG_PORT}/curator_bench}

DATASET=${DATASET:-yfcc100m}
DATASET_KEY=${DATASET_KEY:-yfcc100m}
TEST_SIZE=${TEST_SIZE:-0.01}          # yfcc100m 1M
DIM=${DIM:-192}
LIMIT=${LIMIT:-50000}

# HNSW params
HNSW_M=${HNSW_M:-32}
HNSW_EFC=${HNSW_EFC:-64}

# IVF params
IVF_LISTS=${IVF_LISTS:-200}

########################################
# Conda env activation
########################################
set +u
eval "$(conda shell.bash hook)"
conda activate ann_bench2
set -u

########################################
# Ensure Postgres container is running
########################################
if ! docker ps --format '{{.Names}}' | grep -q "^${PG_CONTAINER_NAME}$"; then
  if docker ps -a --format '{{.Names}}' | grep -q "^${PG_CONTAINER_NAME}$"; then
    echo "[run_insert_ab] Starting existing container: ${PG_CONTAINER_NAME}"
    docker start "${PG_CONTAINER_NAME}" > /dev/null
  else
    echo "[run_insert_ab] Launching container: ${PG_CONTAINER_NAME}"
    docker run --name "${PG_CONTAINER_NAME}" -e POSTGRES_PASSWORD=postgres -p ${PG_PORT}:5432 -d "${PG_IMAGE}" > /dev/null
  fi
fi

########################################
# Reset database (drop & recreate)
########################################
echo "[run_insert_ab] Resetting database curator_bench"
docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -q -X -v ON_ERROR_STOP=1 <<'SQL'
\set QUIET on
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'curator_bench';
DROP DATABASE IF EXISTS curator_bench;
CREATE DATABASE curator_bench;
\c curator_bench
CREATE EXTENSION IF NOT EXISTS vector;
SQL
echo "[run_insert_ab] Database curator_bench reset; extension 'vector' ensured"

# Show DB defaults (note: build sessions use PGOPTIONS overrides)
mwm=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d curator_bench -qAtX -c "SHOW maintenance_work_mem;" 2>/dev/null | tr -d '\r' || true)
pmw=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d curator_bench -qAtX -c "SHOW max_parallel_maintenance_workers;" 2>/dev/null | tr -d '\r' || true)
mpw=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d curator_bench -qAtX -c "SHOW max_parallel_workers;" 2>/dev/null | tr -d '\r' || true)
echo "[run_insert_ab] DB defaults (curator_bench): maintenance_work_mem=${mwm}, max_parallel_maintenance_workers=${pmw}, max_parallel_workers=${mpw} (build uses PGOPTIONS overrides)"
echo "[run_insert_ab] Container resources (cgroups):"
docker exec -i "${PG_CONTAINER_NAME}" bash -lc 'set -e; if [ -f /sys/fs/cgroup/memory.max ]; then echo mem.max=$(cat /sys/fs/cgroup/memory.max); else echo mem.limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown); fi; if [ -f /sys/fs/cgroup/cpu.max ]; then echo cpu.max=$(cat /sys/fs/cgroup/cpu.max); else echo cpu.cfs_quota_us=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo unknown); fi' || true

# Helper: drop 'items' so each run can recreate with LOGGED/UNLOGGED as needed
drop_items() {
  docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -q -X -v ON_ERROR_STOP=1 -d curator_bench <<'SQL'
\set QUIET on
DROP TABLE IF EXISTS items CASCADE;
SQL
}

# Force single-threaded index builds (configurable) and optional memory allocation
PARALLEL_MAINT_WORKERS=${PARALLEL_MAINT_WORKERS:-0}
MAINTENANCE_WORK_MEM=${MAINTENANCE_WORK_MEM:-64GB}
PGOPTIONS_SET="-c max_parallel_maintenance_workers=${PARALLEL_MAINT_WORKERS} -c max_parallel_workers=${PARALLEL_MAINT_WORKERS}"
if [[ -n "${MAINTENANCE_WORK_MEM}" ]]; then
  PGOPTIONS_SET+=" -c maintenance_work_mem=${MAINTENANCE_WORK_MEM}"
fi
export PGOPTIONS="${PGOPTIONS:-} ${PGOPTIONS_SET}"

########################################
# Durable runs (LOGGED + synchronous_commit=on)
########################################
echo "[run_insert_ab] HNSW durable"
drop_items
python -m scripts.pgvector.load_dataset insert_bench \
  --dsn "${DSN}" --dataset "${DATASET}" --dataset_key "${DATASET_KEY}" \
  --dim "${DIM}" --test_size "${TEST_SIZE}" --strategy hnsw \
  --m "${HNSW_M}" --efc "${HNSW_EFC}" --limit "${LIMIT}" --dry_run false

echo "[run_insert_ab] IVF durable (seed then build)"
drop_items
python -m scripts.pgvector.load_dataset insert_bench \
  --dsn "${DSN}" --dataset "${DATASET}" --dataset_key "${DATASET_KEY}" \
  --dim "${DIM}" --test_size "${TEST_SIZE}" --strategy ivf \
  --lists "${IVF_LISTS}" --limit "${LIMIT}" --dry_run false

echo "[run_insert_ab] Prefilter durable (GIN-only)"
drop_items
python -m scripts.pgvector.load_dataset insert_bench \
  --dsn "${DSN}" --dataset "${DATASET}" --dataset_key "${DATASET_KEY}" \
  --dim "${DIM}" --test_size "${TEST_SIZE}" --strategy prefilter \
  --limit "${LIMIT}" --dry_run false

########################################
# Non-durable runs (UNLOGGED)
########################################
echo "[run_insert_ab] HNSW non-durable (UNLOGGED)"
drop_items
python -m scripts.pgvector.load_dataset insert_bench \
  --dsn "${DSN}" --dataset "${DATASET}" --dataset_key "${DATASET_KEY}" \
  --dim "${DIM}" --test_size "${TEST_SIZE}" --strategy hnsw \
  --m "${HNSW_M}" --efc "${HNSW_EFC}" --unlogged true --limit "${LIMIT}" --dry_run false

echo "[run_insert_ab] IVF non-durable (UNLOGGED; seed then build)"
drop_items
python -m scripts.pgvector.load_dataset insert_bench \
  --dsn "${DSN}" --dataset "${DATASET}" --dataset_key "${DATASET_KEY}" \
  --dim "${DIM}" --test_size "${TEST_SIZE}" --strategy ivf \
  --lists "${IVF_LISTS}" --unlogged true --limit "${LIMIT}" --dry_run false

echo "[run_insert_ab] Prefilter non-durable (UNLOGGED)"
drop_items
python -m scripts.pgvector.load_dataset insert_bench \
  --dsn "${DSN}" --dataset "${DATASET}" --dataset_key "${DATASET_KEY}" \
  --dim "${DIM}" --test_size "${TEST_SIZE}" --strategy prefilter \
  --unlogged true --limit "${LIMIT}" --dry_run false

########################################
# Aggregate summary from A/B artifacts
########################################
echo "[run_insert_ab] Summarizing A/B results"
python - "$DATASET_KEY" <<'PY'
import csv, json, os, sys
from pathlib import Path

dataset_key = sys.argv[1]
base = Path("output/pgvector/insert_ab")/dataset_key
runs = [
    ("hnsw","durable"),("hnsw","non_durable"),
    ("ivf","durable"),("ivf","non_durable"),
    ("gin","durable"),("gin","non_durable"),
]
rows = []
for idx, dur in runs:
    p = base/idx/dur/"run.json"
    if not p.exists():
        print(f"[summary] missing {p}")
        continue
    with open(p, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    obj['index'] = idx
    obj['durability'] = dur
    rows.append(obj)
base.mkdir(parents=True, exist_ok=True)
out_csv = base/"summary.csv"
out_json = base/"summary.json"
if rows:
    fns = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fns)
        w.writeheader()
        for r in rows:
            w.writerow(r)
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump({"dataset_key": dataset_key, "runs": rows}, f, indent=2, sort_keys=True)
print("[summary] wrote", out_csv)
print("[summary] wrote", out_json)
PY

echo "[run_insert_ab] DONE. See outputs under output/pgvector/insert_ab/${DATASET_KEY}/"
