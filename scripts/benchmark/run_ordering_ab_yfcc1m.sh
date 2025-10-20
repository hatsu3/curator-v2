#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# End-to-end YFCC 1M HNSW ordering A/B (strict_order vs relaxed_order)
# - Resets database to avoid interference (drop & recreate)
# - Loads dataset into a dedicated DB
# - Builds required indexes (GIN, HNSW)
# - Runs single-label and complex predicate A/B via orchestrator
# - Produces summary with auto recommendation JSON

########################################
# Config (override with env vars or CLI flags)
########################################
PG_CONTAINER_NAME=${PG_CONTAINER_NAME:-pgvector-dev}
PG_IMAGE=${PG_IMAGE:-pgvector/pgvector:0.8.1-pg18-trixie}
PG_PORT=${PG_PORT:-5432}

DB_NAME=${DB_NAME:-curator_ordering}
DSN=${DSN:-postgresql://postgres:postgres@localhost:${PG_PORT}/${DB_NAME}}

DATASET_KEY=${DATASET_KEY:-yfcc100m}
TEST_SIZE=${TEST_SIZE:-0.01}           # yfcc100m 1M
DIM=${DIM:-192}
K=${K:-10}

# HNSW params
HNSW_M=${HNSW_M:-32}
HNSW_EFC=${HNSW_EFC:-128}
HNSW_EFS=${HNSW_EFS:-128}

# Dataset cache (optional, used by single-label baseline)
DATASET_CACHE_PATH=${DATASET_CACHE_PATH:-data/cache}

# Summary thresholds
RECALL_TOL=${RECALL_TOL:-0.001}
P95_SPEEDUP_MIN=${P95_SPEEDUP_MIN:-0.05}

# Index build controls (defaults; can override via flags)
PARALLEL_MAINT_WORKERS=${PARALLEL_MAINT_WORKERS:-0}
MAINTENANCE_WORK_MEM=${MAINTENANCE_WORK_MEM:-64GB}

########################################
# CLI flags
########################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel-maint-workers)
      PARALLEL_MAINT_WORKERS="$2"; shift 2;;
    --maintenance-work-mem)
      MAINTENANCE_WORK_MEM="$2"; shift 2;;
    *) echo "[ordering_ab] Unknown argument: $1" >&2; exit 1;;
  esac
done

########################################
# Conda env activation
########################################
set +u
eval "$(conda shell.bash hook)"
conda activate ann_bench2
set -u

########################################
# Start/ensure Postgres container
########################################
if ! docker ps --format '{{.Names}}' | grep -q "^${PG_CONTAINER_NAME}$"; then
  if docker ps -a --format '{{.Names}}' | grep -q "^${PG_CONTAINER_NAME}$"; then
    echo "[ordering_ab] Starting existing container: ${PG_CONTAINER_NAME}"
    docker start "${PG_CONTAINER_NAME}" > /dev/null
  else
    echo "[ordering_ab] Launching container: ${PG_CONTAINER_NAME}"
    docker run --name "${PG_CONTAINER_NAME}" -e POSTGRES_PASSWORD=postgres -p ${PG_PORT}:5432 -d "${PG_IMAGE}" > /dev/null
  fi
fi

########################################
# Reset database (drop & recreate)
# NOTE: This is destructive. Ensure no experiments are running.
########################################
echo "[ordering_ab] Resetting database ${DB_NAME}"
docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -q -X -v ON_ERROR_STOP=1 <<'SQL'
\set QUIET on
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = :'DB_NAME';
DROP DATABASE IF EXISTS :"DB_NAME";
CREATE DATABASE :"DB_NAME";
\c :"DB_NAME"
CREATE EXTENSION IF NOT EXISTS vector;
SQL
echo "[ordering_ab] Database ${DB_NAME} reset; extension 'vector' ensured"

echo "[ordering_ab] DB defaults (${DB_NAME}):"
mwm=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d "${DB_NAME}" -qAtX -c "SHOW maintenance_work_mem;" 2>/dev/null | tr -d '\r' || true)
pmw=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d "${DB_NAME}" -qAtX -c "SHOW max_parallel_maintenance_workers;" 2>/dev/null | tr -d '\r' || true)
mpw=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d "${DB_NAME}" -qAtX -c "SHOW max_parallel_workers;" 2>/dev/null | tr -d '\r' || true)
echo "[ordering_ab] DB defaults: maintenance_work_mem=$mwm, max_parallel_maintenance_workers=$pmw, max_parallel_workers=$mpw (build uses PGOPTIONS overrides)"
echo "[ordering_ab] Container resources (cgroups):"
docker exec -i "${PG_CONTAINER_NAME}" bash -lc 'set -e; if [ -f /sys/fs/cgroup/memory.max ]; then echo mem.max=$(cat /sys/fs/cgroup/memory.max); else echo mem.limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown); fi; if [ -f /sys/fs/cgroup/cpu.max ]; then echo cpu.max=$(cat /sys/fs/cgroup/cpu.max); else echo cpu.cfs_quota_us=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo unknown); fi' || true

# Force single-threaded build (configurable) and optional memory allocation
echo "[ordering_ab] Overriding # parallel workers: max_parallel_maintenance_workers=${PARALLEL_MAINT_WORKERS}, max_parallel_workers=${PARALLEL_MAINT_WORKERS}"
PGOPTIONS_SET="-c max_parallel_maintenance_workers=${PARALLEL_MAINT_WORKERS} -c max_parallel_workers=${PARALLEL_MAINT_WORKERS} -c hnsw.max_scan_tuples=1000000000"
if [[ -n "${MAINTENANCE_WORK_MEM}" ]]; then
  echo "[ordering_ab] Overriding maintenance_work_mem: ${MAINTENANCE_WORK_MEM}"
  PGOPTIONS_SET+=" -c maintenance_work_mem=${MAINTENANCE_WORK_MEM}"
fi
export PGOPTIONS="${PGOPTIONS:-} ${PGOPTIONS_SET}"

########################################
# Bulk load dataset and build GIN(tags)
########################################
echo "[ordering_ab] Bulk load and build GIN(tags)"
python -m scripts.pgvector.load_dataset bulk \
  --dsn "${DSN}" \
  --dataset yfcc100m --dataset_key "${DATASET_KEY}" --dim "${DIM}" --test_size "${TEST_SIZE}" \
  --copy_format csv --build_index gin

########################################
# Build HNSW index (required for A/B)
########################################
echo "[ordering_ab] Build HNSW index"
python -m scripts.pgvector.setup_db create_index \
  --dsn "${DSN}" --index hnsw --m "${HNSW_M}" --efc "${HNSW_EFC}" --dim "${DIM}"

########################################
# Run A/B orchestrator
########################################
DV=yfcc100m_1m
echo "[ordering_ab] Run single-label A/B (strict vs relaxed)"
python -m benchmark.pgvector_ab.hnsw_ordering_ab ab_single \
  --dsn "${DSN}" \
  --dataset_variant "${DV}" --dataset_key "${DATASET_KEY}" --test_size "${TEST_SIZE}" --k "${K}" \
  --m "${HNSW_M}" --ef_construction "${HNSW_EFC}" --ef_search "${HNSW_EFS}" \
  --dataset_cache_path "${DATASET_CACHE_PATH}"

echo "[ordering_ab] Run complex predicates A/B (AND/OR)"
python -m benchmark.pgvector_ab.hnsw_ordering_ab ab_complex \
  --dsn "${DSN}" \
  --dataset_variant "${DV}" --dataset_key "${DATASET_KEY}" --test_size "${TEST_SIZE}" --k "${K}" \
  --m "${HNSW_M}" --ef_construction "${HNSW_EFC}" --ef_search "${HNSW_EFS}" \
  --n_filters_per_template 50 --n_queries_per_filter 100

########################################
# Summarize and recommend
########################################
echo "[ordering_ab] Summarize results and emit recommendation"
python -m benchmark.pgvector_ab.summarize_ordering_ab run \
  --dataset_variant "${DV}" --recall_tolerance "${RECALL_TOL}" --p95_speedup_min "${P95_SPEEDUP_MIN}" --write_json

echo "[ordering_ab] DONE. See outputs under output/pgvector/hnsw_ordering_ab/${DV}/"
