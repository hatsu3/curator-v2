#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# End-to-end YFCC 1M label-modeling A/B (INT[] + GIN vs Boolean)
# - Resets databases to avoid interference
# - Loads datasets into INT and BOOLEAN DBs
# - Builds vector indexes (HNSW, IVFFlat)
# - Runs four single-label baselines (int_array/boolean x hnsw/ivf)
# - Collects storage metrics and writes summary

########################################
# Config (override with env vars or CLI flags)
########################################
PG_CONTAINER_NAME=${PG_CONTAINER_NAME:-pgvector-dev}
PG_IMAGE=${PG_IMAGE:-pgvector/pgvector:0.8.1-pg18-trixie}
PG_PORT=${PG_PORT:-5432}

DSN_INT=${DSN_INT:-postgresql://postgres:postgres@localhost:${PG_PORT}/curator_int}
DSN_BOOL=${DSN_BOOL:-postgresql://postgres:postgres@localhost:${PG_PORT}/curator_bool}

DATASET_KEY=${DATASET_KEY:-yfcc100m}
TEST_SIZE=${TEST_SIZE:-0.01}           # yfcc100m 1M
DIM=${DIM:-192}
K=${K:-10}

# HNSW params
HNSW_M=${HNSW_M:-32}
HNSW_EFC=${HNSW_EFC:-128}
HNSW_EFS=${HNSW_EFS:-128}

# IVF params
IVF_LISTS=${IVF_LISTS:-4096}
# Default probes to lists for full coverage (can override)
IVF_PROBES=${IVF_PROBES:-${IVF_LISTS}}

# Optional cap on number of queries evaluated per baseline
MAX_QUERIES=${MAX_QUERIES:-}

# Boolean label columns to materialize
TOP_LABEL_IDS=${TOP_LABEL_IDS:-1,2,3,4,5}

# Index build controls (defaults; can override via flags)
PARALLEL_MAINT_WORKERS=${PARALLEL_MAINT_WORKERS:-0}
MAINTENANCE_WORK_MEM=${MAINTENANCE_WORK_MEM:-64GB}

########################################
# CLI flags (overriding defaults)
########################################
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel-maint-workers)
      PARALLEL_MAINT_WORKERS="$2"; shift 2;;
    --maintenance-work-mem)
      MAINTENANCE_WORK_MEM="$2"; shift 2;;
    --max-queries)
      MAX_QUERIES="$2"; shift 2;;
    *)
      echo "[run_ab] Unknown argument: $1" >&2; exit 1;;
  esac
done

########################################
# Conda env activation
# Note: temporarily disable nounset for conda activation scripts which
# may reference unset variables (e.g., ADDR2LINE) under binutils hooks.
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
    echo "[run_ab] Starting existing container: ${PG_CONTAINER_NAME}"
    docker start "${PG_CONTAINER_NAME}" > /dev/null
  else
    echo "[run_ab] Launching container: ${PG_CONTAINER_NAME}"
    docker run --name "${PG_CONTAINER_NAME}" -e POSTGRES_PASSWORD=postgres -p ${PG_PORT}:5432 -d "${PG_IMAGE}" > /dev/null
  fi
fi

########################################
# Reset databases (drop & recreate)
########################################
echo "[run_ab] Resetting databases curator_int and curator_bool"
docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -q -X -v ON_ERROR_STOP=1 <<'SQL'
\set QUIET on
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname IN ('curator_int','curator_bool');
DROP DATABASE IF EXISTS curator_int;
DROP DATABASE IF EXISTS curator_bool;
CREATE DATABASE curator_int;
CREATE DATABASE curator_bool;
\c curator_int
CREATE EXTENSION IF NOT EXISTS vector;
\c curator_bool
CREATE EXTENSION IF NOT EXISTS vector;
SQL
echo "[run_ab] Databases curator_int/curator_bool reset; extension 'vector' ensured"

########################################
# Show DB settings, container resources
# Override default configurations
########################################
for DB in curator_int curator_bool; do
  mwm=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d "$DB" -qAtX -c "SHOW maintenance_work_mem;" 2>/dev/null | tr -d '\r' || true)
  pmw=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d "$DB" -qAtX -c "SHOW max_parallel_maintenance_workers;" 2>/dev/null | tr -d '\r' || true)
  mpw=$(docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -d "$DB" -qAtX -c "SHOW max_parallel_workers;" 2>/dev/null | tr -d '\r' || true)
  echo "[run_ab] DB defaults ($DB): maintenance_work_mem=$mwm, max_parallel_maintenance_workers=$pmw, max_parallel_workers=$mpw (build uses PGOPTIONS overrides)"
done
echo "[run_ab] Container resources (cgroups):"
docker exec -i "${PG_CONTAINER_NAME}" bash -lc 'set -e; if [ -f /sys/fs/cgroup/memory.max ]; then echo mem.max=$(cat /sys/fs/cgroup/memory.max); else echo mem.limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo unknown); fi; if [ -f /sys/fs/cgroup/cpu.max ]; then echo cpu.max=$(cat /sys/fs/cgroup/cpu.max); else echo cpu.cfs_quota_us=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo unknown); fi' || true

# Enforce single-threaded build (configurable) and optional memory allocation
echo "[run_ab] Overriding # parallel workers: max_parallel_maintenance_workers=${PARALLEL_MAINT_WORKERS}, max_parallel_workers=${PARALLEL_MAINT_WORKERS}"
PGOPTIONS_SET="-c max_parallel_maintenance_workers=${PARALLEL_MAINT_WORKERS} -c max_parallel_workers=${PARALLEL_MAINT_WORKERS} -c hnsw.max_scan_tuples=1000000000"
if [[ -n "${MAINTENANCE_WORK_MEM}" ]]; then
  echo "[run_ab] Overriding maintenance_work_mem: ${MAINTENANCE_WORK_MEM}"
  PGOPTIONS_SET+=" -c maintenance_work_mem=${MAINTENANCE_WORK_MEM}"
fi
export PGOPTIONS="${PGOPTIONS:-} ${PGOPTIONS_SET}"

########################################
# Bulk load datasets (HNSW supports incremental builds, 
# but that does not matter for this experiment)
########################################
echo "[run_ab] Bulk load into INT[] DB with GIN build timing"
python -m scripts.pgvector.load_dataset bulk \
  --dsn "${DSN_INT}" \
  --dataset yfcc100m --dataset_key "${DATASET_KEY}" --dim "${DIM}" --test_size "${TEST_SIZE}" \
  --copy_format csv --build_index gin

echo "[run_ab] Bulk load into BOOLEAN DB (no GIN)"
python -m scripts.pgvector.load_dataset bulk \
  --dsn "${DSN_BOOL}" \
  --dataset yfcc100m --dataset_key "${DATASET_KEY}" --dim "${DIM}" --test_size "${TEST_SIZE}" \
  --copy_format csv

echo "[run_ab] Add boolean label columns and backfill: ALL labels"
python -m scripts.pgvector.setup_db create_all_boolean_labels \
  --dsn "${DSN_BOOL}"

########################################
# Run baselines (HNSW first, then IVF) to avoid index isolation drops
########################################
echo "[run_ab] Build HNSW index (int_array)"
python -m scripts.pgvector.setup_db create_index \
  --dsn "${DSN_INT}" --index hnsw \
  --m "${HNSW_M}" --efc "${HNSW_EFC}" --dim "${DIM}"

echo "[run_ab] Run baselines: int_array + hnsw"
python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single \
  --dsn "${DSN_INT}" --strategy hnsw \
  --iter_mode relaxed_order --schema int_array \
  --dataset_key "${DATASET_KEY}" --test_size "${TEST_SIZE}" --k "${K}" \
  --m "${HNSW_M}" --ef_construction "${HNSW_EFC}" --ef_search "${HNSW_EFS}" \
  ${MAX_QUERIES:+ --max_queries "${MAX_QUERIES}"} \
  --output_path output/pgvector/label_ab/yfcc100m_1m/int_array/hnsw/results.csv

echo "[run_ab] Build IVF index (int_array)"
python -m scripts.pgvector.setup_db create_index \
  --dsn "${DSN_INT}" --index ivf \
  --lists "${IVF_LISTS}" --dim "${DIM}"

echo "[run_ab] Run baselines: int_array + ivf"
python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single \
  --dsn "${DSN_INT}" --strategy ivf \
  --iter_mode relaxed_order --schema int_array \
  --dataset_key "${DATASET_KEY}" --test_size "${TEST_SIZE}" --k "${K}" \
  --lists "${IVF_LISTS}" --probes "${IVF_PROBES}" \
  ${MAX_QUERIES:+ --max_queries "${MAX_QUERIES}"} \
  --output_path output/pgvector/label_ab/yfcc100m_1m/int_array/ivf/results.csv

echo "[run_ab] Build HNSW index (boolean)"
python -m scripts.pgvector.setup_db create_index \
  --dsn "${DSN_BOOL}" --index hnsw \
  --m "${HNSW_M}" --efc "${HNSW_EFC}" --dim "${DIM}"

echo "[run_ab] Run baselines: boolean + hnsw"
python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single \
  --dsn "${DSN_BOOL}" --strategy hnsw \
  --iter_mode relaxed_order --schema boolean \
  --dataset_key "${DATASET_KEY}" --test_size "${TEST_SIZE}" --k "${K}" \
  --m "${HNSW_M}" --ef_construction "${HNSW_EFC}" --ef_search "${HNSW_EFS}" \
  ${MAX_QUERIES:+ --max_queries "${MAX_QUERIES}"} \
  --output_path output/pgvector/label_ab/yfcc100m_1m/boolean/hnsw/results.csv

echo "[run_ab] Build IVF index (boolean)"
python -m scripts.pgvector.setup_db create_index \
  --dsn "${DSN_BOOL}" --index ivf \
  --lists "${IVF_LISTS}" --dim "${DIM}"

echo "[run_ab] Run baselines: boolean + ivf"
python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single \
  --dsn "${DSN_BOOL}" --strategy ivf \
  --iter_mode relaxed_order --schema boolean \
  --dataset_key "${DATASET_KEY}" --test_size "${TEST_SIZE}" --k "${K}" \
  --lists "${IVF_LISTS}" --probes "${IVF_PROBES}" \
  ${MAX_QUERIES:+ --max_queries "${MAX_QUERIES}"} \
  --output_path output/pgvector/label_ab/yfcc100m_1m/boolean/ivf/results.csv

########################################
# Storage metrics and summary
########################################
echo "[run_ab] Collect storage metrics"
python -m benchmark.pgvector_ab.label_model_ab run \
  --dsn_int "${DSN_INT}" --dsn_bool "${DSN_BOOL}" \
  --dataset_key "${DATASET_KEY}" --test_size "${TEST_SIZE}" --k "${K}" \
  --lists "${IVF_LISTS}" --probes "${IVF_PROBES}" \
  --m "${HNSW_M}" --ef_construction "${HNSW_EFC}" --ef_search "${HNSW_EFS}"

echo "[run_ab] Summarize A/B results"
python -m benchmark.pgvector_ab.summarize_label_ab summarize --dataset_variant yfcc100m_1m

echo "[run_ab] DONE. See outputs under output/pgvector/label_ab/yfcc100m_1m/"
