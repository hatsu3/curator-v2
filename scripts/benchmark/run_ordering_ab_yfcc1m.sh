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
# Config (override with env vars)
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
HNSW_EFC=${HNSW_EFC:-64}
HNSW_EFS=${HNSW_EFS:-64}

# Dataset cache (optional, used by single-label baseline)
DATASET_CACHE_PATH=${DATASET_CACHE_PATH:-data/cache}

# Summary thresholds
RECALL_TOL=${RECALL_TOL:-0.001}
P95_SPEEDUP_MIN=${P95_SPEEDUP_MIN:-0.05}

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
docker exec -i "${PG_CONTAINER_NAME}" psql -U postgres -v ON_ERROR_STOP=1 <<SQL
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${DB_NAME}';
DROP DATABASE IF EXISTS ${DB_NAME};
CREATE DATABASE ${DB_NAME};
\c ${DB_NAME}
CREATE EXTENSION IF NOT EXISTS vector;
SQL

########################################
# Bulk load dataset and build GIN(tags)
########################################
echo "[ordering_ab] Bulk load and build GIN(tags)"
python -m scripts.pgvector.load_dataset bulk \
  --dsn "${DSN}" \
  --dataset yfcc100m --dataset_key "${DATASET_KEY}" --dim "${DIM}" --test_size "${TEST_SIZE}" \
  --copy_format csv --build_index gin --dry_run false

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
  --dataset_cache_path "${DATASET_CACHE_PATH}" --dry_run false

echo "[ordering_ab] Run complex predicates A/B (AND/OR)"
python -m benchmark.pgvector_ab.hnsw_ordering_ab ab_complex \
  --dsn "${DSN}" \
  --dataset_variant "${DV}" --dataset_key "${DATASET_KEY}" --test_size "${TEST_SIZE}" --k "${K}" \
  --m "${HNSW_M}" --ef_construction "${HNSW_EFC}" --ef_search "${HNSW_EFS}" \
  --n_filters_per_template 50 --n_queries_per_filter 100 --dry_run false

########################################
# Summarize and recommend
########################################
echo "[ordering_ab] Summarize results and emit recommendation"
python -m benchmark.pgvector_ab.summarize_ordering_ab run \
  --dataset_variant "${DV}" --recall_tolerance "${RECALL_TOL}" --p95_speedup_min "${P95_SPEEDUP_MIN}" --write_json true

echo "[ordering_ab] DONE. See outputs under output/pgvector/hnsw_ordering_ab/${DV}/"

