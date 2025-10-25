#!/usr/bin/env bash
set -euo pipefail

# Pgvector complex-predicate runner (YFCC-1M subset)
# Tasks: ivf_build | ivf_search | hnsw_build | hnsw_search
# Reads parameter grids from benchmark/complex_predicate/optimal_baseline_params.json

usage() {
  echo "Usage: $0 <task> [output_dir] [params_json]" >&2
  echo "  task        ivf_build | ivf_search | hnsw_build | hnsw_search" >&2
  echo "  output_dir  default: output/complex_predicate_pgvector" >&2
  echo "  params_json default: benchmark/complex_predicate/optimal_baseline_params.json" >&2
}

# If no arguments are provided, show usage and exit
if [[ $# -lt 1 ]]; then usage; exit 1; fi

TASK="$1"; shift || true
OUT_ROOT="${1:-output/complex_predicate_optimal}"; shift || true
PARAMS_JSON="${1:-benchmark/complex_predicate/optimal_baseline_params.json}"; shift || true

# Fixed dataset: yfcc-1m dataset
DATASET_KEY="yfcc100m"
TEST_SIZE="0.01"
DIM=192

# Activate conda environment
set +u
eval "$(conda shell.bash hook)"
conda activate ann_bench2
set -u

DSN_DEFAULT="postgresql://postgres:postgres@localhost:5434/curator_bench_yfcc1m"
PG_DSN="${PG_DSN:-$DSN_DEFAULT}"

# Build-time knobs
PARALLEL_MAINT_WORKERS="${PARALLEL_MAINT_WORKERS:-0}"
MAINTENANCE_WORK_MEM="${MAINTENANCE_WORK_MEM:-64GB}"

mkdir -p "$OUT_ROOT/pgvector_hnsw" "$OUT_ROOT/pgvector_ivf"

# Write experiment config for plotting (selectivity computation expects this)
CONFIG_FILE="$OUT_ROOT/experiment_config.json"
cat > "$CONFIG_FILE" << EOF
{
  "common_parameters": {
    "dataset_key": "${DATASET_KEY}",
    "test_size": ${TEST_SIZE},
    "templates": ["AND {0} {1}", "OR {0} {1}"],
    "n_filters_per_template": 10,
    "n_queries_per_filter": 100,
    "gt_cache_dir": "data/ground_truth/complex_predicate"
  }
}
EOF
echo "[pgvector][cfg] wrote $CONFIG_FILE"

# Export PGOPTIONS to influence build sessions
PGOPTIONS_SET="-c max_parallel_maintenance_workers=${PARALLEL_MAINT_WORKERS} -c max_parallel_workers=${PARALLEL_MAINT_WORKERS} -c work_mem=1GB"
if [[ -n "${MAINTENANCE_WORK_MEM}" ]]; then
  PGOPTIONS_SET+=" -c maintenance_work_mem=${MAINTENANCE_WORK_MEM}"
fi
export PGOPTIONS="${PGOPTIONS:-} ${PGOPTIONS_SET}"
echo "[pgvector][cfg] PGOPTIONS='${PGOPTIONS}'"

# Max concurrent search jobs (only used for *search tasks)
MAX_PROCS=${MAX_PROCS:-5}

# On Ctrl-C/TERM, stop all background jobs cleanly
trap 'echo "[pgvector][parallel] cancel"; jobs -pr | xargs -r kill; wait; exit 130' INT TERM

# Connectivity and cleanliness checks (mirrors overall_results runner)
check_db_reachable() {
  python - "$PG_DSN" <<'PY'
import sys
import psycopg2
dsn=sys.argv[1]
try:
    with psycopg2.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
    print("[pgvector][check] DB reachable")
except Exception as e:
    print(f"[pgvector][check][ERROR] cannot connect: {e}", file=sys.stderr)
    sys.exit(2)
PY
}

check_clean_for_build() {
  python - "$PG_DSN" <<'PY'
import sys,psycopg2
dsn=sys.argv[1]
with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.items') IS NOT NULL;")
        exists = bool(cur.fetchone()[0])
        if not exists:
            print("[pgvector][check][ERROR] table 'items' missing. Run setup_db create_schema first.", file=sys.stderr)
            sys.exit(3)
        cur.execute("SELECT COUNT(*) FROM items;")
        n=int(cur.fetchone()[0])
        if n!=0:
            print(f"[pgvector][check][ERROR] table 'items' not clean (rows={n}). Please reset before build.", file=sys.stderr)
            sys.exit(4)
print("[pgvector][check] items is present and empty")
PY
}

# Parameter extractor
extract_param() {
  # $1: baseline JSON name, $2: dataset (yfcc100m), $3: json path (dot keys), prints scalar or JSON
  local base="$1" ds="$2" path="$3"
  python - <<PY
import json,sys
p=json.load(open("$PARAMS_JSON"))
v=p.get("$base",{}).get("$ds",{})
for k in "$path".split('.'): v=v.get(k, {})
if isinstance(v, (int,float,str)): print(v)
else: print(json.dumps(v))
PY
}

# Read construction/search parameter spaces via extract_param
HNSW_JSON_NAME="pgvector HNSW"
IVF_JSON_NAME="pgvector IVF"

M=$(extract_param "$HNSW_JSON_NAME" "$DATASET_KEY" construction_params.m)
EFC=$(extract_param "$HNSW_JSON_NAME" "$DATASET_KEY" construction_params.construction_ef)
HNSW_EF_LIST=$(extract_param "$HNSW_JSON_NAME" "$DATASET_KEY" search_param_combinations.ef_search)
SCAN_ARR=$(extract_param "$HNSW_JSON_NAME" "$DATASET_KEY" search_param_combinations.max_scan_tuples)
# pick first ef for sweep naming; per our design ef list has one value
EF=$(python -c "import json,sys; a=json.loads(sys.argv[1]); print(a[0] if isinstance(a,list) and a else a)" "$HNSW_EF_LIST")

LISTS=$(extract_param "$IVF_JSON_NAME" "$DATASET_KEY" construction_params.lists)
NPROBE_LIST=$(extract_param "$IVF_JSON_NAME" "$DATASET_KEY" search_param_combinations.nprobe)
NPROBE=$(python -c "import json,sys; a=json.loads(sys.argv[1]); print(a[0] if isinstance(a,list) and a else a)" "$NPROBE_LIST")
OVERSCAN_LIST=$(extract_param "$IVF_JSON_NAME" "$DATASET_KEY" search_param_combinations.overscan)
MAXP_LIST=$(extract_param "$IVF_JSON_NAME" "$DATASET_KEY" search_param_combinations.max_probes)
MAXP=$(python -c "import json,sys; a=json.loads(sys.argv[1]); print(a[0] if isinstance(a,list) and a else a)" "$MAXP_LIST")

echo "[pgvector][cfg] DSN=$PG_DSN DATASET_KEY=$DATASET_KEY TEST_SIZE=$TEST_SIZE DIM=$DIM"
echo "[pgvector][cfg] HNSW m=$M efc=$EFC ef=$EF max_scan_tuples=$SCAN_ARR"
echo "[pgvector][cfg] IVF nlist=$LISTS nprobe=$NPROBE max_probes=$MAXP overscan_list=$OVERSCAN_LIST"

case "$TASK" in
  ivf_build)
    check_db_reachable
    check_clean_for_build
    python -m scripts.pgvector.load_dataset bulk \
      --dsn "$PG_DSN" \
      --dataset_key "$DATASET_KEY" \
      --dim $DIM \
      --test_size $TEST_SIZE \
      --copy_format binary \
      --create_gin \
      --unlogged \
      --sync_commit_off \
      --build_index ivf \
      --lists "$LISTS" \
      --truncate
    ;;

  hnsw_build)
    check_db_reachable
    check_clean_for_build
    python -m scripts.pgvector.load_dataset bulk \
      --dsn "$PG_DSN" \
      --dataset_key "$DATASET_KEY" \
      --dim $DIM \
      --test_size $TEST_SIZE \
      --copy_format binary \
      --create_gin \
      --unlogged \
      --sync_commit_off \
      --build_index hnsw \
      --m "$M" \
      --efc "$EFC" \
      --truncate
    ;;

  ivf_search)
    OV_LIST=$(python -c "import json; print(' '.join(str(x) for x in json.loads('$OVERSCAN_LIST')))" )
    echo "[pgvector][ivf_search] running in parallel: MAX_PROCS=${MAX_PROCS}"
    pids=()
    logs=()
    for ov in $OV_LIST; do
      OUT_JSON="$OUT_ROOT/pgvector_ivf/overscan${ov}.json"
      log="$OUT_ROOT/pgvector_ivf/overscan${ov}.log"
      while [ "$(jobs -pr | wc -l)" -ge "$MAX_PROCS" ]; do wait -n || true; done
      echo "[pgvector][ivf_search] launch nprobe=$NPROBE overscan=$ov (max_probes=$MAXP) -> $OUT_JSON (log=$log)"
      python -u -m benchmark.complex_predicate.baselines.pgvector \
        exp_pgvector_complex \
        --dsn "$PG_DSN" \
        --strategy ivf \
        --iter_mode relaxed_order \
        --dataset_key "$DATASET_KEY" \
        --test_size "$TEST_SIZE" \
        --k 10 \
        --lists "$LISTS" \
        --probes "$NPROBE" \
        --ivf_max_probes "$MAXP" \
        --ivf_overscan "$ov" \
        --output_path "$OUT_JSON" >"$log" 2>&1 &
      pids+=("$!")
      logs+=("$log")
    done
    # wait for all
    fail=0
    for i in "${!pids[@]}"; do
      pid="${pids[$i]}"; log="${logs[$i]}"
      if ! wait "$pid"; then
        echo "[pgvector][ivf_search][ERROR] job pid=$pid failed; see $log" >&2
        fail=1
      fi
    done
    if [[ "$fail" -ne 0 ]]; then
      echo "[pgvector][ivf_search][ERROR] one or more search jobs failed" >&2
      exit 1
    fi
    # merge JSONs into results.json with per_template_results
    python - "$OUT_ROOT/pgvector_ivf" "$DATASET_KEY" "$TEST_SIZE" "$LISTS" "$NPROBE" "${MAXP}" <<'PY'
import json,glob,sys,os
out_dir, dataset_key, test_size, nlist, nprobe, maxp = sys.argv[1:7]
parts = sorted(glob.glob(os.path.join(out_dir, 'overscan*.json')))
merged = []
for p in parts:
    d = json.load(open(p))
    # extract overscan from filename
    base=os.path.basename(p)
    ov = int(base.replace('overscan','').replace('.json',''))
    merged.append({
        'dataset_key': dataset_key,
        'test_size': float(test_size),
        'strategy': 'ivf',
        'nlist': int(nlist),
        'nprobe': int(nprobe),
        **({'max_probes': int(maxp)} if (maxp and maxp!='') else {}),
        'overscan': int(ov),
        'per_template_results': d.get('results', {}),
    })
json.dump(merged, open(os.path.join(out_dir, 'results.json'), 'w'))
print(f"[pgvector][merge] wrote {out_dir}/results.json with {len(merged)} parts")
PY
    ;;

  hnsw_search)
    MT_LIST=$(python -c "import json; print(' '.join(str(x) for x in json.loads('$SCAN_ARR')))" )
    echo "[pgvector][hnsw_search] running in parallel: MAX_PROCS=${MAX_PROCS}"
    pids=()
    logs=()
    for st in $MT_LIST; do
      OUT_JSON="$OUT_ROOT/pgvector_hnsw/ef${EF}_scan${st}.json"
      log="$OUT_ROOT/pgvector_hnsw/ef${EF}_scan${st}.log"
      while [ "$(jobs -pr | wc -l)" -ge "$MAX_PROCS" ]; do wait -n || true; done
      echo "[pgvector][hnsw_search] launch ef_search=$EF max_scan_tuples=$st -> $OUT_JSON (log=$log)"
      python -u -m benchmark.complex_predicate.baselines.pgvector \
        exp_pgvector_complex \
        --dsn "$PG_DSN" \
        --strategy hnsw \
        --iter_mode relaxed_order \
        --dataset_key "$DATASET_KEY" \
        --test_size "$TEST_SIZE" \
        --k 10 \
        --m "$M" \
        --ef_construction "$EFC" \
        --ef_search "$EF" \
        --hnsw_max_scan_tuples "$st" \
        --output_path "$OUT_JSON" >"$log" 2>&1 &
      pids+=("$!")
      logs+=("$log")
    done
    # wait for all
    fail=0
    for i in "${!pids[@]}"; do
      pid="${pids[$i]}"; log="${logs[$i]}"
      if ! wait "$pid"; then
        echo "[pgvector][hnsw_search][ERROR] job pid=$pid failed; see $log" >&2
        fail=1
      fi
    done
    if [[ "$fail" -ne 0 ]]; then
      echo "[pgvector][hnsw_search][ERROR] one or more search jobs failed" >&2
      exit 1
    fi
    # merge JSONs into results.json with per_template_results
    python - "$OUT_ROOT/pgvector_hnsw" "$DATASET_KEY" "$TEST_SIZE" "$M" "$EFC" "$EF" <<'PY'
import json,glob,sys,os
out_dir, dataset_key, test_size, m, efc, ef = sys.argv[1:7]
parts = sorted(glob.glob(os.path.join(out_dir, 'ef*_scan*.json')))
merged = []
for p in parts:
    d = json.load(open(p))
    base=os.path.basename(p)
    # efXXX_scanYYY.json
    try:
        scan = int(base.split('_scan')[1].split('.')[0])
    except Exception:
        scan = None
    merged.append({
        'dataset_key': dataset_key,
        'test_size': float(test_size),
        'strategy': 'hnsw',
        'm': int(m),
        'efc': int(efc),
        'ef_search': int(ef),
        'max_scan_tuples': (int(scan) if scan is not None else None),
        'per_template_results': d.get('results', {}),
    })
json.dump(merged, open(os.path.join(out_dir, 'results.json'), 'w'))
print(f"[pgvector][merge] wrote {out_dir}/results.json with {len(merged)} parts")
PY
    ;;

  *)
    echo "[pgvector][ERR] unknown task: $TASK" >&2; usage; exit 1
    ;;
esac

echo "[pgvector] done: $TASK"
