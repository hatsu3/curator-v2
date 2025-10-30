#!/usr/bin/env bash
set -euo pipefail

# Run pgvector single-label overall results on yfcc100m-10m or arxiv-large-10.
# Tasks:
#   - ivf_build  : bulk load + CREATE INDEX ivfflat (training only)
#   - ivf_search : run exp_pgvector_single sweeps over nprobe
#   - hnsw_build : insert_bench (non-durable UNLOGGED) to measure build from inserts
#   - hnsw_search: run exp_pgvector_single sweeps over ef_search
#
# Notes
# - This script does NOT create schema or reset the DB. For build tasks it checks
#   that the DB is reachable and the table is clean (items exists and is empty).
# - DSN comes from PG_DSN env or defaults to local curator_bench.

TASK="${1:-}"
DATASET="${2:-}"
MODE="${3:-iter}"   # iter | classic (ivf only)
OUTPUT_DIR="${4:-output/overall_results2}"
PARAMS_FILE="${5:-benchmark/overall_results/optimal_baseline_params.json}"

if [[ -z "$TASK" || -z "$DATASET" ]]; then
  echo "Usage: $0 <ivf_build|ivf_search|hnsw_build|hnsw_search> <yfcc100m|arxiv> [iter|classic] [output_dir] [params_file]" >&2
  exit 1
fi

# Activate conda environment
set +u
eval "$(conda shell.bash hook)"
conda activate ann_bench2
set -u

DSN="${PG_DSN:-postgresql://postgres:postgres@localhost:5432/curator_bench}"

# Build-time knobs
PARALLEL_MAINT_WORKERS=${PARALLEL_MAINT_WORKERS:-0}
MAINTENANCE_WORK_MEM=${MAINTENANCE_WORK_MEM:-64GB}

case "$DATASET" in
  yfcc100m)
    DATASET_KEY="yfcc100m-10m"
    TEST_SIZE=0.001
    DIM=192
    ;;
  arxiv)
    DATASET_KEY="arxiv-large-10"
    TEST_SIZE=0.005
    DIM=384
    ;;
  *)
    echo "Invalid dataset: $DATASET (expected yfcc100m|arxiv)" >&2
    exit 1
    ;;
esac

algo_dir_for() {
  local algo="$1"
  case "$algo" in
    hnsw) echo "pgvector_hnsw";;
    ivf)  echo "pgvector_ivf";;
  esac
}

OUT_BASE_HNSW="$OUTPUT_DIR/$(algo_dir_for hnsw)/${DATASET_KEY}_test${TEST_SIZE}"
OUT_BASE_IVF="$OUTPUT_DIR/$(algo_dir_for ivf)/${DATASET_KEY}_test${TEST_SIZE}"
mkdir -p "$OUT_BASE_HNSW" "$OUT_BASE_IVF"

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

check_db_reachable() {
  python - "$DSN" <<'PY'
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
  # Ensure items exists and is empty; do not modify DB here.
  python - "$DSN" <<'PY'
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

extract_param() {
  # $1: baseline JSON name, $2: dataset (yfcc100m|arxiv), $3: json path (dot keys), prints scalar
  local base="$1" ds="$2" path="$3"
  python - <<PY
import json,sys
p=json.load(open("$PARAMS_FILE"))
v=p.get("$base",{}).get("$ds",{})
for k in "$path".split('.'): v=v.get(k, {})
if isinstance(v, (int,float,str)): print(v)
else: print(json.dumps(v))
PY
}

# Read construction/search parameter spaces
HNSW_JSON_NAME="pgvector HNSW"
M=$(extract_param "$HNSW_JSON_NAME" "$DATASET" construction_params.m)
EFC=$(extract_param "$HNSW_JSON_NAME" "$DATASET" construction_params.construction_ef)
HNSW_EF_LIST=$(extract_param "$HNSW_JSON_NAME" "$DATASET" search_param_combinations.search_ef)
HNSW_MAX_TUPLES_LIST=$(extract_param "$HNSW_JSON_NAME" "$DATASET" search_param_combinations.max_scan_tuples)

IVF_JSON_NAME="pgvector IVF"
NLIST=$(extract_param "$IVF_JSON_NAME" "$DATASET" construction_params.nlist)
NPROBE_ITER=$(extract_param "$IVF_JSON_NAME" "$DATASET" search_param_combinations.nprobe_iter)
NPROBE_CLASSIC_LIST=$(extract_param "$IVF_JSON_NAME" "$DATASET" search_param_combinations.nprobe_classic)
IVF_MAX_PROBES_LIST=$(extract_param "$IVF_JSON_NAME" "$DATASET" search_param_combinations.max_probes)
IVF_OVERSCAN_LIST=$(extract_param "$IVF_JSON_NAME" "$DATASET" search_param_combinations.overscan)

echo "[pgvector][cfg] DSN=$DSN DATASET_KEY=$DATASET_KEY TEST_SIZE=$TEST_SIZE DIM=$DIM"
echo "[pgvector][cfg] HNSW m=$M efc=$EFC ef_search_list=$HNSW_EF_LIST max_scan_tuples_list=$HNSW_MAX_TUPLES_LIST"
echo "[pgvector][cfg] IVF nlist=$NLIST nprobe_iter=$NPROBE_ITER nprobe_classic_list=$NPROBE_CLASSIC_LIST max_probes_list=$IVF_MAX_PROBES_LIST overscan_list=${IVF_OVERSCAN_LIST}"

case "$TASK" in
  ivf_build)
    check_db_reachable
    check_clean_for_build
    python -m scripts.pgvector.load_dataset bulk \
      --dsn "$DSN" \
      --dataset_key "$DATASET_KEY" \
      --dim "$DIM" \
      --test_size "$TEST_SIZE" \
      --copy_format binary \
      --create_gin \
      --unlogged \
      --sync_commit_off \
      --build_index ivf \
      --lists "$NLIST" \
      --truncate
    ;;
  ivf_search)
    case "$MODE" in
      iter|classic) ;;
      *) echo "[pgvector][ivf_search][ERROR] mode must be 'iter' or 'classic'" >&2; exit 1;;
    esac
    case "$MODE" in
      iter)
        # Assume overscan list exists; nprobe and max_probes are fixed
        NP_FIXED=$NPROBE_ITER
        MP_FIXED=$(python -c "import json; a=json.loads('$IVF_MAX_PROBES_LIST'); print(a[0] if isinstance(a,list) and a else a)" )
        OV_LIST=$(python -c "import json; print(' '.join(str(x) for x in json.loads('$IVF_OVERSCAN_LIST')))" )
        echo "[pgvector][ivf_search][iter] MAX_PROCS=${MAX_PROCS} (nprobe=$NP_FIXED max_probes=$MP_FIXED)"
        pids=(); logs=()
        for ov in $OV_LIST; do
          out="$OUT_BASE_IVF/nprobe${NP_FIXED}_overscan${ov}.csv"
          while [ "$(jobs -pr | wc -l)" -ge "$MAX_PROCS" ]; do wait -n || true; done
          log="$OUT_BASE_IVF/nprobe${NP_FIXED}_overscan${ov}.log"
          echo "[pgvector][ivf_search] launch overscan=$ov -> $out (log=$log)"
          python -u -m benchmark.overall_results.baselines.pgvector exp_pgvector_single \
            --strategy ivf \
            --iter_search true \
            --dataset_key "$DATASET_KEY" \
            --test_size "$TEST_SIZE" \
            --k 10 \
            --lists "$NLIST" \
            --probes "$NP_FIXED" \
            --ivf_max_probes "$MP_FIXED" \
            --ivf_overscan "$ov" \
            --output_path "$out" >"$log" 2>&1 &
          pids+=("$!")
          logs+=("$log")
        done
        ;;
      classic)
        # Vary nprobe; iterative off, no overscan/max_probes
        NPS=$(python -c "import json; print(' '.join(str(x) for x in json.loads('$NPROBE_CLASSIC_LIST')))" )
        echo "[pgvector][ivf_search][classic] MAX_PROCS=${MAX_PROCS} (nprobe list=$NPS)"
        pids=(); logs=()
        for np in $NPS; do
          out="$OUT_BASE_IVF/nprobe${np}.csv"
          while [ "$(jobs -pr | wc -l)" -ge "$MAX_PROCS" ]; do wait -n || true; done
          log="$OUT_BASE_IVF/nprobe${np}.log"
          echo "[pgvector][ivf_search] launch nprobe=$np -> $out (log=$log)"
          python -u -m benchmark.overall_results.baselines.pgvector exp_pgvector_single \
            --strategy ivf \
            --iter_search false \
            --dataset_key "$DATASET_KEY" \
            --test_size "$TEST_SIZE" \
            --k 10 \
            --lists "$NLIST" \
            --probes "$np" \
            --output_path "$out" >"$log" 2>&1 &
          pids+=("$!")
          logs+=("$log")
        done
        ;;
    esac
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
    # merge only files for the selected mode; also write a mode-specific file
    python - "$OUT_BASE_IVF" "$MODE" <<'PY'
import pandas as pd,glob,sys,shutil,os
p=sys.argv[1]
mode=sys.argv[2]
if mode == 'iter':
    fs=sorted(glob.glob(p+"/nprobe*_overscan*.csv"))
    assert fs, f"no iterative files to merge under {p}"
    out=os.path.join(p,"results_iter.csv")
elif mode == 'classic':
    allfs=sorted(glob.glob(p+"/nprobe*.csv"))
    fs=[f for f in allfs if "_overscan" not in os.path.basename(f)]
    assert fs, f"no classic files to merge under {p}"
    out=os.path.join(p,"results_classic.csv")
else:
    raise SystemExit(f"unknown mode: {mode}")
pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).to_csv(out,index=False)
shutil.copyfile(out, os.path.join(p,"results.csv"))
print(f"[pgvector][merge] wrote {out} and updated results.csv ({len(fs)} parts)")
PY
    ;;
  hnsw_build)
    check_db_reachable
    check_clean_for_build
    python -m scripts.pgvector.load_dataset bulk \
      --dsn "$DSN" \
      --dataset_key "$DATASET_KEY" \
      --dim "$DIM" \
      --test_size "$TEST_SIZE" \
      --copy_format binary \
      --create_gin \
      --unlogged \
      --sync_commit_off \
      --build_index hnsw \
      --m "$M" \
      --efc "$EFC" \
      --truncate
    ;;
  hnsw_search)
    EF_LIST=$(python -c "import json; print(' '.join(str(x) for x in json.loads('$HNSW_EF_LIST')))" )
    MT_LIST=$(python -c "import json; print(' '.join(str(x) for x in json.loads('$HNSW_MAX_TUPLES_LIST')))" )
    echo "[pgvector][hnsw_search] running in parallel: MAX_PROCS=${MAX_PROCS}"
    pids=()
    logs=()
    for ef in $EF_LIST; do
      for mt in $MT_LIST; do
        out="$OUT_BASE_HNSW/ef${ef}_scan${mt}.csv"
        # throttle concurrency
        while [ "$(jobs -pr | wc -l)" -ge "$MAX_PROCS" ]; do
          wait -n || true
        done
        log="$OUT_BASE_HNSW/ef${ef}_scan${mt}.log"
        echo "[pgvector][hnsw_search] launch ef_search=$ef max_scan_tuples=$mt -> $out (log=$log)"
        python -u -m benchmark.overall_results.baselines.pgvector exp_pgvector_single \
          --strategy hnsw \
          --iter_search true \
          --dataset_key "$DATASET_KEY" \
          --test_size "$TEST_SIZE" \
          --k 10 \
          --m "$M" \
          --ef_construction "$EFC" \
          --ef_search "$ef" \
          --hnsw_max_scan_tuples "$mt" \
          --output_path "$out" >"$log" 2>&1 &
        pids+=("$!")
        logs+=("$log")
      done
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
    # merge
    python - "$OUT_BASE_HNSW" <<'PY'
import pandas as pd,glob,sys
p=sys.argv[1]
fs=sorted(glob.glob(p+"/ef*_scan*.csv"))
assert fs, f"no files to merge under {p}"
pd.concat([pd.read_csv(f) for f in fs],ignore_index=True).to_csv(p+"/results.csv",index=False)
print(f"[pgvector][merge] wrote {p}/results.csv with {len(fs)} parts")
PY
    ;;
  *)
    echo "Invalid task: $TASK (expected ivf_build|ivf_search|hnsw_build|hnsw_search)" >&2
    exit 1
    ;;
esac

echo "[pgvector] Done: $TASK on $DATASET ($DATASET_KEY)"
