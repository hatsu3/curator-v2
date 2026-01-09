#!/bin/bash

# Overall Results Evaluation Script with Optimal Parameters
# Uses optimal parameters from benchmark/overall_results/optimal_baseline_params.json
# Usage: ./run_overall_results.sh <baseline_name> <dataset> [output_dir] [cpu_cores] [params_file] [return_verbose]
#        baseline_name: curator, per_label_hnsw, per_label_ivf, shared_hnsw,
#                      shared_ivf, parlay_ivf, filtered_diskann, acorn_1, 
#                      acorn_gamma, "all", or "plot"
#        dataset: yfcc100m, arxiv
#        output_dir: optional output directory (default: output/overall_results2)
#        cpu_cores: optional CPU cores to use with taskset (e.g., "0-3" or "1,3,5", default: no affinity)
#        params_file: optional path to optimal parameters JSON file (default: benchmark/overall_results/optimal_baseline_params.json)
#        return_verbose: optional verbose output flag (default: true)

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 <baseline_name> <dataset> [output_dir] [cpu_cores] [params_file] [return_verbose]"
    echo ""
    echo "Arguments:"
    echo "  baseline_name    Algorithm to run:"
    echo "                   - curator"
    echo "                   - per_label_hnsw"
    echo "                   - per_label_ivf"
    echo "                   - shared_hnsw"
    echo "                   - shared_ivf"
    echo "                   - parlay_ivf"
    echo "                   - filtered_diskann"
    echo "                   - acorn_1"
    echo "                   - acorn_gamma"
    echo "                   - all (runs all available algorithms for the dataset)"
    echo "                   - plot (only generate plots from existing results)"
    echo "                   - preproc_dataset (preprocess and cache dataset)"
    echo ""
    echo "  dataset          Dataset to use:"
    echo "                   - yfcc100m"
    echo "                   - arxiv"
    echo ""
    echo "  output_dir       Output directory (default: output/overall_results2)"
    echo ""
    echo "  cpu_cores        CPU cores to use with taskset (e.g., '0-3', '1,3,5')"
    echo "                   (default: no CPU affinity, uses all available cores)"
    echo "                   Also determines construct_threads for algorithms that support it"
    echo ""
    echo "  params_file      Path to optimal parameters JSON file"
    echo "                   (default: benchmark/overall_results/optimal_baseline_params.json)"
    echo ""
    echo "  return_verbose   Return verbose output from experiments (true/false, default: true)"
    echo ""
    echo "Examples:"
    echo "  $0 curator yfcc100m"
    echo "  $0 all arxiv /path/to/results"
    echo "  $0 shared_hnsw yfcc100m output/my_experiment"
    echo "  $0 plot yfcc100m output/existing_results"
    echo "  $0 preproc_dataset yfcc100m"
    echo "  $0 curator yfcc100m output/results '0-3'"
    echo "  $0 curator yfcc100m output/results '0-3' my_params.json"
    echo "  $0 all yfcc100m output/results '1,3,5,7' my_params.json true"
    echo "  $0 curator yfcc100m output/results '0-3' my_params.json false"
}

# Parse arguments
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

BASELINE_NAME="$1"
DATASET="$2"
OUTPUT_DIR="${3:-output/overall_results2}"
CPU_CORES="${4:-}"
PARAMS_FILE="${5:-benchmark/overall_results/optimal_baseline_params.json}"
RETURN_VERBOSE="${6:-True}"
PLOT_RESULTS=false

# Function to count CPUs from taskset format
count_cpus_from_taskset() {
    local cpu_spec="$1"
    if [ -z "$cpu_spec" ]; then
        echo $(nproc)
        return
    fi
    
    # Handle different taskset formats:
    # "0-3" -> 4 cores
    # "1,3,5" -> 3 cores  
    # "0-3,8-11" -> 8 cores
    local total=0
    
    # Split by comma and process each range/individual core
    IFS=',' read -ra RANGES <<< "$cpu_spec"
    for range in "${RANGES[@]}"; do
        if [[ "$range" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            # Range format like "0-3"
            local start=${BASH_REMATCH[1]}
            local end=${BASH_REMATCH[2]}
            total=$((total + end - start + 1))
        elif [[ "$range" =~ ^[0-9]+$ ]]; then
            # Individual core like "5"
            total=$((total + 1))
        fi
    done
    
    echo $total
}

# Set up taskset command prefix and determine CPU count
if [ -n "${CPU_CORES}" ]; then
    TASKSET_CMD="taskset -c ${CPU_CORES}"
    NUM_THREADS=$(count_cpus_from_taskset "${CPU_CORES}")
    echo "CPU affinity will be set to cores: ${CPU_CORES} (${NUM_THREADS} cores)"
else
    TASKSET_CMD=""
    NUM_THREADS=$(nproc)
    echo "No CPU affinity specified - using all available cores (${NUM_THREADS} cores)"
fi

# Validate baseline name
VALID_BASELINES=("curator" "per_label_hnsw" "per_label_ivf" "shared_hnsw" "shared_ivf" "parlay_ivf" "filtered_diskann" "acorn_1" "acorn_gamma" "all" "plot" "preproc_dataset")
if [[ ! " ${VALID_BASELINES[@]} " =~ " ${BASELINE_NAME} " ]]; then
    echo "Error: Invalid baseline name '${BASELINE_NAME}'"
    echo "Valid options: ${VALID_BASELINES[@]}"
    exit 1
fi

# Validate dataset
VALID_DATASETS=("yfcc100m" "arxiv")
if [[ ! " ${VALID_DATASETS[@]} " =~ " ${DATASET} " ]]; then
    echo "Error: Invalid dataset '${DATASET}'"
    echo "Valid options: ${VALID_DATASETS[@]}"
    exit 1
fi

# Validate params file exists
if [ ! -f "${PARAMS_FILE}" ]; then
    echo "Error: Parameters file not found: ${PARAMS_FILE}"
    exit 1
fi

# Validate return_verbose parameter
if [[ "${RETURN_VERBOSE}" != "True" && "${RETURN_VERBOSE}" != "False" ]]; then
    echo "Error: return_verbose must be 'True' or 'False', got: ${RETURN_VERBOSE}"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Activate conda environment
echo "Activating ann_bench2 conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ann_bench2
echo "Active environment: $CONDA_DEFAULT_ENV"

# Unset OMP_NUM_THREADS to allow FAISS to use all available threads
# This is important because some environments may set OMP_NUM_THREADS=1
# which would override our taskset CPU affinity and threading configuration
if [ -n "${OMP_NUM_THREADS}" ]; then
    echo "Warning: OMP_NUM_THREADS was set to '${OMP_NUM_THREADS}' - unsetting it to allow proper threading"
    unset OMP_NUM_THREADS
else
    echo "OMP_NUM_THREADS is not set - good for multi-threading"
fi
echo

echo "=== Overall Results Evaluation with Optimal Parameters ==="
echo "Baseline(s): ${BASELINE_NAME}"
echo "Dataset: ${DATASET}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Parameters File: ${PARAMS_FILE}"
if [ -n "${CPU_CORES}" ]; then
    echo "CPU Cores: ${CPU_CORES}"
else
    echo "CPU Cores: All available (no affinity set)"
fi
echo "Return Verbose: ${RETURN_VERBOSE}"
echo "Timestamp: $(date)"
echo

# Configuration based on dataset
if [ "${DATASET}" == "yfcc100m" ]; then
    DATASET_KEY="yfcc100m-10m"
    TEST_SIZE=0.001
elif [ "${DATASET}" == "arxiv" ]; then
    DATASET_KEY="arxiv-large-10"
    TEST_SIZE=0.005
fi

# Set dataset cache path
DATASET_CACHE_PATH="data/preprocessed/${DATASET_KEY}_test${TEST_SIZE}"

TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Save configuration to JSON file (append to maintain a list)
CONFIG_FILE="${OUTPUT_DIR}/experiment_config.json"

# Create new configuration entry
NEW_CONFIG=$(cat << EOF
{
  "timestamp": "${TIMESTAMP}",
  "baseline_requested": "${BASELINE_NAME}",
  "dataset": "${DATASET}",
  "dataset_key": "${DATASET_KEY}",
  "test_size": ${TEST_SIZE},
  "dataset_cache_path": "${DATASET_CACHE_PATH}",
  "output_directory": "${OUTPUT_DIR}",
  "conda_environment": "${CONDA_DEFAULT_ENV}",
  "optimal_params_source": "${PARAMS_FILE}",
  "cpu_cores": "${CPU_CORES}",
  "num_threads": ${NUM_THREADS},
  "return_verbose": $(echo "${RETURN_VERBOSE}" | tr '[:upper:]' '[:lower:]')
}
EOF
)

# Append to existing configuration list or create new one
${TASKSET_CMD} python -c "
import json
import os
from pathlib import Path

config_file = '${CONFIG_FILE}'
new_config = json.loads('''${NEW_CONFIG}''')

# Load existing configurations or create empty list
if os.path.exists(config_file):
    try:
        with open(config_file, 'r') as f:
            configs = json.load(f)
        # Ensure it's a list
        if not isinstance(configs, list):
            configs = [configs]  # Convert single config to list
    except (json.JSONDecodeError, FileNotFoundError):
        configs = []
else:
    configs = []

# Append new configuration
configs.append(new_config)

# Save updated configuration list
with open(config_file, 'w') as f:
    json.dump(configs, f, indent=2)
"

echo "Configuration added to: ${CONFIG_FILE}"
echo

# Function to extract parameters from JSON using Python
extract_params() {
    local baseline_name="$1"
    local dataset="$2"
    local param_type="$3"  # "construction_params" or "search_param_combinations"
    
    ${TASKSET_CMD} python -c "
import json
import sys

# Load optimal parameters
with open('${PARAMS_FILE}', 'r') as f:
    params = json.load(f)

baseline_name = '$baseline_name'
dataset = '$dataset'
param_type = '$param_type'

if baseline_name not in params:
    print('ERROR: Baseline not found in optimal params', file=sys.stderr)
    sys.exit(1)

if dataset not in params[baseline_name]:
    print('ERROR: Dataset not found for baseline', file=sys.stderr)
    sys.exit(1)

if param_type not in params[baseline_name][dataset]:
    print('ERROR: Parameter type not found', file=sys.stderr)
    sys.exit(1)

result = params[baseline_name][dataset][param_type]
print(json.dumps(result))
"
}

# Function to preprocess dataset
preproc_dataset() {
    echo "=== Preprocessing Dataset ==="
    echo "Dataset: ${DATASET}"
    echo "Dataset Key: ${DATASET_KEY}"
    echo "Test Size: ${TEST_SIZE}"
    echo "Cache Path: ${DATASET_CACHE_PATH}"
    echo
    
    ${TASKSET_CMD} python -m benchmark.overall_results.preproc_dataset \
        --dataset_key "${DATASET_KEY}" \
        --test_size ${TEST_SIZE} \
        --output_dir "${DATASET_CACHE_PATH}"
    
    echo "Dataset preprocessing completed!"
    echo "Cached dataset available at: ${DATASET_CACHE_PATH}"
    echo
}

# Function to run a specific baseline
run_baseline() {
    local algo="$1"
    local algo_output_dir="${OUTPUT_DIR}/${algo}/${DATASET_KEY}_test${TEST_SIZE}"
    
    # Create algorithm-specific output directory
    mkdir -p "${algo_output_dir}"
    
    # Set up per-baseline logging
    local log_file="${algo_output_dir}/$(date +%Y%m%d_%H%M%S).log"
    
    echo "=== Running ${algo} ==="
    echo "Results will be saved to: ${algo_output_dir}"
    echo "Log file: ${log_file}"
    
    # Redirect all output for this baseline to its own log file
    exec 3>&1 4>&2  # Save original stdout and stderr
    exec 1> >(tee -a "${log_file}") 2>&1
    
    # Map algorithm names to baseline names in JSON
    local baseline_json_name=""
    case $algo in
        "curator")
            baseline_json_name="Curator"
            ;;
        "per_label_hnsw")
            baseline_json_name="Per-Label HNSW"
            ;;
        "per_label_ivf")
            baseline_json_name="Per-Label IVF"
            ;;
        "shared_hnsw")
            baseline_json_name="Shared HNSW"
            ;;
        "shared_ivf")
            baseline_json_name="Shared IVF"
            ;;
        "parlay_ivf")
            baseline_json_name="Parlay IVF"
            ;;
        "filtered_diskann")
            baseline_json_name="Filtered DiskANN"
            ;;
        "acorn_1")
            baseline_json_name="ACORN-1"
            ;;
        "acorn_gamma")
            baseline_json_name="ACORN-gamma"
            ;;
        *)
            echo "Error: Unknown algorithm: $algo"
            return 1
            ;;
    esac
    
    # Extract optimal parameters from JSON
    echo "Extracting optimal parameters for ${baseline_json_name} on ${DATASET}..."
    
    construction_params=$(extract_params "${baseline_json_name}" "${DATASET}" "construction_params")
    if [ $? -ne 0 ]; then
        echo "Error: Could not extract construction parameters for ${baseline_json_name} on ${DATASET}"
        echo "Skipping ${algo}..."
        return 1
    fi
    
    search_param_combinations=$(extract_params "${baseline_json_name}" "${DATASET}" "search_param_combinations")
    if [ $? -ne 0 ]; then
        echo "Error: Could not extract search parameters for ${baseline_json_name} on ${DATASET}"
        echo "Skipping ${algo}..."
        return 1
    fi
    
    echo "Construction params: ${construction_params}"
    echo "Search param combinations: ${search_param_combinations}"
    
    # Save algorithm parameters to output directory
    local params_file="${algo_output_dir}/parameters.json"
    cat > "${params_file}" << EOF
{
  "algorithm": "${algo}",
  "baseline_json_name": "${baseline_json_name}",
  "dataset": "${DATASET}",
  "dataset_key": "${DATASET_KEY}",
  "test_size": ${TEST_SIZE},
  "dataset_cache_path": "${DATASET_CACHE_PATH}",
  "timestamp": "$(date +"%Y-%m-%d %H:%M:%S")",
  "construction_params": ${construction_params},
  "search_param_combinations": ${search_param_combinations},
  "experiment_config": {
    "cpu_cores": "${CPU_CORES}",
    "num_threads": ${NUM_THREADS},
    "return_verbose": $(echo "${RETURN_VERBOSE}" | tr '[:upper:]' '[:lower:]'),
    "params_source": "${PARAMS_FILE}"
  }
}
EOF
    echo "Algorithm parameters saved to: ${params_file}"
    
    case $algo in
        "curator")
            # Extract individual construction parameters
            nlist=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['nlist'])")
            max_sl_size=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['max_sl_size'])")
            
            # Extract search parameter arrays
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['search_ef']))")
            beam_size_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['beam_size']))")
            variance_boost_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['variance_boost']))")
            
            ${TASKSET_CMD} python -m benchmark.overall_results.baselines.curator \
                exp_curator_opt \
                --output_path "${algo_output_dir}/results.csv" \
                --dataset_cache_path "${DATASET_CACHE_PATH}" \
                --nlist ${nlist} \
                --max_sl_size ${max_sl_size} \
                --search_ef_space "${search_ef_space}" \
                --beam_size_space "${beam_size_space}" \
                --variance_boost_space "${variance_boost_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --return_verbose ${RETURN_VERBOSE}
            ;;
        "per_label_hnsw")
            construction_ef=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['construction_ef'])")
            m=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['m'])")
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['search_ef']))")
            
            ${TASKSET_CMD} python -m benchmark.overall_results.baselines.per_label_hnsw \
                exp_per_label_hnsw \
                --output_path "${algo_output_dir}/results.csv" \
                --dataset_cache_path "${DATASET_CACHE_PATH}" \
                --construction_ef ${construction_ef} \
                --m ${m} \
                --search_ef_space "${search_ef_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --return_verbose ${RETURN_VERBOSE}
            ;;
        "per_label_ivf")
            nlist=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['nlist'])")
            nprobe_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['nprobe']))")
            
            ${TASKSET_CMD} python -m benchmark.overall_results.baselines.per_label_ivf \
                exp_per_label_ivf \
                --output_path "${algo_output_dir}/results.csv" \
                --dataset_cache_path "${DATASET_CACHE_PATH}" \
                --nlist ${nlist} \
                --nprobe_space "${nprobe_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --return_verbose ${RETURN_VERBOSE}
            ;;
        "shared_hnsw")
            construction_ef=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['construction_ef'])")
            m=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['m'])")
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['search_ef']))")
            
            ${TASKSET_CMD} python -m benchmark.overall_results.baselines.shared_hnsw \
                exp_shared_hnsw \
                --output_path "${algo_output_dir}/results.csv" \
                --dataset_cache_path "${DATASET_CACHE_PATH}" \
                --construction_ef ${construction_ef} \
                --m ${m} \
                --search_ef_space "${search_ef_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --return_verbose ${RETURN_VERBOSE}
            ;;
        "shared_ivf")
            nlist=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['nlist'])")
            nprobe_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['nprobe']))")
            
            # For shared IVF, we run each nprobe value separately
            for nprobe in $(echo "${nprobe_space}" | ${TASKSET_CMD} python -c "import json, sys; [print(x) for x in json.load(sys.stdin)]"); do
                ${TASKSET_CMD} python -m benchmark.overall_results.baselines.shared_ivf \
                    exp_shared_ivf \
                    --output_path "${algo_output_dir}/results_nprobe${nprobe}.csv" \
                    --dataset_cache_path "${DATASET_CACHE_PATH}" \
                    --nlist ${nlist} \
                    --nprobe ${nprobe} \
                    --dataset_key "${DATASET_KEY}" \
                    --test_size ${TEST_SIZE} \
                    --return_verbose ${RETURN_VERBOSE}
            done
            
            # Merge all nprobe CSV files into a single results.csv
            echo "Merging shared_ivf results files..."
            ${TASKSET_CMD} python -c "
import pandas as pd
import glob
from pathlib import Path

# Find all results_nprobe*.csv files
csv_files = glob.glob('${algo_output_dir}/results_nprobe*.csv')
csv_files.sort()  # Sort for consistent ordering

if csv_files:
    # Read and concatenate all CSV files
    dfs = []
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged results
    output_path = '${algo_output_dir}/results.csv'
    merged_df.to_csv(output_path, index=False)
    
    print(f'Merged {len(csv_files)} files into {output_path}')
    print(f'Total rows: {len(merged_df)}')
else:
    print('No nprobe CSV files found to merge')
"
            ;;
        "parlay_ivf")
            cutoff=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['cutoff'])")
            ivf_cluster_size=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['ivf_cluster_size'])")
            graph_degree=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['graph_degree'])")
            ivf_max_iter=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['ivf_max_iter'])")
            ivf_search_radius_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['ivf_search_radius']))")
            graph_search_L_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['graph_search_L']))")
            
            # Prepare dataset for Parlay IVF
            echo "Preparing Parlay IVF dataset..."
            dataset_dir="data/parlay_ivf/overall_results"
            ${TASKSET_CMD} python -m benchmark.overall_results.baselines.parlay_ivf \
                write_dataset \
                --dataset_dir "${dataset_dir}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --overwrite true
            
            ${TASKSET_CMD} python -m benchmark.overall_results.baselines.parlay_ivf \
                exp_parlay_ivf \
                --output_path "${algo_output_dir}/results.csv" \
                --dataset_cache_path "${DATASET_CACHE_PATH}" \
                --dataset_dir "${dataset_dir}" \
                --cutoff ${cutoff} \
                --ivf_cluster_size ${ivf_cluster_size} \
                --graph_degree ${graph_degree} \
                --ivf_max_iter ${ivf_max_iter} \
                --ivf_search_radius_space "${ivf_search_radius_space}" \
                --graph_search_L_space "${graph_search_L_space}" \
                --construct_threads ${NUM_THREADS} \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --return_verbose ${RETURN_VERBOSE}
            ;;
        "filtered_diskann")
            graph_degree=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['graph_degree'])")
            ef_construct=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['ef_construct'])")
            alpha=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['alpha'])")
            ef_search_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['ef_search']))")
            
            # Set up index directory for caching
            index_cache_dir="${algo_output_dir}/index"
            
            # Check if index already exists and set skip_build accordingly
            if [ -d "${index_cache_dir}" ] && [ "$(ls -A "${index_cache_dir}" 2>/dev/null)" ]; then
                echo "Found existing index at ${index_cache_dir}, skipping build..."
                skip_build_flag="--skip_build true"
            else
                echo "No existing index found, will build new index..."
                skip_build_flag=""
            fi
            
            ${TASKSET_CMD} python -m benchmark.overall_results.baselines.filtered_diskann \
                exp_filtered_diskann \
                --output_path "${algo_output_dir}/results.csv" \
                --dataset_cache_path "${DATASET_CACHE_PATH}" \
                --graph_degree ${graph_degree} \
                --ef_construct ${ef_construct} \
                --alpha ${alpha} \
                --ef_search_space "${ef_search_space}" \
                --construct_threads ${NUM_THREADS} \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --return_verbose ${RETURN_VERBOSE} \
                --index_dir "${index_cache_dir}" \
                ${skip_build_flag}
            ;;
        "acorn_1"|"acorn_gamma")
            m=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['m'])")
            gamma=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['gamma'])")
            m_beta=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['m_beta'])")
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['search_ef']))")
            
            # Prepare dataset for ACORN
            echo "Preparing ACORN dataset..."
            dataset_dir="data/acorn/overall_results/${DATASET_KEY}_test${TEST_SIZE}"
            ${TASKSET_CMD} python -m indexes.acorn \
                write_dataset_by_key \
                --dataset_dir "${dataset_dir}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --overwrite true
            
            # Note: ACORN threading is now controlled via OMP_NUM_THREADS environment variable
            # The binary has been modified to respect the num_threads parameter
            
            ${TASKSET_CMD} python -m benchmark.overall_results.baselines.acorn \
                exp_acorn \
                --output_path "${algo_output_dir}/results.csv" \
                --dataset_cache_path "${DATASET_CACHE_PATH}" \
                --dataset_dir "${dataset_dir}" \
                --index_dir "${algo_output_dir}/index" \
                --m ${m} \
                --gamma ${gamma} \
                --m_beta ${m_beta} \
                --search_ef_space "${search_ef_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --return_verbose ${RETURN_VERBOSE}
            ;;
        *)
            echo "Error: Unknown algorithm: $algo"
            return 1
            ;;
    esac
    
    echo "Completed: ${algo}"
    echo
    
    # Restore original stdout and stderr
    exec 1>&3 2>&4
    exec 3>&- 4>&-
}

# Get list of available algorithms for the dataset
get_available_algorithms() {
    local dataset="$1"
    ${TASKSET_CMD} python -c "
import json

with open('${PARAMS_FILE}', 'r') as f:
    params = json.load(f)

algorithms = []
for baseline_name, baseline_data in params.items():
    if '$dataset' in baseline_data:
        # Map JSON names to script names
        name_mapping = {
            'Curator': 'curator',
            'Per-Label HNSW': 'per_label_hnsw',
            'Per-Label IVF': 'per_label_ivf',
            'Shared HNSW': 'shared_hnsw',
            'Shared IVF': 'shared_ivf',
            'Parlay IVF': 'parlay_ivf',
            'Filtered DiskANN': 'filtered_diskann',
            'ACORN-1': 'acorn_1',
            'ACORN-gamma': 'acorn_gamma'
        }
        if baseline_name in name_mapping:
            algorithms.append(name_mapping[baseline_name])

print(' '.join(algorithms))
"
}

# Run requested baseline(s)
if [ "${BASELINE_NAME}" == "all" ]; then
    echo "Running all available baseline algorithms for ${DATASET}..."
    echo
    
    AVAILABLE_ALGORITHMS=($(get_available_algorithms "${DATASET}"))
    
    if [ ${#AVAILABLE_ALGORITHMS[@]} -eq 0 ]; then
        echo "Error: No algorithms available for dataset ${DATASET}"
        exit 1
    fi
    
    echo "Available algorithms for ${DATASET}: ${AVAILABLE_ALGORITHMS[@]}"
    echo
    
    for algo in "${AVAILABLE_ALGORITHMS[@]}"; do
        run_baseline "$algo"
    done
    
    echo "=== All Available Baseline Evaluations Completed Successfully! ==="
    echo "Results saved to: ${OUTPUT_DIR}"
    echo
    echo "Individual algorithm results:"
    for algo in "${AVAILABLE_ALGORITHMS[@]}"; do
        echo "  ${algo}: ${OUTPUT_DIR}/${algo}/${DATASET_KEY}_test${TEST_SIZE}"
    done
elif [ "${BASELINE_NAME}" == "plot" ]; then
    echo "Skipping experiments - only generating plots from existing results..."
    echo
    
    # Force plotting to true for plot mode
    PLOT_RESULTS=true
elif [ "${BASELINE_NAME}" == "preproc_dataset" ]; then
    echo "Preprocessing dataset..."
    echo
    
    preproc_dataset
    
    echo "=== Dataset Preprocessing Completed Successfully! ==="
    echo "Dataset cached to: ${DATASET_CACHE_PATH}"
    exit 0
else
    echo "Running single baseline: ${BASELINE_NAME}"
    echo
    
    run_baseline "${BASELINE_NAME}"
    
    echo "=== Evaluation Completed Successfully! ==="
    echo "Results saved to: ${OUTPUT_DIR}/${BASELINE_NAME}/${DATASET_KEY}_test${TEST_SIZE}"
fi

# Generate plots if requested
if [ "$PLOT_RESULTS" == "true" ]; then
    echo
    echo "=== Generating Plots ==="

    mkdir -p "${OUTPUT_DIR}/figs"

    # Recall vs Latency (per-dataset, since this script doesn't support combined)
    echo "Generating recall vs latency plots..."
    for ds in yfcc100m arxiv; do
        python -m benchmark.overall_results.plotting.recall_vs_latency \
            plot_recall_vs_latency \
            --results_dir "${OUTPUT_DIR}" \
            --dataset_name "${ds}" \
            --output_path "${OUTPUT_DIR}/figs/recall_vs_latency_${ds}.pdf"
    done

    # Memory vs Latency (combined - both datasets)
    echo "Generating memory vs latency plot..."
    python -m benchmark.overall_results.plotting.memory_vs_latency \
        plot_memory_vs_latency_vs_build_time \
        --results_dir "${OUTPUT_DIR}" \
        --dataset_names '["yfcc100m", "arxiv"]' \
        --selectivity_threshold 0.15 \
        --target_recall 0.9 \
        --output_path "${OUTPUT_DIR}/figs/memory_vs_latency.pdf" \
        --annotation_config_path "benchmark/overall_results/plotting/annotation_offsets_sample.yaml"

    # Build Time (combined - both datasets)
    echo "Generating build time plot..."
    python -m benchmark.overall_results.plotting.build_time \
        plot_construction_time \
        --output_dir "${OUTPUT_DIR}" \
        --datasets '["yfcc100m", "arxiv"]' \
        --output_path "${OUTPUT_DIR}/figs/build_time.pdf"

    # Memory Footprint (combined - both datasets)
    echo "Generating memory footprint plot..."
    python -m benchmark.overall_results.plotting.memory_footprint \
        plot_memory_footprint \
        --output_dir "${OUTPUT_DIR}" \
        --datasets '["yfcc100m", "arxiv"]' \
        --output_path "${OUTPUT_DIR}/figs/memory_footprint.pdf"

    # Update Performance (combined - both datasets)
    echo "Generating update performance plot..."
    python -m benchmark.overall_results.plotting.update_perf \
        plot_update_results \
        --output_dir "${OUTPUT_DIR}" \
        --datasets '["yfcc100m", "arxiv"]' \
        --output_path "${OUTPUT_DIR}/figs/update_perf.pdf"

    echo "Plots saved to: ${OUTPUT_DIR}/figs/"
fi

echo
echo "Experiment configuration: ${CONFIG_FILE}"
echo "Optimal parameters source: ${PARAMS_FILE}"
echo "Timestamp: $(date)"
