#!/bin/bash

# Complex Predicate Optimal Parameters Evaluation Script
# Uses optimal construction parameters from optimal_complex_predicate_params.json
# Usage: ./run_complex_predicate.sh <baseline_name> [output_dir] [cpu_cores] [params_file]
#        baseline_name: curator, curator_with_index, shared_hnsw, per_predicate_hnsw,
#                      shared_ivf, per_predicate_ivf, parlay_ivf, acorn, "all", or "plot"
#        output_dir: optional output directory (default: output/complex_predicate_optimal)
#        cpu_cores: optional CPU cores to use with taskset (e.g., "0-3" or "1,3,5", default: no affinity)
#        params_file: optional path to optimal parameters JSON file (default: benchmark/complex_predicate/optimal_complex_predicate_params.json)

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 <baseline_name> [output_dir] [cpu_cores] [params_file]"
    echo ""
    echo "Arguments:"
    echo "  baseline_name    Algorithm to run:"
    echo "                   - curator"
    echo "                   - curator_with_index"
    echo "                   - shared_hnsw"
    echo "                   - per_predicate_hnsw"
    echo "                   - shared_ivf"
    echo "                   - per_predicate_ivf"
    echo "                   - parlay_ivf"
    echo "                   - acorn"
    echo "                   - all (runs all algorithms)"
    echo "                   - plot (only generate plots from existing results)"
    echo ""
    echo "  output_dir       Output directory (default: output/complex_predicate_optimal)"
    echo "  cpu_cores        CPU cores for taskset (e.g., '0-3' or '1,3,5', default: no affinity)"
    echo "  params_file      Path to optimal parameters JSON (default: benchmark/complex_predicate/optimal_complex_predicate_params.json)"
    echo ""
    echo "Examples:"
    echo "  $0 curator"
    echo "  $0 all /path/to/results"
    echo "  $0 shared_hnsw output/my_experiment"
    echo "  $0 plot output/existing_results"
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

BASELINE_NAME="$1"
OUTPUT_DIR="${2:-output/complex_predicate_optimal}"
CPU_CORES="${3:-}"
PARAMS_FILE="${4:-benchmark/complex_predicate/optimal_baseline_params.json}"

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

# Validate baseline name
VALID_BASELINES=("curator" "curator_with_index" "shared_hnsw" "per_predicate_hnsw" "shared_ivf" "per_predicate_ivf" "parlay_ivf" "acorn" "all" "plot")
if [[ ! " ${VALID_BASELINES[@]} " =~ " ${BASELINE_NAME} " ]]; then
    echo "Error: Invalid baseline name '${BASELINE_NAME}'"
    echo "Valid options: ${VALID_BASELINES[@]}"
    exit 1
fi

# Setup CPU affinity
if [ -n "${CPU_CORES}" ]; then
    TASKSET_CMD="taskset -c ${CPU_CORES}"
    NUM_THREADS=$(count_cpus_from_taskset "${CPU_CORES}")
    echo "CPU affinity will be set to cores: ${CPU_CORES} (${NUM_THREADS} cores)"
else
    TASKSET_CMD=""
    NUM_THREADS=$(nproc)
    echo "No CPU affinity specified - using all available cores (${NUM_THREADS} cores)"
fi

# Activate conda environment
echo "Activating ann_bench2 conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ann_bench2
echo "Active environment: $CONDA_DEFAULT_ENV"
echo

echo "=== Complex Predicate Optimal Parameters Evaluation ==="
echo "Baseline(s): ${BASELINE_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "CPU Cores: ${CPU_CORES:-all available}"
echo "Threads: ${NUM_THREADS}"
echo "Parameters File: ${PARAMS_FILE}"
echo "Timestamp: $(date)"
echo

# Configuration
DATASET_KEY="yfcc100m"
TEST_SIZE=0.01
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Common parameters
TEMPLATES='["AND {0} {1}", "OR {0} {1}"]'
N_FILTERS_PER_TEMPLATE=10
N_QUERIES_PER_FILTER=100
GT_CACHE_DIR="data/ground_truth/complex_predicate"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Save configuration to JSON file
CONFIG_FILE="${OUTPUT_DIR}/experiment_config.json"
cat > "${CONFIG_FILE}" << EOF
{
  "experiment_info": {
    "timestamp": "${TIMESTAMP}",
    "baseline_requested": "${BASELINE_NAME}",
    "output_directory": "${OUTPUT_DIR}",
    "conda_environment": "${CONDA_DEFAULT_ENV}",
    "optimal_params_source": "${PARAMS_FILE}",
    "cpu_cores": "${CPU_CORES}",
    "num_threads": ${NUM_THREADS}
  },
  "common_parameters": {
    "dataset_key": "${DATASET_KEY}",
    "test_size": ${TEST_SIZE},
    "templates": ${TEMPLATES},
    "n_filters_per_template": ${N_FILTERS_PER_TEMPLATE},
    "n_queries_per_filter": ${N_QUERIES_PER_FILTER},
    "gt_cache_dir": "${GT_CACHE_DIR}"
  }
}
EOF

echo "Configuration saved to: ${CONFIG_FILE}"
echo

# Function to run a specific baseline with optimal parameters
run_baseline() {
    local algo="$1"
    local algo_output_dir="${OUTPUT_DIR}/${algo}"

    echo "=== Running ${algo} with Optimal Parameters ==="
    echo "Results will be saved to: ${algo_output_dir}"

    # Map algorithm names to JSON names
    case $algo in
        "curator")
            baseline_json_name="Curator"
            ;;
        "curator_with_index")
            baseline_json_name="Curator (Indexed)"
            ;;
        "shared_hnsw")
            baseline_json_name="Shared HNSW"
            ;;
        "per_predicate_hnsw")
            baseline_json_name="Per-Predicate HNSW"
            ;;
        "shared_ivf")
            baseline_json_name="Shared IVF"
            ;;
        "per_predicate_ivf")
            baseline_json_name="Per-Predicate IVF"
            ;;
        "parlay_ivf")
            baseline_json_name="Parlay IVF"
            ;;
        "acorn")
            baseline_json_name="ACORN"
            ;;
        *)
            echo "Error: Unknown algorithm: $algo"
            return 1
            ;;
    esac
    
    # Extract optimal parameters from JSON for each template
    echo "Extracting optimal parameters for ${baseline_json_name} on ${DATASET_KEY}..."
    
    # Create a combined parameter file for this algorithm
    mkdir -p "${algo_output_dir}"
    local combined_params_file="${algo_output_dir}/optimal_parameters.json"
    
    # Initialize the combined parameters structure
    echo "{\"templates\": {}}" > "${combined_params_file}"
    
    # Extract optimal parameters from OR template (used for all templates)
    echo "  Extracting optimal parameters from OR template..."
    
    # Verify that only OR template exists and extract its parameters
    ${TASKSET_CMD} python -c "
import json
import sys

# Load optimal parameters
with open('${PARAMS_FILE}', 'r') as f:
    params = json.load(f)

baseline_name = '${baseline_json_name}'
dataset = '${DATASET_KEY}'

if baseline_name not in params:
    print('ERROR: Baseline not found in optimal params', file=sys.stderr)
    sys.exit(1)

if dataset not in params[baseline_name]:
    print('ERROR: Dataset not found for baseline', file=sys.stderr)
    sys.exit(1)

# Check that only OR template exists (for simplicity)
templates = list(params[baseline_name][dataset].keys())
if len(templates) != 1:
    print(f'ERROR: Expected single template, but found: {templates}', file=sys.stderr)
    sys.exit(1)

# Extract template parameters
template_params = params[baseline_name][dataset][templates[0]]
construction_params = template_params['construction_params']
search_param_combinations = template_params['search_param_combinations']

print('CONSTRUCTION:' + json.dumps(construction_params))
print('SEARCH:' + json.dumps(search_param_combinations))
" > "${algo_output_dir}/extracted_params.txt"
    
    if [ $? -ne 0 ]; then
        echo "Error: Could not extract optimal parameters"
        return 1
    fi
    
    # Parse the extracted parameters
    construction_params=$(grep "^CONSTRUCTION:" "${algo_output_dir}/extracted_params.txt" | sed 's/^CONSTRUCTION://')
    search_param_combinations=$(grep "^SEARCH:" "${algo_output_dir}/extracted_params.txt" | sed 's/^SEARCH://')
    
    echo "    Construction params: ${construction_params}"
    echo "    Search parameter spaces: ${search_param_combinations}"
    
    echo "Combined parameters saved to: ${combined_params_file}"
    
    # Run the algorithm with optimal parameters
    case $algo in
        "curator")
            # Extract individual construction parameters (now includes beam_size and variance_boost)
            nlist=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['nlist'])")
            max_sl_size=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['max_sl_size'])")
            beam_size=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print([json.load(sys.stdin)['beam_size']])")
            variance_boost=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print([json.load(sys.stdin)['variance_boost']])")
            
            # Extract search parameter arrays (now only search_ef)
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['search_ef'])")
            
            ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.curator \
                exp_curator_complex_predicate \
                --output_path "${algo_output_dir}/results.json" \
                --nlist ${nlist} \
                --max_sl_size ${max_sl_size} \
                --search_ef_space "${search_ef_space}" \
                --beam_size_space "${beam_size}" \
                --variance_boost_space "${variance_boost}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"
            ;;
        "curator_with_index")
            # Extract individual construction parameters (now includes beam_size and variance_boost)
            nlist=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['nlist'])")
            max_sl_size=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['max_sl_size'])")
            beam_size=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print([json.load(sys.stdin)['beam_size']])")
            variance_boost=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print([json.load(sys.stdin)['variance_boost']])")
            
            # Extract search parameter arrays (now only search_ef)
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['search_ef']))")
            
            ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.curator_with_index \
                exp_curator_with_index_complex_predicate \
                --output_path "${algo_output_dir}/results.json" \
                --nlist ${nlist} \
                --max_sl_size ${max_sl_size} \
                --beam_size_space "${beam_size}" \
                --variance_boost_space "${variance_boost}" \
                --search_ef_space "${search_ef_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"
            ;;
        "shared_hnsw")
            # Extract individual construction parameters
            construction_ef=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['construction_ef'])")
            m=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['m'])")
            
            # Extract search parameter arrays
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['search_ef'])")
            
            ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.shared_hnsw \
                exp_shared_hnsw_complex_predicate \
                --output_path "${algo_output_dir}/results.json" \
                --construction_ef ${construction_ef} \
                --m ${m} \
                --search_ef_space "${search_ef_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"
            ;;
        "per_predicate_hnsw")
            # Extract individual construction parameters
            construction_ef=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['construction_ef'])")
            m=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['m'])")
            
            # Extract search parameter arrays
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['search_ef'])")
            
            ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.per_predicate_hnsw \
                exp_per_predicate_hnsw_complex_predicate \
                --output_path "${algo_output_dir}/results.json" \
                --construction_ef ${construction_ef} \
                --m ${m} \
                --search_ef_space "${search_ef_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"
            ;;
        "shared_ivf")
            # Extract individual construction parameters
            nlist=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['nlist'])")
            
            # Extract search parameter arrays
            nprobe_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['nprobe']))")
            
            # For shared IVF, we run each nprobe value separately
            for nprobe in $(echo "${nprobe_space}" | ${TASKSET_CMD} python -c "import json, sys; [print(x) for x in json.load(sys.stdin)]"); do
                ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.shared_ivf \
                    exp_shared_ivf_complex_predicate \
                    --output_path "${algo_output_dir}/results_nprobe${nprobe}.json" \
                    --nlist ${nlist} \
                    --nprobe ${nprobe} \
                    --dataset_key "${DATASET_KEY}" \
                    --test_size ${TEST_SIZE} \
                    --templates "${TEMPLATES}" \
                    --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                    --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                    --gt_cache_dir "${GT_CACHE_DIR}"
            done

            # Combine results into a single file
            echo "Combining results into ${algo_output_dir}/results.json"
            ${TASKSET_CMD} python -c "
import json
import glob
from pathlib import Path

output_dir = Path('${algo_output_dir}')
result_files = list(output_dir.glob('results_nprobe*.json'))

if not result_files:
    print('Warning: No nprobe result files found')
    exit(1)

combined_results = []
for result_file in sorted(result_files):
    with open(result_file, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            combined_results.extend(data)
        else:
            combined_results.append(data)

# Save combined results
with open(output_dir / 'results.json', 'w') as f:
    json.dump(combined_results, f, indent=2)
"
            ;;
        "per_predicate_ivf")
            # Extract individual construction parameters
            nlist=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['nlist'])")
            
            # Extract search parameter arrays
            nprobe_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['nprobe']))")
            
            ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.per_predicate_ivf \
                exp_per_predicate_ivf_complex_predicate \
                --output_path "${algo_output_dir}/results.json" \
                --nlist ${nlist} \
                --nprobe_space "${nprobe_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"
            ;;
        "parlay_ivf")
            echo "Preparing Parlay IVF dataset..."
            dataset_dir="data/parlay_ivf/complex_predicate"
            mkdir -p "${dataset_dir}"
            
            ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.parlay_ivf \
                write_dataset \
                --dataset_dir "${dataset_dir}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --overwrite true
            
            echo "Running Parlay IVF experiments with optimal parameters..."
            
            # Extract individual construction parameters
            ivf_cluster_size=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['ivf_cluster_size'])")
            graph_degree=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['graph_degree'])")
            ivf_max_iter=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['ivf_max_iter'])")
            
            # Extract search parameter arrays
            ivf_search_radius_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['ivf_search_radius']))")
            graph_search_L_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['graph_search_L']))")
            
            ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.parlay_ivf \
                exp_parlay_ivf_complex_predicate \
                --output_path "${algo_output_dir}/results.json" \
                --dataset_dir "${dataset_dir}" \
                --ivf_cluster_size ${ivf_cluster_size} \
                --graph_degree ${graph_degree} \
                --ivf_max_iter ${ivf_max_iter} \
                --ivf_search_radius_space "${ivf_search_radius_space}" \
                --graph_search_L_space "${graph_search_L_space}" \
                --construct_threads ${NUM_THREADS} \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"
            ;;
        "acorn")
            echo "Preparing ACORN dataset..."
            dataset_dir="data/acorn/complex_predicate/${DATASET_KEY}_test${TEST_SIZE}"
            ${TASKSET_CMD} python -m indexes.acorn \
                write_complex_predicate_dataset_by_key \
                --dataset_dir "${dataset_dir}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"

            echo "Running ACORN experiments with optimal parameters..."
            
            # Extract individual construction parameters
            m=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['m'])")
            gamma=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['gamma'])")
            m_beta=$(echo "${construction_params}" | ${TASKSET_CMD} python -c "import json, sys; print(json.load(sys.stdin)['m_beta'])")
            
            # Extract search parameter arrays
            search_ef_space=$(echo "${search_param_combinations}" | ${TASKSET_CMD} python -c "import json, sys; print(json.dumps(json.load(sys.stdin)['search_ef']))")
            
            # ACORN needs special index directory
            index_dir="${algo_output_dir}/index"
            
            ${TASKSET_CMD} python -m benchmark.complex_predicate.baselines.acorn \
                exp_acorn_complex_predicate \
                --output_path "${algo_output_dir}/results.json" \
                --dataset_dir "${dataset_dir}" \
                --index_dir "${index_dir}" \
                --m ${m} \
                --gamma ${gamma} \
                --m_beta ${m_beta} \
                --search_ef_space "${search_ef_space}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"
            ;;
        *)
            echo "Error: Unknown algorithm: $algo"
            return 1
            ;;
    esac

    echo "Completed: ${algo}"
    echo
}

# Run requested baseline(s)
if [ "${BASELINE_NAME}" == "all" ]; then
    echo "Running all baseline algorithms with optimal parameters..."
    echo

    ALL_ALGORITHMS=("curator" "curator_with_index" "shared_hnsw" "per_predicate_hnsw" "shared_ivf" "per_predicate_ivf" "parlay_ivf" "acorn")

    for algo in "${ALL_ALGORITHMS[@]}"; do
        run_baseline "$algo"
    done

    echo "=== All Baseline Evaluations Completed Successfully! ==="
    echo "Results saved to: ${OUTPUT_DIR}"
    echo
    echo "Individual algorithm results:"
    for algo in "${ALL_ALGORITHMS[@]}"; do
        echo "  ${algo}: ${OUTPUT_DIR}/${algo}"
    done
elif [ "${BASELINE_NAME}" == "plot" ]; then
    echo "Skipping experiments - only generating plots from existing results..."
    echo

    # Generate plots from optimal parameter results
    echo "=== Generating Comparison Plots ==="

    PLOT_OUTPUT="${OUTPUT_DIR}/figs/comparison.pdf"
    echo "Plot will be saved to: ${PLOT_OUTPUT}"

    python -m benchmark.complex_predicate.plotting \
        plot_optimal_results_clean \
        --output_dir "${OUTPUT_DIR}" \
        --templates "[\"OR\", \"AND\"]" \
        --output_path "${PLOT_OUTPUT}"

    if [ $? -eq 0 ]; then
        echo "Plots generated successfully: ${PLOT_OUTPUT}"
    else
        echo "Warning: Plot generation failed"
    fi
else
    echo "Running single baseline with optimal parameters: ${BASELINE_NAME}"
    echo

    run_baseline "${BASELINE_NAME}"

    echo "=== Evaluation Completed Successfully! ==="
    echo "Results saved to: ${OUTPUT_DIR}/${BASELINE_NAME}"
fi

echo
echo "Experiment configuration: ${CONFIG_FILE}" 
