#!/bin/bash

# Complex Predicate Parameter Sweep Evaluation Script
# Supports running individual baseline algorithms or all baselines for comparison
# Usage: ./run_complex_predicate_param_sweep.sh <baseline_name> [output_dir]
#        baseline_name: curator, curator_with_index, shared_hnsw, per_predicate_hnsw,
#                      shared_ivf, per_predicate_ivf, parlay_ivf, "all", or "plot"
#        output_dir: optional output directory (default: output/complex_predicate)

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 <baseline_name> [output_dir]"
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
    echo "  output_dir       Output directory (default: output/complex_predicate)"
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
OUTPUT_DIR="${2:-output/complex_predicate}"
PLOT_RESULTS=false

# Validate baseline name
VALID_BASELINES=("curator" "curator_with_index" "shared_hnsw" "per_predicate_hnsw" "shared_ivf" "per_predicate_ivf" "parlay_ivf" "acorn" "all" "plot")
if [[ ! " ${VALID_BASELINES[@]} " =~ " ${BASELINE_NAME} " ]]; then
    echo "Error: Invalid baseline name '${BASELINE_NAME}'"
    echo "Valid options: ${VALID_BASELINES[@]}"
    exit 1
fi

# Activate conda environment
echo "Activating ann_bench2 conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ann_bench2
echo "Active environment: $CONDA_DEFAULT_ENV"
echo

echo "=== Complex Predicate Search Evaluation ==="
echo "Baseline(s): ${BASELINE_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Timestamp: $(date)"
echo

# Configuration
DATASET_KEY="yfcc100m"
TEST_SIZE=0.01
TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# Common parameters across all algorithms (no quotes around JSON to avoid issues)
CPU_GROUPS='["0-3", "4-7", "8-11", "12-15"]'
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
    "conda_environment": "${CONDA_DEFAULT_ENV}"
  },
  "common_parameters": {
    "dataset_key": "${DATASET_KEY}",
    "test_size": ${TEST_SIZE},
    "cpu_groups": ${CPU_GROUPS},
    "templates": ${TEMPLATES},
    "n_filters_per_template": ${N_FILTERS_PER_TEMPLATE},
    "n_queries_per_filter": ${N_QUERIES_PER_FILTER},
    "gt_cache_dir": "${GT_CACHE_DIR}"
  },
  "algorithm_parameters": {
    "curator": {
      "nlist_space": [8, 16, 32],
      "max_sl_size_space": [64, 128, 256],
      "search_ef_space": [32, 64, 128, 256, 512],
      "beam_size_space": [1, 2, 4, 8],
      "variance_boost_space": [0.4]
    },
    "curator_with_index": {
      "nlist_space": [8, 16, 32],
      "max_sl_size_space": [64, 128, 256],
      "search_ef_space": [32, 64, 128, 256, 512],
      "beam_size_space": [1, 2, 4, 8],
      "variance_boost_space": [0.4]
    },
    "shared_hnsw": {
      "construction_ef_space": [16, 32, 64],
      "m_space": [16, 32, 64],
      "search_ef_space": [32, 64, 128, 256]
    },
    "per_predicate_hnsw": {
      "construction_ef_space": [8, 16, 32],
      "m_space": [8, 16, 32],
      "search_ef_space": [16, 32, 64, 128]
    },
    "shared_ivf": {
      "nlist_space": [200, 400, 800, 1600],
      "nprobe_space": [8, 16, 32, 64]
    },
    "per_predicate_ivf": {
      "nlist_space": [10, 20, 30, 40],
      "nprobe_space": [1, 2, 4, 8]
    },
    "parlay_ivf": {
      "ivf_cluster_size_space": [100, 500, 1000],
      "graph_degree_space": [8, 12, 16],
      "ivf_max_iter_space": [10],
      "ivf_search_radius_space": [500, 1000, 2000],
      "graph_search_L_space": [32, 64, 128],
      "construct_threads": 4
    },
    "acorn": {
      "m_space": [16, 32, 64],
      "gamma_space": [1, 10, 20],
      "m_beta_multiplier_space": [1, 2, 4],
      "search_ef_space": [16, 32, 64, 128, 256],
      "dataset_dir": "data/acorn/complex_predicate/yfcc100m_test0.01"
    }
  }
}
EOF

echo "Configuration saved to: ${CONFIG_FILE}"
echo

# Function to run a specific baseline
run_baseline() {
    local algo="$1"
    local algo_output_dir="${OUTPUT_DIR}/${algo}"

    echo "=== Running ${algo} ==="
    echo "Results will be saved to: ${algo_output_dir}"

    case $algo in
        "curator")
            python -m benchmark.complex_predicate.baselines.curator \
                exp_curator_complex_predicate_param_sweep \
                --cpu_groups "${CPU_GROUPS}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}" \
                --nlist_space "[8, 16, 32]" \
                --max_sl_size_space "[64, 128, 256]" \
                --search_ef_space "[32, 64, 128, 256, 512]" \
                --beam_size_space "[1, 2, 4, 8]" \
                --variance_boost_space "[0.4]" \
                --output_dir "${algo_output_dir}"
            ;;
        "curator_with_index")
            python -m benchmark.complex_predicate.baselines.curator_with_index \
                exp_curator_with_index_complex_predicate_param_sweep \
                --cpu_groups "${CPU_GROUPS}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}" \
                --nlist_space "[8, 16, 32]" \
                --max_sl_size_space "[64, 128, 256]" \
                --search_ef_space "[32, 64, 128, 256, 512]" \
                --beam_size_space "[1, 2, 4, 8]" \
                --variance_boost_space "[0.4]" \
                --output_dir "${algo_output_dir}"
            ;;
        "shared_hnsw")
            python -m benchmark.complex_predicate.baselines.shared_hnsw \
                exp_shared_hnsw_complex_predicate_param_sweep \
                --cpu_groups "${CPU_GROUPS}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}" \
                --construction_ef_space "[16, 32, 64]" \
                --m_space "[16, 32, 64]" \
                --search_ef_space "[32, 64, 128, 256]" \
                --output_dir "${algo_output_dir}"
            ;;
        "per_predicate_hnsw")
            python -m benchmark.complex_predicate.baselines.per_predicate_hnsw \
                exp_per_predicate_hnsw_complex_predicate_param_sweep \
                --cpu_groups "${CPU_GROUPS}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}" \
                --construction_ef_space "[8, 16, 32]" \
                --m_space "[8, 16, 32]" \
                --search_ef_space "[16, 32, 64, 128]" \
                --output_dir "${algo_output_dir}"
            ;;
        "shared_ivf")
            python -m benchmark.complex_predicate.baselines.shared_ivf \
                exp_shared_ivf_complex_predicate_param_sweep \
                --cpu_groups "${CPU_GROUPS}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}" \
                --nlist_space "[200, 400, 800, 1600]" \
                --nprobe_space "[8, 16, 32, 64]" \
                --output_dir "${algo_output_dir}"
            ;;
        "per_predicate_ivf")
            python -m benchmark.complex_predicate.baselines.per_predicate_ivf \
                exp_per_predicate_ivf_complex_predicate_param_sweep \
                --cpu_groups "${CPU_GROUPS}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}" \
                --nlist_space "[10, 20, 30, 40]" \
                --nprobe_space "[1, 2, 4, 8]" \
                --output_dir "${algo_output_dir}"
            ;;
        "parlay_ivf")
            echo "Preparing Parlay IVF dataset..."
            dataset_dir="data/parlay_ivf/complex_predicate"
            mkdir -p "${dataset_dir}"
            
            python -m benchmark.complex_predicate.baselines.parlay_ivf \
                write_dataset \
                --dataset_dir "${dataset_dir}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --overwrite true
            
            echo "Running Parlay IVF experiments..."
            python -m benchmark.complex_predicate.baselines.parlay_ivf \
                exp_parlay_ivf_complex_predicate_param_sweep \
                --cpu_groups "${CPU_GROUPS}" \
                --dataset_dir "${dataset_dir}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}" \
                --ivf_cluster_size_space "[100, 500, 1000]" \
                --graph_degree_space "[8, 12, 16]" \
                --ivf_max_iter_space "[10]" \
                --ivf_search_radius_space "[500, 1000, 2000]" \
                --graph_search_L_space "[32, 64, 128]" \
                --construct_threads 4 \
                --output_dir "${algo_output_dir}"
            ;;
        "acorn")
            echo "Preparing ACORN dataset..."
            dataset_dir="data/acorn/complex_predicate/yfcc100m_test0.01"
            python -m indexes.acorn \
                write_complex_predicate_dataset_by_key \
                --dataset_dir "${dataset_dir}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}"

            echo "Running ACORN experiments..."
            python -m benchmark.complex_predicate.baselines.acorn \
                exp_acorn_complex_predicate_param_sweep \
                --cpu_groups "${CPU_GROUPS}" \
                --dataset_key "${DATASET_KEY}" \
                --test_size ${TEST_SIZE} \
                --templates "${TEMPLATES}" \
                --n_filters_per_template ${N_FILTERS_PER_TEMPLATE} \
                --n_queries_per_filter ${N_QUERIES_PER_FILTER} \
                --gt_cache_dir "${GT_CACHE_DIR}" \
                --m_space "[16, 32, 64]" \
                --gamma_space "[1, 10, 20]" \
                --m_beta_multiplier_space "[1, 2, 4]" \
                --search_ef_space "[16, 32, 64, 128, 256]" \
                --dataset_dir "${dataset_dir}" \
                --output_dir "${algo_output_dir}"
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
    echo "Running all baseline algorithms..."
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

    # Force plotting to true for plot mode
    PLOT_RESULTS=true
else
    echo "Running single baseline: ${BASELINE_NAME}"
    echo

    run_baseline "${BASELINE_NAME}"

    echo "=== Evaluation Completed Successfully! ==="
    echo "Results saved to: ${OUTPUT_DIR}/${BASELINE_NAME}"
fi

# Generate plots if requested
if [ "$PLOT_RESULTS" == "true" ]; then
    echo
    echo "=== Generating Comparison Plots ==="

    PLOT_OUTPUT="${OUTPUT_DIR}/figs/comparison.pdf"
    echo "Plot will be saved to: ${PLOT_OUTPUT}"

    python -m benchmark.complex_predicate.plotting \
        plot_optimal_results \
        --output_dir "${OUTPUT_DIR}" \
        --templates "[\"OR\", \"AND\"]" \
        --output_path "${PLOT_OUTPUT}"

    if [ $? -eq 0 ]; then
        echo "Plots generated successfully: ${PLOT_OUTPUT}"
    else
        echo "Warning: Plot generation failed"
    fi
fi

echo
echo "Experiment configuration: ${CONFIG_FILE}"
