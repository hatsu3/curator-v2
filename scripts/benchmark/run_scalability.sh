#!/bin/bash

# Scalability Benchmark Script
# Runs scalability experiments (memory vs nlabels) and generates figures
# Usage: ./run_scalability.sh <mode> [output_dir] [n_labels_list]
#        mode: baseline name, "all", or "plot"
#        output_dir: optional output directory (default: output/scalability)
#        n_labels_list: comma-separated list of n_labels values (default: 19,38,76,152,304,608,1216,2432,4864,9728)

set -e  # Exit on any error

# Default n_labels values for scalability experiments
DEFAULT_N_LABELS="19,38,76,152,304,608,1216,2432,4864,9728"

# Function to show usage
show_usage() {
    echo "Usage: $0 <mode> [output_dir] [n_labels_list]"
    echo ""
    echo "Arguments:"
    echo "  mode           Operation mode:"
    echo "                 - curator"
    echo "                 - per_label_hnsw"
    echo "                 - per_label_ivf"
    echo "                 - shared_hnsw"
    echo "                 - parlay_ivf"
    echo "                 - filtered_diskann"
    echo "                 - all (run all baselines)"
    echo "                 - plot (generate figures from existing results)"
    echo ""
    echo "  output_dir     Output directory (default: output/scalability)"
    echo "  n_labels_list  Comma-separated n_labels values (default: ${DEFAULT_N_LABELS})"
    echo ""
    echo "Examples:"
    echo "  $0 curator                           # Run curator for all n_labels"
    echo "  $0 all output/scalability            # Run all baselines"
    echo "  $0 curator output/scalability 19,38  # Run curator for n_labels=19,38"
    echo "  $0 plot                              # Generate figures from existing results"
    echo ""
    echo "Output structure:"
    echo "  <output_dir>/<baseline>/yfcc100m_test0.01/results_nlabels<N>.csv"
    echo ""
    echo "Figure output:"
    echo "  output/scalability_nvec/figs/revision_memory_vs_nvec_nlabel_combined.pdf"
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

MODE="$1"
OUTPUT_DIR="${2:-output/scalability}"
N_LABELS_LIST="${3:-${DEFAULT_N_LABELS}}"

# Validate mode
VALID_MODES=("curator" "per_label_hnsw" "per_label_ivf" "shared_hnsw" "parlay_ivf" "filtered_diskann" "all" "plot")
if [[ ! " ${VALID_MODES[@]} " =~ " ${MODE} " ]]; then
    echo "Error: Invalid mode '${MODE}'"
    echo "Valid options: ${VALID_MODES[@]}"
    exit 1
fi

# Activate conda environment
echo "Activating ann_bench2 conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ann_bench2

# Convert comma-separated n_labels to array
IFS=',' read -ra N_LABELS_ARRAY <<< "${N_LABELS_LIST}"

# Function to run a single baseline for all n_labels values
run_baseline() {
    local baseline="$1"
    local module_name=""
    local func_name=""

    case "${baseline}" in
        curator)
            module_name="benchmark.scalability.nlabels.baselines.curator"
            func_name="exp_curator_scalability"
            ;;
        per_label_hnsw)
            module_name="benchmark.scalability.nlabels.baselines.per_label_hnsw"
            func_name="exp_per_label_hnsw_scalability"
            ;;
        per_label_ivf)
            module_name="benchmark.scalability.nlabels.baselines.per_label_ivf"
            func_name="exp_per_label_ivf_scalability"
            ;;
        shared_hnsw)
            module_name="benchmark.scalability.nlabels.baselines.shared_hnsw"
            func_name="exp_shared_hnsw_scalability"
            ;;
        parlay_ivf)
            module_name="benchmark.scalability.nlabels.baselines.parlay_ivf"
            func_name="exp_parlay_ivf_scalability"
            ;;
        filtered_diskann)
            module_name="benchmark.scalability.nlabels.baselines.filtered_diskann"
            func_name="exp_filtered_diskann_scalability"
            ;;
        *)
            echo "Error: Unknown baseline '${baseline}'"
            exit 1
            ;;
    esac

    echo "=== Running ${baseline} scalability experiments ==="
    for n_labels in "${N_LABELS_ARRAY[@]}"; do
        local output_path="${OUTPUT_DIR}/${baseline}/yfcc100m_test0.01/results_nlabels${n_labels}.csv"
        echo "Running ${baseline} with n_labels=${n_labels}..."
        echo "Output: ${output_path}"

        python -m "${module_name}" "${func_name}" \
            --output_path "${output_path}" \
            --n_labels "${n_labels}"

        echo "Completed ${baseline} n_labels=${n_labels}"
        echo
    done
}

# Main execution
if [ "${MODE}" == "plot" ]; then
    echo "=== Scalability Plot Mode ==="
    echo "Generating scalability figure from existing results..."
    echo

    PLOT_OUTPUT="output/scalability_nvec/figs/revision_memory_vs_nvec_nlabel_combined.pdf"
    echo "Plot will be saved to: ${PLOT_OUTPUT}"
    echo

    python -m benchmark.scalability.nlabels.plotting.plotting \
        plot_memory_vs_nvec_nlabel_combined \
        --output_path "${PLOT_OUTPUT}"

    echo
    echo "=== Plot generated successfully ==="
    echo "Output: ${PLOT_OUTPUT}"

elif [ "${MODE}" == "all" ]; then
    echo "=== Running all scalability experiments ==="
    echo "Output directory: ${OUTPUT_DIR}"
    echo "n_labels values: ${N_LABELS_LIST}"
    echo

    for baseline in curator per_label_hnsw per_label_ivf shared_hnsw parlay_ivf filtered_diskann; do
        run_baseline "${baseline}"
    done

    echo "=== All scalability experiments completed ==="

else
    echo "=== Running ${MODE} scalability experiments ==="
    echo "Output directory: ${OUTPUT_DIR}"
    echo "n_labels values: ${N_LABELS_LIST}"
    echo

    run_baseline "${MODE}"

    echo "=== ${MODE} scalability experiments completed ==="
fi

echo
