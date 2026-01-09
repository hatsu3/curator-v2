#!/bin/bash

# Ablation Study Benchmark Script
# Generates ablation figures (structural constraint and skewness analysis)
# Usage: ./run_ablation.sh <mode> [output_dir]
#        mode: "plot" to generate figures from existing results
#        output_dir: optional output directory (default: output/ablation)

set -e  # Exit on any error

# Function to show usage
show_usage() {
    echo "Usage: $0 <mode> [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  mode         Operation mode:"
    echo "               - plot (generate figures from existing results)"
    echo ""
    echo "  output_dir   Output directory for figures (default: output/ablation)"
    echo ""
    echo "Examples:"
    echo "  $0 plot"
    echo "  $0 plot output/ablation"
    echo ""
    echo "Expected input directories:"
    echo "  - output/overall_results/curator_opt/       (structural constraint - curator)"
    echo "  - output/structural/per_label_curator/      (structural constraint - unconstrained)"
    echo "  - output/skewness/                          (skewness analysis)"
    echo "  - output/scalability_nvec/curator/          (latency breakdown)"
    echo ""
    echo "Output:"
    echo "  - <output_dir>/figs/structure_and_skewness.pdf"
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

MODE="$1"
OUTPUT_DIR="${2:-output/ablation}"

# Validate mode
VALID_MODES=("plot")
if [[ ! " ${VALID_MODES[@]} " =~ " ${MODE} " ]]; then
    echo "Error: Invalid mode '${MODE}'"
    echo "Valid options: ${VALID_MODES[@]}"
    exit 1
fi

# Activate conda environment
echo "Activating ann_bench2 conda environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ann_bench2

if [ "${MODE}" == "plot" ]; then
    echo "=== Ablation Plot Mode ==="
    echo "Generating ablation figure from existing results..."
    echo

    PLOT_OUTPUT="${OUTPUT_DIR}/figs/structure_and_skewness.pdf"
    echo "Plot will be saved to: ${PLOT_OUTPUT}"
    echo

    # Create output directory if it doesn't exist
    mkdir -p "${OUTPUT_DIR}/figs"

    python -m benchmark.ablation.plotting \
        plot_combined \
        --output_path "${PLOT_OUTPUT}"

    echo
    echo "=== Plot generated successfully ==="
    echo "Output: ${PLOT_OUTPUT}"
fi

echo
