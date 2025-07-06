#! /bin/bash

# Recall vs Latency
python -m benchmark.overall_results.plotting.recall_vs_latency2 plot_recall_vs_latency \
    --results_dir "output/overall_results2" \
    --dataset_name "yfcc100m" \
    --output_path "output/overall_results2/figs/recall_vs_latency_yfcc100m.pdf"

python -m benchmark.overall_results.plotting.recall_vs_latency2 plot_recall_vs_latency \
    --results_dir "output/overall_results2" \
    --dataset_name "arxiv" \
    --output_path "output/overall_results2/figs/recall_vs_latency_arxiv.pdf"


# Memory vs Latency vs Build Time
python -m benchmark.overall_results.plotting.memory_vs_latency2 generate_annotation_config \
    --results_dir "output/overall_results2" \
    --output_path "benchmark/overall_results/plotting/annotation_offsets_sample.yaml" \
    --dataset_names "['yfcc100m', 'arxiv']" \
    --selectivity_threshold 0.15 \
    --target_recall 0.9

python -m benchmark.overall_results.plotting.memory_vs_latency2 plot_memory_vs_latency_vs_build_time \
    --results_dir "output/overall_results2" \
    --dataset_names "['yfcc100m', 'arxiv']" \
    --selectivity_threshold 0.15 \
    --target_recall 0.9 \
    --output_path "output/overall_results2/figs/memory_vs_latency.pdf" \
    --annotation_config_path "benchmark/overall_results/plotting/annotation_offsets_sample.yaml"


# Build Time
python -m benchmark.overall_results.plotting.build_time2 plot_construction_time \
    --output_dir "output/overall_results2" \
    --datasets '["yfcc100m", "arxiv"]' \
    --output_path "output/overall_results2/figs/build_time.pdf"


# Memory Footprint
python -m benchmark.overall_results.plotting.memory_footprint2 plot_memory_footprint \
    --output_dir "output/overall_results2" \
    --datasets '["yfcc100m", "arxiv"]' \
    --output_path "output/overall_results2/figs/memory_footprint.pdf"


# Complex Predicate
python -m benchmark.complex_predicate.plotting plot_optimal_results_clean \
    --output_dir "output/complex_predicate_optimal" \
    --output_path "output/complex_predicate_optimal/figs/recall_vs_latency.pdf"
