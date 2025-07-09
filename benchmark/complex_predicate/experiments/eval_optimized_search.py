"""
Plot optimized vs unoptimized search comparison.

Example usage:

python -m benchmark.complex_predicate.baselines.curator \
    exp_curator_complex_predicate \
        --output_path curator_unoptimized.json \
        --nlist 16 \
        --max_sl_size 256 \
        --search_ef_space "[32, 64, 128, 256, 512]" \
        --beam_size_space "[4]" \
        --variance_boost_space "[0.4]" \
        --dataset_key yfcc100m \
        --test_size 0.01 \
        --templates '["AND {0} {1}", "OR {0} {1}"]' \
        --n_filters_per_template 10 \
        --n_queries_per_filter 100 \
        --gt_cache_dir "data/ground_truth/complex_predicate" \
        --use_optimized_search False \
        --templates '["AND {0} {1}", "OR {0} {1}"]' \
        --n_filters_per_template 10 \
        --n_queries_per_filter 100 \
        --gt_cache_dir "data/ground_truth/complex_predicate" \
        --use_optimized_search False

python -m benchmark.complex_predicate.baselines.curator \
    exp_curator_complex_predicate \
        --output_path curator_optimized.json \
        --nlist 16 \
        --max_sl_size 256 \
        --search_ef_space "[32, 64, 128, 256, 512]" \
        --beam_size_space "[4]" \
        --variance_boost_space "[0.4]" \
        --dataset_key yfcc100m \
        --test_size 0.01 \
        --templates '["AND {0} {1}", "OR {0} {1}"]' \
        --n_filters_per_template 10 \
        --n_queries_per_filter 100 \
        --gt_cache_dir "data/ground_truth/complex_predicate" \
        --use_optimized_search True \
        --templates '["AND {0} {1}", "OR {0} {1}"]' \
        --n_filters_per_template 10 \
        --n_queries_per_filter 100 \
        --gt_cache_dir "data/ground_truth/complex_predicate" \
        --use_optimized_search True

python -m benchmark.complex_predicate.experiments.eval_optimized_search \
    plot_optimized_vs_unoptimized \
    --unoptimized_results_path curator_unoptimized.json \
    --optimized_results_path curator_optimized.json \
    --output_path optimized_search_comparison.pdf
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_optimized_vs_unoptimized(
    unoptimized_results_path: str,
    optimized_results_path: str,
    output_path: str = "optimized_search_comparison.pdf",
    templates: List[str] = ["OR", "AND"],
    subplot_size: Tuple[float, float] = (3.0, 2.5),
    font_size: int = 14,
) -> None:
    """
    Plot recall vs latency comparison between optimized and unoptimized search.

    Parameters
    ----------
    unoptimized_results_path : str
        Path to JSON results file from curator.py with --use_optimized_search False
    optimized_results_path : str
        Path to JSON results file from curator.py with --use_optimized_search True
    output_path : str
        Path to save the plot image
    templates : List[str]
        List of templates to plot
    subplot_size : Tuple[float, float]
        Size of each subplot (width, height) in inches
    font_size : int
        Font size for the plot
    """
    print(f"=== Plotting Optimized vs Unoptimized Search Comparison ===")
    print(f"Unoptimized results: {unoptimized_results_path}")
    print(f"Optimized results: {optimized_results_path}")
    print(f"Templates: {templates}")
    print(f"Output path: {output_path}")
    print()

    # Load results
    with open(unoptimized_results_path, "r") as f:
        unoptimized_results = json.load(f)

    with open(optimized_results_path, "r") as f:
        optimized_results = json.load(f)

    # Extract data into pandas DataFrame
    df_unopt = _extract_dataframe(unoptimized_results, "Unoptimized")
    df_opt = _extract_dataframe(optimized_results, "Optimized")

    # Combine dataframes
    df = pd.concat([df_unopt, df_opt], ignore_index=True)

    if df.empty:
        print("No data found in results files!")
        return

    # Clean template names and filter to requested templates
    df["template"] = df["template"].str.split(" ").str[0]  # Remove {0} {1} parameters
    df = df[df["template"].isin(templates)]

    # Sort by search_ef for consistent ordering
    df = df.sort_values(["template", "mode", "search_ef"])

    print(f"Loaded {len(df)} data points")
    print(f"Templates: {df['template'].unique().tolist()}")
    print(f"Search EF range: {df['search_ef'].min()} - {df['search_ef'].max()}")

    # Set up plotting following repository conventions
    plt.rcParams.update({"font.size": font_size})
    fig_width = subplot_size[0] * len(templates)
    fig_height = subplot_size[1]
    fig, axes = plt.subplots(1, len(templates), figsize=(fig_width, fig_height))

    # Handle single template case
    if len(templates) == 1:
        axes = [axes]

    # Colors and styles following repository conventions
    mode_colors = {
        "Optimized": "#1f77b4",  # blue
        "Unoptimized": "#ff7f0e",  # orange
    }

    mode_markers = {
        "Optimized": "o",
        "Unoptimized": "^",
    }

    # Plot each template
    for i, (template, ax) in enumerate(zip(templates, axes)):
        template_data = df[df["template"] == template]

        if template_data.empty:
            ax.set_title(f"{template} - No Data")
            continue

        # Create line plot for each mode
        for mode in ["Unoptimized", "Optimized"]:
            mode_mask = template_data["mode"] == mode
            mode_data = template_data[mode_mask]
            if len(mode_data) > 0:
                sns.lineplot(
                    data=mode_data,
                    x="latency_ms",
                    y="recall",
                    color=mode_colors[mode],
                    marker=mode_markers[mode],
                    markersize=6,
                    linewidth=2,
                    label=mode,
                    ax=ax,
                )

        # Styling following repository conventions
        if i == 0:
            ax.set_ylabel("Recall@10")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Query Latency (ms)")
        ax.set_xscale("log")
        ax.set_title(f"{template} Template", fontsize=font_size - 2)
        ax.grid(visible=True, which="major", axis="both", linestyle="-", alpha=0.6)
        # Turn off minor ticks only for y-axis, keep x-axis minor ticks for log scale
        ax.tick_params(axis="y", which="minor", left=False)
        ax.tick_params(axis="x", which="minor", bottom=True, labelbottom=False)

        # Set specific x-axis limits and ticks for AND template
        if template == "AND":
            ax.set_xlim(0.01, 0.1)
            ax.set_xticks([0.01, 0.1])
            ax.set_xticklabels(["10⁻²", "10⁻¹"])

    # Create shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        legend = fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=len(handles),
            fontsize=font_size - 2,
            columnspacing=1.0,
            handletextpad=0.5,
        )

        # Remove individual legends
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()

    fig.tight_layout()

    # Save plot
    print(f"Saving plot to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if "legend" in locals():
        plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")
    else:
        plt.savefig(output_path, bbox_inches="tight")

    print("Plot saved successfully!")


def _extract_dataframe(results: List[Dict], mode: str) -> pd.DataFrame:
    """
    Extract data from benchmark results into a pandas DataFrame.

    Parameters
    ----------
    results : List[Dict]
        List of result dictionaries from curator.py
    mode : str
        Mode name (Optimized/Unoptimized)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: template, recall, latency_ms, search_ef, mode
    """
    data_rows = []

    for result in results:
        search_ef = result.get("search_ef", 0)
        per_template_results = result.get("per_template_results", {})

        for template, template_data in per_template_results.items():
            recall = template_data.get("recall_at_k", 0.0)
            latency_ms = template_data.get("query_lat_avg", 0.0) * 1000

            data_rows.append(
                {
                    "template": template,
                    "recall": recall,
                    "latency_ms": latency_ms,
                    "search_ef": search_ef,
                    "mode": mode,
                }
            )

    df = pd.DataFrame(data_rows)

    print(f"{mode} data extracted:")
    for template in df["template"].unique():
        template_data = df[df["template"] == template]
        avg_recall = template_data["recall"].mean()
        avg_latency = template_data["latency_ms"].mean()
        print(
            f"  {template}: avg_recall={avg_recall:.3f}, avg_latency={avg_latency:.2f}ms"
        )

    return df


if __name__ == "__main__":
    """
    python -m benchmark.complex_predicate.experiments.eval_optimized_search \
        plot_optimized_vs_unoptimized \
        --unoptimized_results_path curator_unoptimized.json \
        --optimized_results_path curator_optimized.json \
        --output_path optimized_search_comparison.pdf
    """
    fire.Fire()
