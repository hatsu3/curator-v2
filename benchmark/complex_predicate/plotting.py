"""
Complex Predicate Plotting Script

This script provides plotting functionality for complex predicate experiments
run with optimal construction parameters (using run_complex_predicate.sh).

Since construction parameters are pre-determined from optimal baseline parameters,
this script focuses on visualizing search parameter trade-offs and comparing algorithms
at their optimal construction configurations.

Usage Examples:

1. Plot recall vs latency comparison using optimal parameters:
   python -m benchmark.complex_predicate.plotting plot_optimal_results \
       --output_dir "output/complex_predicate_optimal" \
       --output_path "output/complex_predicate_optimal/figs/recall_vs_latency.pdf"

2. Generate summary statistics:
   python -m benchmark.complex_predicate.plotting print_optimal_summary \
       --output_dir "output/complex_predicate_optimal" \
       --dataset_key "yfcc100m" \
       --test_size 0.01
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark.complex_predicate.dataset import ComplexPredicateDataset

# Global algorithm mappings for consistency with overall plotting
ALGORITHM_MAPPING = {
    "curator": "Curator",
    "curator_with_index": "Curator (Indexed)",
    "shared_hnsw": "Shared HNSW",
    "per_predicate_hnsw": "Per-Pred HNSW",
    "shared_ivf": "Shared IVF",
    "per_predicate_ivf": "Per-Pred IVF",
    "parlay_ivf": "Parlay IVF",
    "acorn": "ACORN",
}

# Global algorithm order for consistent plotting
ALGORITHM_ORDER = [
    "Curator",
    "Curator (Indexed)",
    "Shared HNSW",
    "Per-Pred HNSW",
    "Shared IVF",
    "Per-Pred IVF",
    "Parlay IVF",
    "ACORN",
]


def load_optimal_algo_results(
    base_output_dir: str | Path = "output/complex_predicate_optimal",
    algorithm_name: str = "curator",
) -> pd.DataFrame:
    """Load results from optimal parameter experiments.

    Args:
        base_output_dir: Base output directory containing algorithm results
        algorithm_name: Algorithm name (e.g., 'curator', 'shared_hnsw', etc.)

    Returns:
        DataFrame with results including template, recall, latency, and other metrics
    """
    algo_dir = Path(base_output_dir) / algorithm_name
    results_file = algo_dir / "results.json"

    if not results_file.exists():
        raise ValueError(f"Results file {results_file} does not exist")

    try:
        with open(results_file, "r") as f:
            data = json.load(f)

        all_results = []

        # Handle both single result and list of results
        if isinstance(data, list):
            results = data
        else:
            results = [data]

        for result in results:
            if "per_template_results" in result:
                for template, per_template_result in result[
                    "per_template_results"
                ].items():
                    # Extract configuration parameters from result
                    config_info = {
                        "template": template,
                        **per_template_result,
                    }

                    # Add any global parameters from the result
                    for key, value in result.items():
                        if key != "per_template_results" and not key.startswith("_"):
                            config_info[f"global_{key}"] = value

                    all_results.append(config_info)

        if not all_results:
            raise ValueError(f"No valid results found in {results_file}")

        all_results_df = pd.DataFrame(all_results)

        # Clean up template names for better display
        if "template" in all_results_df.columns:
            all_results_df["template"] = (
                all_results_df["template"].str.split(" ").str[0]
            )

        return all_results_df

    except Exception as e:
        raise ValueError(f"Could not load {results_file}: {e}")


def load_all_optimal_results(
    output_dir: str | Path,
    algorithms: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load results from all available optimal parameter experiments.

    Args:
        output_dir: Base output directory containing results
        algorithms: List of algorithm names to include. If None, discovers all available.

    Returns:
        Dictionary mapping algorithm display names to their results DataFrames
    """
    output_dir = Path(output_dir)

    if algorithms is None:
        # Discover available algorithms
        algorithms = []
        for algo_dir in output_dir.iterdir():
            if algo_dir.is_dir() and algo_dir.name in ALGORITHM_MAPPING:
                results_file = algo_dir / "results.json"
                if results_file.exists():
                    algorithms.append(algo_dir.name)

    all_results = {}

    for algorithm_name in algorithms:
        try:
            display_name = ALGORITHM_MAPPING[algorithm_name]
            print(f"Loading {display_name} results...")

            df = load_optimal_algo_results(
                base_output_dir=output_dir,
                algorithm_name=algorithm_name,
            )

            all_results[display_name] = df
            print(f"  Loaded {len(df)} result rows")

        except Exception as e:
            print(f"Warning: Could not load {algorithm_name}: {e}")
            continue

    return all_results


# Short names for annotations
SHORT_NAMES = {
    "Curator": "Curator",
    "Curator (Indexed)": "Curator-I",
    "Shared HNSW": "S-HNSW",
    "Per-Pred HNSW": "P-HNSW",
    "Shared IVF": "S-IVF",
    "Per-Pred IVF": "P-IVF",
    "Parlay IVF": "Parlay",
    "ACORN": r"ACORN-$\gamma$",
}


def compute_template_selectivities(
    output_dir: str | Path,
) -> Dict[str, float]:
    """Compute average selectivity for each template by loading the dataset.

    Args:
        output_dir: Output directory containing experiment configuration

    Returns:
        Dictionary mapping template names to their average selectivities
    """
    output_dir = Path(output_dir)

    # Load experiment configuration
    config_file = output_dir / "experiment_config.json"
    if not config_file.exists():
        raise FileNotFoundError(
            f"Could not find experiment_config.json in {output_dir}"
        )

    try:
        with open(config_file, "r") as f:
            config = json.load(f)

        # Extract common parameters for dataset construction
        common_params = config.get("common_parameters", {})
        if not common_params:
            raise RuntimeError(
                f"No common_parameters found in experiment config: {config_file}"
            )

        print(f"Loading dataset with parameters: {common_params}")

        # Create dataset using the common parameters
        dataset = ComplexPredicateDataset.from_dataset_key(**common_params)

        # Compute average selectivity for each template
        template_selectivities = {}
        for template, filters in dataset.template_to_filters.items():
            selectivities = [dataset.filter_to_selectivity[f] for f in filters]
            avg_selectivity = sum(selectivities) / len(selectivities)
            # Clean template name (remove parameters like {0} {1})
            clean_template = template.split(" ")[0]
            template_selectivities[clean_template] = avg_selectivity

        return template_selectivities

    except Exception as e:
        print(f"Warning: Could not load dataset selectivities: {e}")
        raise e


def plot_optimal_results(
    output_dir: str | Path = "output/complex_predicate_optimal",
    templates: List[str] = ["OR", "AND"],
    algorithms: Optional[List[str]] = None,
    output_path: str = "output/complex_predicate_optimal/figs/recall_vs_latency.pdf",
    subplot_size: tuple = (3, 2.5),
    font_size: int = 14,
):
    """Plot recall vs latency for all algorithms using optimal construction parameters.

    Args:
        output_dir: Base output directory containing results
        templates: List of query templates to plot
        algorithms: List of algorithm names to include. If None, includes all available.
        output_path: Path to save the output plot
        figsize: Figure size per template subplot
        font_size: Font size for the plot
    """
    print(f"=== Plotting Optimal Parameter Results ===")
    print(f"Output directory: {output_dir}")
    print(f"Templates: {templates}")
    print(f"Output path: {output_path}")
    print()

    # Load selectivity information
    template_selectivities = compute_template_selectivities(output_dir)
    print(f"Template selectivities: {template_selectivities}")

    # Load all baseline results
    all_results = load_all_optimal_results(
        output_dir=output_dir,
        algorithms=algorithms,
    )

    if not all_results:
        raise ValueError("No baseline results found")

    # Set up plotting
    plt.rcParams.update({"font.size": font_size})
    fig_width = subplot_size[0] * len(templates)
    fig_height = subplot_size[1]
    fig, axes = plt.subplots(1, len(templates), figsize=(fig_width, fig_height))

    # Handle single template case
    if len(templates) == 1:
        axes = [axes]

    # Filter to only available algorithms in the desired order
    available_algorithms = [alg for alg in ALGORITHM_ORDER if alg in all_results]

    # Custom color palette
    colors = sns.color_palette("tab10", n_colors=5)
    custom_palette = {
        "Curator": colors[0],
        "Curator (Indexed)": colors[0],
        "Shared HNSW": colors[1],
        "Per-Pred HNSW": colors[1],
        "Shared IVF": colors[2],
        "Per-Pred IVF": colors[2],
        "Parlay IVF": colors[3],
        "ACORN": colors[4],
    }

    # Plot each template
    for i, (template, ax) in enumerate(zip(templates, axes)):
        template_results = []

        for algorithm_name in available_algorithms:
            if algorithm_name not in all_results:
                continue

            df = all_results[algorithm_name]
            template_df = df[df["template"] == template].copy()

            if len(template_df) == 0:
                print(
                    f"Warning: No results found for {algorithm_name} with template {template}"
                )
                continue

            # Since we're using optimal construction parameters,
            # all results should be useful (no Pareto front selection needed)
            template_df["algorithm"] = algorithm_name
            template_results.append(template_df)

        if not template_results:
            ax.set_title(f"{template} - No Data")
            continue

        # Combine results
        combined_df = pd.concat(template_results, ignore_index=True)
        combined_df["query_lat_avg"] *= 1000  # Convert to ms

        # Create line plot
        sns.lineplot(
            data=combined_df,
            x="query_lat_avg",
            y="recall_at_k",
            hue="algorithm",
            hue_order=available_algorithms,
            style="algorithm",
            style_order=available_algorithms,
            ax=ax,
            markers=True,
            dashes=False,
            palette=custom_palette,
        )

        # Styling
        if i == 0:
            ax.set_ylabel("Recall@10")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Query Latency (ms)")
        ax.set_xscale("log")

        # Add selectivity to title
        selectivity = template_selectivities.get(template, None)
        if selectivity is not None:
            title = f"{template} (Sel = {selectivity:.1e})"
        else:
            title = template
        ax.set_title(title, fontsize=font_size - 2)

        ax.minorticks_off()
        ax.grid(visible=True, which="major", axis="both", linestyle="-", alpha=0.6)

    # Create shared legend with short names
    if available_algorithms and axes[0].get_legend():
        # Map algorithm names to short names
        short_names = [SHORT_NAMES.get(alg, alg) for alg in available_algorithms]

        # Calculate ncol to make legend width roughly match plot width
        # Aim for legend width similar to plot width
        max_ncol = min(len(available_algorithms), 4)  # Cap at 4 columns max
        ncol = min(max_ncol, (len(available_algorithms) + 1) // 2)

        legend = fig.legend(
            axes[0].get_legend().legend_handles,
            short_names,  # Use short names instead of full names
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),  # Slightly lower to accommodate smaller text
            ncol=ncol,
            fontsize=font_size - 2,  # Make legend text smaller
            columnspacing=1.0,  # Adjust spacing between columns
            handletextpad=0.5,  # Space between legend marker and text
        )

        # Remove individual legends
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()

        fig.tight_layout()

        # Save plot
        print(f"Saving plot to {output_path} ...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")
    else:
        fig.tight_layout()
        print(f"Saving plot to {output_path} ...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)

    print("Plot saved successfully!")


def print_optimal_summary(
    output_dir: str | Path = "output/complex_predicate_optimal",
):
    """Print summary statistics for optimal parameter experiments.

    Args:
        output_dir: Base output directory containing results
    """
    print(f"=== Optimal Parameter Results Summary ===")
    print(f"Output directory: {output_dir}")
    print()

    # Load all results
    all_results = load_all_optimal_results(
        output_dir=output_dir,
    )

    if not all_results:
        print("No results found!")
        return

    templates = ["AND", "OR"]

    for template in templates:
        print(f"--- Template: {template} ---")

        for algorithm_name in ALGORITHM_ORDER:
            if algorithm_name not in all_results:
                continue

            df = all_results[algorithm_name]
            template_df = df[df["template"] == template]

            if len(template_df) == 0:
                continue

            # Find best result (highest recall, if tied then lowest latency)
            recall_series = template_df["recall_at_k"]
            max_recall_idx = recall_series.idxmax()
            best_idx = template_df.loc[max_recall_idx]

            # If there are multiple with same max recall, pick the one with min latency
            max_recall_value = recall_series.max()
            max_recall_df = template_df[template_df["recall_at_k"] == max_recall_value]

            if len(max_recall_df) > 1:
                latency_series = max_recall_df["query_lat_avg"]
                min_latency_idx = latency_series.idxmin()
                best_idx = max_recall_df.loc[min_latency_idx]

            print(f"  {algorithm_name}:")
            print(
                f"    Best: Recall@10={best_idx['recall_at_k']:.3f}, Latency={float(best_idx['query_lat_avg'])*1000:.2f}ms"
            )
            print(
                f"    Range: Recall@10=[{float(recall_series.min()):.3f}, {float(recall_series.max()):.3f}]"
            )
            latency_series = template_df["query_lat_avg"]
            print(
                f"           Latency=[{float(latency_series.min())*1000:.2f}, {float(latency_series.max())*1000:.2f}]ms"
            )
            print(
                f"    Configurations: {len(template_df)} search parameter combinations"
            )

        print()


def plot_optimal_results_clean(
    output_dir: str | Path = "output/complex_predicate_optimal",
    templates: List[str] = ["OR", "AND"],
    algorithms: Optional[List[str]] = None,
    output_path: str = "output/complex_predicate_optimal/figs/recall_vs_latency_clean.pdf",
    subplot_size: tuple = (2.5, 2.5),  # Make subplots more compact
    font_size: int = 14,
):
    """Plot recall vs latency with clean visualization handling clustering.

    Creates 3 subplots: OR, AND Fast, AND Slow with independent y-axes.

    Args:
        output_dir: Base output directory containing results
        templates: List of query templates to plot
        algorithms: List of algorithm names to include. If None, includes all available.
        output_path: Path to save the output plot
        subplot_size: Figure size per template subplot
        font_size: Font size for the plot
    """
    print(f"=== Plotting Clean Optimal Parameter Results ===")
    print(f"Output directory: {output_dir}")
    print(f"Templates: {templates}")
    print(f"Output path: {output_path}")
    print()

    # Load selectivity information
    template_selectivities = compute_template_selectivities(output_dir)
    print(f"Template selectivities: {template_selectivities}")

    # Load all baseline results
    all_results = load_all_optimal_results(
        output_dir=output_dir,
        algorithms=algorithms,
    )

    if not all_results:
        raise ValueError("No baseline results found")

    # Filter to only available algorithms in the desired order
    available_algorithms = [alg for alg in ALGORITHM_ORDER if alg in all_results]

    # Custom color palette
    colors = sns.color_palette("tab10", n_colors=5)
    custom_palette = {
        "Curator": colors[0],
        "Curator (Indexed)": colors[0],
        "Shared HNSW": colors[1],
        "Per-Pred HNSW": colors[1],
        "Shared IVF": colors[2],
        "Per-Pred IVF": colors[2],
        "Parlay IVF": colors[3],
        "ACORN": colors[4],
    }

    # Define markers for each algorithm to ensure consistency
    marker_map = {
        "Curator": "^",
        "Curator (Indexed)": "v",
        "Shared HNSW": "h",
        "Per-Pred HNSW": "o",
        "Shared IVF": "s",
        "Per-Pred IVF": "X",
        "Parlay IVF": "d",
        "ACORN": "p",
    }

    # Define slow algorithms for AND template
    slow_algorithms = {"Shared HNSW", "Shared IVF", "ACORN", "Parlay IVF"}

    # Set up plotting - 3 subplots with independent y-axes
    plt.rcParams.update({"font.size": font_size})
    fig_width = subplot_size[0] * 3
    fig_height = subplot_size[1]
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    subplot_titles = []

    # Subplot 1: OR
    template = "OR"
    ax = axes[0]
    template_results = []

    for algorithm_name in available_algorithms:
        if algorithm_name not in all_results:
            continue

        df = all_results[algorithm_name]
        template_df = df[df["template"] == template].copy()

        if len(template_df) == 0:
            continue

        template_df["algorithm"] = algorithm_name
        template_results.append(template_df)

    if template_results:
        combined_df = pd.concat(template_results, ignore_index=True)
        combined_df["query_lat_avg"] *= 1000  # Convert to ms

        # Use sns.lineplot with custom markers
        sns.lineplot(
            data=combined_df,
            x="query_lat_avg",
            y="recall_at_k",
            hue="algorithm",
            hue_order=available_algorithms,
            style="algorithm",
            style_order=available_algorithms,
            ax=ax,
            markers=marker_map,  # Pass custom marker mapping
            dashes=False,
            palette=custom_palette,
            legend=False,
        )

        ax.set_xlabel("Query Latency (ms)")
        ax.set_xscale("log")

        ax.set_xticks([1, 100])
        ax.set_xticklabels(["10⁰", "10²"])

        # Add selectivity to title
        selectivity = template_selectivities.get(template, None)
        if selectivity is not None:
            title = f"{template} (Sel = {selectivity:.1e})"
        else:
            title = template
        subplot_titles.append(title)
    else:
        subplot_titles.append(f"{template} - No Data")

    # Subplot 2 & 3: AND Fast and AND Slow
    template = "AND"
    template_results = []

    for algorithm_name in available_algorithms:
        if algorithm_name not in all_results:
            continue

        df = all_results[algorithm_name]
        template_df = df[df["template"] == template].copy()

        if len(template_df) == 0:
            continue

        template_df["algorithm"] = algorithm_name
        template_results.append(template_df)

    if template_results:
        combined_df = pd.concat(template_results, ignore_index=True)
        combined_df["query_lat_avg"] *= 1000  # Convert to ms

        # Separate fast and slow algorithms
        fast_data = combined_df[~combined_df["algorithm"].isin(list(slow_algorithms))]
        slow_data = combined_df[combined_df["algorithm"].isin(list(slow_algorithms))]

        print(f"Fast algorithms: {fast_data['algorithm'].unique().tolist()}")
        print(f"Slow algorithms: {slow_data['algorithm'].unique().tolist()}")

        # Get selectivity for title
        selectivity = template_selectivities.get(template, None)
        if selectivity is not None:
            base_title = f"{template} (Sel = {selectivity:.1e})"
        else:
            base_title = template

        # Subplot 2: AND Fast
        ax_fast = axes[1]
        if len(fast_data) > 0:
            # Filter out results with recall < 0.7 for clearer plotting
            fast_data_filtered = fast_data[fast_data["recall_at_k"] >= 0.7]
            print(
                f"AND Fast: Filtered {len(fast_data) - len(fast_data_filtered)} results with recall < 0.7"
            )

            fast_algorithms = [
                alg for alg in available_algorithms if alg not in slow_algorithms
            ]

            # Create marker mapping for fast algorithms only
            fast_marker_map = {
                alg: marker_map[alg] for alg in fast_algorithms if alg in marker_map
            }

            sns.lineplot(
                data=fast_data_filtered,
                x="query_lat_avg",
                y="recall_at_k",
                hue="algorithm",
                hue_order=fast_algorithms,
                style="algorithm",
                style_order=fast_algorithms,
                ax=ax_fast,
                markers=fast_marker_map,  # Pass custom marker mapping for fast algorithms
                dashes=False,
                palette=custom_palette,
                legend=False,
            )

            ax_fast.set_xlabel("Query Latency (ms)")
            ax_fast.set_xscale("log")

            ax_fast.set_xticks([0.01, 0.1])
            ax_fast.set_xticklabels(["10⁻²", "10⁻¹"])
            ax_fast.tick_params(axis="x", which="minor", labelbottom=False)

            subplot_titles.append(f"{base_title} - Fast")
        else:
            subplot_titles.append(f"{base_title} - Fast (No Data)")

        # Subplot 3: AND Slow
        ax_slow = axes[2]
        if len(slow_data) > 0:
            slow_algorithms_list = [
                alg for alg in available_algorithms if alg in slow_algorithms
            ]

            # Create marker mapping for slow algorithms only
            slow_marker_map = {
                alg: marker_map[alg]
                for alg in slow_algorithms_list
                if alg in marker_map
            }

            sns.lineplot(
                data=slow_data,
                x="query_lat_avg",
                y="recall_at_k",
                hue="algorithm",
                hue_order=slow_algorithms_list,
                style="algorithm",
                style_order=slow_algorithms_list,
                ax=ax_slow,
                markers=slow_marker_map,  # Pass custom marker mapping for slow algorithms
                dashes=False,
                palette=custom_palette,
                legend=False,
            )

            ax_slow.set_xlabel("Query Latency (ms)")
            ax_slow.set_xscale("log")

            ax_slow.set_xticks([0.1, 10, 1000])
            ax_slow.set_xticklabels(["10⁻¹", "10¹", "10³"])

            subplot_titles.append(f"{base_title} - Slow")
        else:
            subplot_titles.append(f"{base_title} - Slow (No Data)")
    else:
        subplot_titles.append(f"{template} - Fast (No Data)")
        subplot_titles.append(f"{template} - Slow (No Data)")

    # Set titles and styling
    for i, (ax, title) in enumerate(zip(axes, subplot_titles)):
        ax.set_title(title, fontsize=font_size - 2)
        ax.set_ylabel("Recall@10" if i == 0 else None)
        # Turn off minor ticks only for y-axis, keep x-axis minor ticks
        ax.tick_params(axis="y", which="minor", left=False)
        ax.grid(visible=True, which="major", axis="both", linestyle="-", alpha=0.6)

    # Create shared legend with short names in two rows
    if available_algorithms:
        # Manually create legend from available algorithms with correct markers
        legend_handles = []
        legend_labels = []

        for alg in available_algorithms:
            if alg in custom_palette:
                from matplotlib.lines import Line2D

                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=custom_palette[alg],
                        marker=marker_map.get(alg, "o"),
                        linestyle="-",
                        markersize=4,
                    )
                )
                # legend_labels.append(SHORT_NAMES.get(alg, alg))
                legend_labels.append(alg)

        if legend_handles:
            # Two-row legend, moved higher to avoid overlap
            ncol = min(len(legend_handles), 4)  # Max 4 columns
            legend = fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.20),  # Moved higher from 1.15 to 1.20
                ncol=ncol,
                fontsize=font_size - 2,
                columnspacing=1.0,
                handletextpad=0.5,
            )

    fig.tight_layout()

    # Save plot
    print(f"Saving clean plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if "legend" in locals():
        plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")
    else:
        plt.savefig(output_path, bbox_inches="tight")

    print("Clean plot saved successfully!")


"""
Plotting profiling results produced by experiments/curator_prof.py.

    python -m benchmark.complex_predicate.plotting plot_profiling_latency_breakdown \
        --profiling_results_path test_curator_prof.json \
        --output_path test_curator_prof_breakdown.pdf

    python -m benchmark.complex_predicate.plotting print_profiling_summary \
        --profiling_results_path test_curator_prof.json
    
    python -m benchmark.complex_predicate.plotting plot_profiling_pie_charts \
        --profiling_results_path test_curator_prof.json \
        --output_path test_curator_prof_pie.pdf
"""


def plot_profiling_latency_breakdown(
    profiling_results_path: str,
    output_path: str = "profiling_latency_breakdown.pdf",
    templates: List[str] = ["AND", "OR"],
    font_size: int = 14,
    figsize: tuple = (12, 6),
):
    """Plot stacked bar chart showing search latency breakdown from profiling results.

    The search_with_bitmap_filter function has 4 phases:
    1. Preprocessing: Convert external labels to internal vector IDs
    2. Sorting: Sort qualified vectors' IDs in ascending order
    3. Build temporary index: Build temporary index structure
    4. Search: Search the temporary index

    Args:
        profiling_results_path: Path to JSON file with profiling results
        output_path: Path to save the output plot
        templates: List of query templates to analyze
        font_size: Font size for the plot
        figsize: Figure size
    """
    print(f"=== Plotting Profiling Latency Breakdown ===")
    print(f"Profiling results: {profiling_results_path}")
    print(f"Templates: {templates}")
    print(f"Output path: {output_path}")
    print()

    # Load profiling results
    with open(profiling_results_path, "r") as f:
        data = json.load(f)

    # Handle both single result and list of results
    if isinstance(data, list):
        results = data
    else:
        results = [data]

    # Extract profiling data for each template
    template_data = {}

    for result in results:
        per_template_results = result.get("per_template_results", {})

        for template_key, template_result in per_template_results.items():
            # Clean template name (remove parameters like {0} {1})
            clean_template = template_key.split(" ")[0]

            if clean_template not in templates:
                continue

            if clean_template not in template_data:
                template_data[clean_template] = {
                    "preproc_times": [],
                    "sort_times": [],
                    "build_temp_index_times": [],
                    "search_times": [],
                    "total_times": [],
                    "config_labels": [],
                }

            # Extract profiling times (convert to ms if needed)
            preproc_times = template_result.get("preproc_times_ms", [])
            sort_times = template_result.get("sort_times_ms", [])
            build_times = template_result.get("build_temp_index_times_ms", [])
            search_times = template_result.get("search_times_ms", [])

            if preproc_times and sort_times and build_times and search_times:
                # Calculate average times for this configuration
                avg_preproc = np.mean(preproc_times)
                avg_sort = np.mean(sort_times)
                avg_build = np.mean(build_times)
                avg_search = np.mean(search_times)
                total_time = avg_preproc + avg_sort + avg_build + avg_search

                template_data[clean_template]["preproc_times"].append(avg_preproc)
                template_data[clean_template]["sort_times"].append(avg_sort)
                template_data[clean_template]["build_temp_index_times"].append(
                    avg_build
                )
                template_data[clean_template]["search_times"].append(avg_search)
                template_data[clean_template]["total_times"].append(total_time)

                # Store search_ef for x-axis labeling
                search_ef = result.get("search_ef", "?")
                template_data[clean_template]["config_labels"].append(search_ef)

    if not template_data:
        print("No profiling data found for specified templates!")
        return

    # Set up plotting
    plt.rcParams.update({"font.size": font_size})
    n_templates = len(template_data)
    fig, axes = plt.subplots(1, n_templates, figsize=figsize)

    if n_templates == 1:
        axes = [axes]

    # Colors for each phase
    colors = {
        "preproc": "#1f77b4",  # blue
        "sort": "#ff7f0e",  # orange
        "build": "#2ca02c",  # green
        "search": "#d62728",  # red
    }

    for i, (template, data) in enumerate(template_data.items()):
        ax = axes[i]

        n_configs = len(data["preproc_times"])
        x_pos = np.arange(n_configs)

        # Create stacked bar chart
        bottom = np.zeros(n_configs)

        # Preprocessing phase
        bars1 = ax.bar(
            x_pos,
            data["preproc_times"],
            color=colors["preproc"],
            label="Preprocessing",
            bottom=bottom,
        )
        bottom += data["preproc_times"]

        # Sorting phase
        bars2 = ax.bar(
            x_pos,
            data["sort_times"],
            color=colors["sort"],
            label="Sorting",
            bottom=bottom,
        )
        bottom += data["sort_times"]

        # Build temp index phase
        bars3 = ax.bar(
            x_pos,
            data["build_temp_index_times"],
            color=colors["build"],
            label="Build Temp Index",
            bottom=bottom,
        )
        bottom += data["build_temp_index_times"]

        # Search phase
        bars4 = ax.bar(
            x_pos,
            data["search_times"],
            color=colors["search"],
            label="Search",
            bottom=bottom,
        )

        # Formatting
        ax.set_xlabel("search_ef")
        if i == 0:
            ax.set_ylabel("Latency (ms)")
        ax.set_title(f"{template} Template")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(data["config_labels"], rotation=45)
        ax.grid(True, alpha=0.3)

        # Add percentage labels on bars
        for j in range(n_configs):
            total = data["total_times"][j]

            # Add percentage text for each phase
            phases = [
                (data["preproc_times"][j], "Prep"),
                (data["sort_times"][j], "Sort"),
                (data["build_temp_index_times"][j], "Build"),
                (data["search_times"][j], "Search"),
            ]

            y_offset = 0
            for phase_time, phase_label in phases:
                if phase_time > total * 0.05:  # Only show if >5% of total
                    pct = 100 * phase_time / total
                    ax.text(
                        j,
                        y_offset + phase_time / 2,
                        f"{pct:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=font_size - 4,
                        color="white",
                        weight="bold",
                    )
                y_offset += phase_time

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)

    plt.tight_layout()

    # Save plot
    print(f"Saving profiling plot to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print("Profiling plot saved successfully!")


def print_profiling_summary(
    profiling_results_path: str,
    templates: List[str] = ["AND", "OR"],
):
    """Print summary statistics of profiling results showing latency breakdown percentages.

    Args:
        profiling_results_path: Path to JSON file with profiling results
        templates: List of query templates to analyze
    """
    print(f"=== Profiling Summary ===")
    print(f"Profiling results: {profiling_results_path}")
    print(f"Templates: {templates}")
    print()

    # Load profiling results
    with open(profiling_results_path, "r") as f:
        data = json.load(f)

    # Handle both single result and list of results
    if isinstance(data, list):
        results = data
    else:
        results = [data]

    # Process each template
    for template in templates:
        print(f"--- Template: {template} ---")

        all_preproc_times = []
        all_sort_times = []
        all_build_times = []
        all_search_times = []
        all_total_times = []
        all_qualified_counts = []
        all_temp_nodes_counts = []

        config_count = 0

        for result in results:
            per_template_results = result.get("per_template_results", {})

            for template_key, template_result in per_template_results.items():
                # Clean template name (remove parameters like {0} {1})
                clean_template = template_key.split(" ")[0]

                if clean_template != template:
                    continue

                config_count += 1

                # Extract profiling times
                preproc_times = template_result.get("preproc_times_ms", [])
                sort_times = template_result.get("sort_times_ms", [])
                build_times = template_result.get("build_temp_index_times_ms", [])
                search_times = template_result.get("search_times_ms", [])
                qualified_counts = template_result.get("qualified_labels_counts", [])
                temp_nodes_counts = template_result.get("temp_nodes_counts", [])

                if preproc_times and sort_times and build_times and search_times:
                    all_preproc_times.extend(preproc_times)
                    all_sort_times.extend(sort_times)
                    all_build_times.extend(build_times)
                    all_search_times.extend(search_times)

                    # Calculate total times for each query
                    total_times = [
                        p + s + b + search
                        for p, s, b, search in zip(
                            preproc_times, sort_times, build_times, search_times
                        )
                    ]
                    all_total_times.extend(total_times)

                    all_qualified_counts.extend(qualified_counts)
                    all_temp_nodes_counts.extend(temp_nodes_counts)

        if not all_total_times:
            print(f"  No profiling data found for {template}")
            continue

        print(f"  Configurations analyzed: {config_count}")
        print(f"  Total queries: {len(all_total_times)}")
        print()

        # Calculate statistics for each phase
        total_avg = np.mean(all_total_times)
        total_std = np.std(all_total_times)

        preproc_avg = np.mean(all_preproc_times)
        preproc_std = np.std(all_preproc_times)
        preproc_pct_avg = 100 * preproc_avg / total_avg
        preproc_pct_std = 100 * preproc_std / total_avg

        sort_avg = np.mean(all_sort_times)
        sort_std = np.std(all_sort_times)
        sort_pct_avg = 100 * sort_avg / total_avg
        sort_pct_std = 100 * sort_std / total_avg

        build_avg = np.mean(all_build_times)
        build_std = np.std(all_build_times)
        build_pct_avg = 100 * build_avg / total_avg
        build_pct_std = 100 * build_std / total_avg

        search_avg = np.mean(all_search_times)
        search_std = np.std(all_search_times)
        search_pct_avg = 100 * search_avg / total_avg
        search_pct_std = 100 * search_std / total_avg

        print(f"  Overall latency: {total_avg:.3f} ± {total_std:.3f} ms")
        print()
        print(f"  Phase breakdown (avg ± std):")
        print(
            f"    Preprocessing:     {preproc_avg:.3f} ± {preproc_std:.3f} ms ({preproc_pct_avg:.1f} ± {preproc_pct_std:.1f}%)"
        )
        print(
            f"    Sorting:           {sort_avg:.3f} ± {sort_std:.3f} ms ({sort_pct_avg:.1f} ± {sort_pct_std:.1f}%)"
        )
        print(
            f"    Build Temp Index:  {build_avg:.3f} ± {build_std:.3f} ms ({build_pct_avg:.1f} ± {build_pct_std:.1f}%)"
        )
        print(
            f"    Search:            {search_avg:.3f} ± {search_std:.3f} ms ({search_pct_avg:.1f} ± {search_pct_std:.1f}%)"
        )
        print()

        # Additional statistics
        qualified_avg = np.mean(all_qualified_counts)
        qualified_std = np.std(all_qualified_counts)
        nodes_avg = np.mean(all_temp_nodes_counts)
        nodes_std = np.std(all_temp_nodes_counts)

        print(f"  Additional metrics:")
        print(f"    Qualified labels:  {qualified_avg:.1f} ± {qualified_std:.1f}")
        print(f"    Temp nodes:        {nodes_avg:.1f} ± {nodes_std:.1f}")
        print()


def plot_profiling_pie_charts(
    profiling_results_path: str,
    output_path: str = "profiling_pie_charts.pdf",
    templates: List[str] = ["AND", "OR"],
    font_size: int = 14,
    figsize: tuple = (10, 5),
):
    """Plot pie charts showing average latency breakdown for each template.

    Args:
        profiling_results_path: Path to JSON file with profiling results
        output_path: Path to save the output plot
        templates: List of query templates to analyze
        font_size: Font size for the plot
        figsize: Figure size
    """
    print(f"=== Plotting Profiling Pie Charts ===")
    print(f"Profiling results: {profiling_results_path}")
    print(f"Templates: {templates}")
    print(f"Output path: {output_path}")
    print()

    # Load profiling results
    with open(profiling_results_path, "r") as f:
        data = json.load(f)

    # Handle both single result and list of results
    if isinstance(data, list):
        results = data
    else:
        results = [data]

    # Calculate average times for each template
    template_averages = {}

    for template in templates:
        all_preproc_times = []
        all_sort_times = []
        all_build_times = []
        all_search_times = []

        for result in results:
            per_template_results = result.get("per_template_results", {})

            for template_key, template_result in per_template_results.items():
                # Clean template name (remove parameters like {0} {1})
                clean_template = template_key.split(" ")[0]

                if clean_template != template:
                    continue

                # Extract profiling times
                preproc_times = template_result.get("preproc_times_ms", [])
                sort_times = template_result.get("sort_times_ms", [])
                build_times = template_result.get("build_temp_index_times_ms", [])
                search_times = template_result.get("search_times_ms", [])

                if preproc_times and sort_times and build_times and search_times:
                    all_preproc_times.extend(preproc_times)
                    all_sort_times.extend(sort_times)
                    all_build_times.extend(build_times)
                    all_search_times.extend(search_times)

        if all_preproc_times:
            template_averages[template] = {
                "preproc": np.mean(all_preproc_times),
                "sort": np.mean(all_sort_times),
                "build": np.mean(all_build_times),
                "search": np.mean(all_search_times),
            }

    if not template_averages:
        print("No profiling data found for specified templates!")
        return

    # Set up plotting
    plt.rcParams.update({"font.size": font_size})
    n_templates = len(template_averages)
    fig, axes = plt.subplots(1, n_templates, figsize=figsize)

    if n_templates == 1:
        axes = [axes]

    # Colors for each phase (same as stacked bar chart)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    labels = ["Preprocessing", "Sorting", "Build Temp Index", "Search"]

    for i, (template, averages) in enumerate(template_averages.items()):
        ax = axes[i]

        # Calculate values and percentages
        values = [
            averages["preproc"],
            averages["sort"],
            averages["build"],
            averages["search"],
        ]
        total = sum(values)
        percentages = [100 * v / total for v in values]

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": font_size - 2},
        )

        # Make percentage text bold and white
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_weight("bold")

        ax.set_title(
            f"{template} Template\n(Total: {total:.2f} ms)", fontsize=font_size
        )

    plt.tight_layout()

    # Save plot
    print(f"Saving profiling pie charts to {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print("Profiling pie charts saved successfully!")


"""
Plotting functions for temp index evaluation results produced by experiments/eval_temp_index.py

python -m benchmark.complex_predicate.plotting \
    print_eval_temp_index_summary \
        --results_path eval_temp_index.json \
        --templates '["AND {0} {1}", "OR {0} {1}"]'

python -m benchmark.complex_predicate.plotting \
    plot_eval_temp_index \
        --results_path eval_temp_index.json \
        --output_path eval_temp_index.pdf \
        --templates '["OR", "AND"]'
"""


def print_eval_temp_index_summary(
    results_path: str,
    templates: List[str] = ["AND", "OR"],
):
    """Print summary statistics for temp index evaluation results comparing strategies.

    Args:
        results_path: Path to JSON file with temp index evaluation results
        templates: List of query templates to analyze
    """
    print(f"=== Temp Index Strategy Comparison Summary ===")
    print(f"Results file: {results_path}")
    print(f"Templates: {templates}")
    print()

    # Load results
    with open(results_path, "r") as f:
        data = json.load(f)

    # Handle both single result and list of results
    if isinstance(data, list):
        results = data
    else:
        results = [data]

    # Group results by index strategy
    strategy_results = {}

    for result in results:
        strategy = result.get("index_strategy", "unknown")
        if strategy not in strategy_results:
            strategy_results[strategy] = []
        strategy_results[strategy].append(result)

    # Process each template
    for template in templates:
        print(f"--- Template: {template} ---")

        for strategy, strategy_data in strategy_results.items():
            print(f"  Strategy: {strategy}")

            # Collect performance metrics
            recalls = []
            latencies = []
            search_efs = []

            # Collect indexing metrics (should be same across search_ef for same strategy)
            filter_index_times = []
            filter_index_size_kb = []
            temp_index_size_kb = []
            total_filters = None

            for result in strategy_data:
                per_template_results = result.get("per_template_results", {})

                for template_key, template_result in per_template_results.items():
                    # Clean template name (remove parameters like {0} {1})
                    clean_template = template_key.split(" ")[0]

                    if clean_template != template:
                        continue

                    # Query performance metrics
                    recalls.append(template_result.get("recall_at_k", 0))
                    latencies.append(
                        template_result.get("query_lat_avg", 0) * 1000
                    )  # Convert to ms
                    search_efs.append(result.get("search_ef", 0))

                # Indexing metrics (collect once per strategy)
                if result.get("filter_index_times"):
                    filter_index_times.extend(result["filter_index_times"])
                if result.get("filter_index_size_kb") is not None:
                    filter_index_size_kb.append(result["filter_index_size_kb"])

                # Handle both possible field names for temp index size
                if result.get("temp_index_size_kb") is not None:
                    temp_index_size_kb.append(result["temp_index_size_kb"])
                elif result.get("temp_index_size_bytes") is not None:
                    temp_index_size_kb.append(result["temp_index_size_bytes"] / 1024)

                if result.get("total_filters") is not None:
                    total_filters = result["total_filters"]

            if not recalls:
                print(f"    No data found for {template}")
                continue

            # Performance summary
            if recalls and latencies:
                print(f"    Query Performance:")
                print(
                    f"      Recall@10: [{min(recalls):.3f}, {max(recalls):.3f}] (best: {max(recalls):.3f})"
                )
                print(
                    f"      Latency:   [{min(latencies):.2f}, {max(latencies):.2f}] ms (best: {min(latencies):.2f} ms)"
                )
                print(f"      Search_ef: [{min(search_efs)}, {max(search_efs)}]")
                print(f"      Configurations: {len(recalls)}")

            # Indexing summary
            if filter_index_times:
                avg_filter_time = sum(filter_index_times) / len(filter_index_times)
                print(f"    Indexing Performance:")
                print(f"      Total filters: {total_filters}")
                print(f"      Avg filter index time: {avg_filter_time:.4f} seconds")
                if filter_index_size_kb:
                    avg_filter_size = sum(filter_index_size_kb) / len(
                        filter_index_size_kb
                    )
                    print(f"      Filter index memory: {avg_filter_size:.2f} KB")
                if temp_index_size_kb:
                    avg_temp_size = sum(temp_index_size_kb) / len(temp_index_size_kb)
                    print(
                        f"      Temp index memory: {avg_temp_size:.2f} KB ({avg_temp_size/1024:.2f} MB)"
                    )

            print()

        # Compare strategies for this template
        if (
            len(strategy_results) == 2
            and "direct_indexing" in strategy_results
            and "temp_caching" in strategy_results
        ):
            print(f"  Strategy Comparison:")

            # Get best performance for each strategy
            direct_results = []
            temp_results = []

            for result in strategy_results["direct_indexing"]:
                per_template_results = result.get("per_template_results", {})
                for template_key, template_result in per_template_results.items():
                    clean_template = template_key.split(" ")[0]
                    if clean_template == template:
                        direct_results.append(
                            (
                                template_result.get("recall_at_k", 0),
                                template_result.get("query_lat_avg", 0) * 1000,
                                result.get("search_ef", 0),
                            )
                        )

            for result in strategy_results["temp_caching"]:
                per_template_results = result.get("per_template_results", {})
                for template_key, template_result in per_template_results.items():
                    clean_template = template_key.split(" ")[0]
                    if clean_template == template:
                        temp_results.append(
                            (
                                template_result.get("recall_at_k", 0),
                                template_result.get("query_lat_avg", 0) * 1000,
                                result.get("search_ef", 0),
                            )
                        )

            if direct_results and temp_results:
                # Find best recall for each strategy
                direct_best = max(direct_results, key=lambda x: x[0])
                temp_best = max(temp_results, key=lambda x: x[0])

                # Find best latency for each strategy (among results with max recall)
                direct_max_recall = max(r[0] for r in direct_results)
                temp_max_recall = max(r[0] for r in temp_results)

                direct_best_latency = min(
                    r[1] for r in direct_results if r[0] == direct_max_recall
                )
                temp_best_latency = min(
                    r[1] for r in temp_results if r[0] == temp_max_recall
                )

                print(
                    f"    Best recall - Direct: {direct_best[0]:.3f}, Temp: {temp_best[0]:.3f}"
                )
                print(
                    f"    Best latency - Direct: {direct_best_latency:.2f} ms, Temp: {temp_best_latency:.2f} ms"
                )

                # Memory comparison
                direct_memory = strategy_results["direct_indexing"][0].get(
                    "filter_index_size_kb", 0
                )
                temp_memory = strategy_results["temp_caching"][0].get(
                    "temp_index_size_kb", 0
                )
                if temp_memory == 0:
                    temp_memory = (
                        strategy_results["temp_caching"][0].get(
                            "temp_index_size_bytes", 0
                        )
                        / 1024
                    )

                if direct_memory > 0 and temp_memory > 0:
                    memory_ratio = temp_memory / direct_memory
                    print(f"    Memory ratio (temp/direct): {memory_ratio:.2f}x")

                print()

        print()


def plot_eval_temp_index(
    results_path: str,
    output_path: str = "eval_temp_index.pdf",
    templates: List[str] = ["OR", "AND"],
    font_size: int = 14,
    subplot_size: tuple = (3, 2.5),
):
    """Plot recall vs latency curves comparing direct_indexing and temp_caching strategies.

    Args:
        results_path: Path to JSON file with temp index evaluation results
        output_path: Path to save the output plot
        templates: List of query templates to plot
        font_size: Font size for the plot
        subplot_size: Figure size per template subplot
    """
    print(f"=== Plotting Temp Index Strategy Comparison ===")
    print(f"Results file: {results_path}")
    print(f"Templates: {templates}")
    print(f"Output path: {output_path}")
    print()

    # Load results
    with open(results_path, "r") as f:
        data = json.load(f)

    # Handle both single result and list of results
    if isinstance(data, list):
        results = data
    else:
        results = [data]

    # Set up plotting
    plt.rcParams.update({"font.size": font_size})
    fig_width = subplot_size[0] * len(templates)
    fig_height = subplot_size[1]
    fig, axes = plt.subplots(1, len(templates), figsize=(fig_width, fig_height))

    # Handle single template case
    if len(templates) == 1:
        axes = [axes]

    # Colors and markers for strategies
    strategy_colors = {
        "direct_indexing": "#1f77b4",  # blue
        "temp_caching": "#ff7f0e",  # orange
    }

    strategy_markers = {
        "direct_indexing": "o",
        "temp_caching": "^",
    }

    strategy_labels = {
        "direct_indexing": "Direct Indexing",
        "temp_caching": "Temp Caching",
    }

    # Plot each template
    for i, (template, ax) in enumerate(zip(templates, axes)):
        template_results = []

        # Group results by strategy
        strategy_data = {}
        for result in results:
            strategy = result.get("index_strategy", "unknown")
            if strategy not in strategy_data:
                strategy_data[strategy] = []
            strategy_data[strategy].append(result)

        # Process each strategy
        for strategy, strategy_results in strategy_data.items():
            if strategy not in strategy_colors:
                continue

            strategy_points = []

            for result in strategy_results:
                per_template_results = result.get("per_template_results", {})

                for template_key, template_result in per_template_results.items():
                    # Clean template name (remove parameters like {0} {1})
                    clean_template = template_key.split(" ")[0]

                    if clean_template != template:
                        continue

                    recall = template_result.get("recall_at_k", 0)
                    latency = (
                        template_result.get("query_lat_avg", 0) * 1000
                    )  # Convert to ms
                    search_ef = result.get("search_ef", 0)

                    strategy_points.append(
                        {
                            "recall": recall,
                            "latency": latency,
                            "search_ef": search_ef,
                            "strategy": strategy_labels[strategy],
                        }
                    )

            if strategy_points:
                strategy_df = pd.DataFrame(strategy_points)
                strategy_df = strategy_df.sort_values("search_ef")

                ax.plot(
                    strategy_df["latency"],
                    strategy_df["recall"],
                    color=strategy_colors[strategy],
                    marker=strategy_markers[strategy],
                    markersize=6,
                    linewidth=2,
                    label=strategy_labels[strategy],
                    alpha=0.8,
                )

        if i == 0:
            ax.set_ylabel("Recall@10")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Query Latency (ms)")
        ax.set_xscale("log")
        ax.set_title(f"{template} Template", fontsize=font_size - 2)

        # Ensure x-axis ticks are visible
        ax.tick_params(axis="x", which="major", labelsize=font_size - 2)
        ax.tick_params(axis="y", which="major", labelsize=font_size - 2)

        ax.grid(visible=True, which="major", axis="both", linestyle="-", alpha=0.6)
        ax.minorticks_off()

    # Create shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=len(strategy_labels),
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
    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")

    print("Plot saved successfully!")


if __name__ == "__main__":
    fire.Fire()
