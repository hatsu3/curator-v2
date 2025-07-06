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
            max_recall_idx = template_df["recall_at_k"].idxmax()
            best_idx = template_df.loc[max_recall_idx]

            # If there are multiple with same max recall, pick the one with min latency
            max_recall_value = template_df["recall_at_k"].max()
            max_recall_df = template_df[template_df["recall_at_k"] == max_recall_value]

            if len(max_recall_df) > 1:
                min_latency_idx = max_recall_df["query_lat_avg"].idxmin()
                best_idx = max_recall_df.loc[min_latency_idx]

            print(f"  {algorithm_name}:")
            print(
                f"    Best: Recall@10={best_idx['recall_at_k']:.3f}, Latency={float(best_idx['query_lat_avg'])*1000:.2f}ms"
            )
            print(
                f"    Range: Recall@10=[{float(template_df['recall_at_k'].min()):.3f}, {float(template_df['recall_at_k'].max()):.3f}]"
            )
            print(
                f"           Latency=[{float(template_df['query_lat_avg'].min())*1000:.2f}, {float(template_df['query_lat_avg'].max())*1000:.2f}]ms"
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
        fast_data = combined_df[~combined_df["algorithm"].isin(slow_algorithms)]
        slow_data = combined_df[combined_df["algorithm"].isin(slow_algorithms)]

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

            ax_slow.set_xticks([10, 1000])
            ax_slow.set_xticklabels(["10¹", "10³"])

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


if __name__ == "__main__":
    fire.Fire()
