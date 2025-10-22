"""
Plotting functionality for complex predicate experiments with optimal parameters.
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

# Mapping from output dir name to display name
ALGORITHM_MAPPING = {
    "curator": "Curator",
    "curator_with_index": "Curator (Idx)",
    "shared_hnsw": "Shared HNSW",
    "per_predicate_hnsw": "Per-Pred HNSW",
    "shared_ivf": "Shared IVF",
    "per_predicate_ivf": "Per-Pred IVF",
    "parlay_ivf": "Parlay IVF",
    "acorn": "ACORN",
    "pre_filtering": "Pre-Filter",
    "pgvector_hnsw": "Pg-HNSW",
    "pgvector_ivf": "Pg-IVF",
}

# Global algorithm order for consistent plotting
ALGORITHM_ORDER = [
    "Curator",
    "Curator (Idx)",
    "Shared HNSW",
    "Per-Pred HNSW",
    "Shared IVF",
    "Per-Pred IVF",
    "Pg-HNSW",
    "Pg-IVF",
    "Parlay IVF",
    "ACORN",
    "Pre-Filter",
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
    "Pg-HNSW": "Pg-HNSW",
    "Pg-IVF": "Pg-IVF",
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


def _load_prefilter_model(
    prefilter_model_dir: str | Path, dataset_key: str, test_size: float
):
    prefilter_model_dir = Path(prefilter_model_dir)
    exact_dir = prefilter_model_dir / f"{dataset_key}_test{test_size}"
    exact_path = exact_dir / "linreg.json"
    if exact_path.exists():
        data = json.load(open(exact_path))
        return float(data["a"]), float(data["b"])
    for p in prefilter_model_dir.rglob("linreg.json"):
        try:
            data = json.load(open(p))
            return float(data["a"]), float(data["b"])
        except Exception:
            continue
    return None


def _compute_prefilter_points(
    output_dir: str | Path,
    prefilter_model_dir: str | Path = "output/overall_results2/pre_filtering",
):
    output_dir = Path(output_dir)
    cfg_path = output_dir / "experiment_config.json"
    if not cfg_path.exists():
        return pd.DataFrame()
    cfg = json.load(open(cfg_path))
    params = cfg.get("common_parameters", {})
    dataset_key = params.get("dataset_key")
    test_size = float(params.get("test_size", 0.01))

    pf = _load_prefilter_model(prefilter_model_dir, dataset_key, test_size)
    if pf is None:
        return pd.DataFrame()
    a, b = pf

    ds = ComplexPredicateDataset.from_dataset_key(**params)
    n_train = len(ds.train_vecs)

    rows = []
    for template, filters in ds.template_to_filters.items():
        sels = [ds.filter_to_selectivity[f] for f in filters]
        if not sels:
            continue
        sel_avg = float(sum(sels) / len(sels))
        n_avg = sel_avg * n_train
        lat_ms = max(a * n_avg + max(b, 0.0), 0.0) * 1e3
        print(f"Template: {template}, Selectivity: {sel_avg}, Latency: {lat_ms} ms")
        rows.append(
            {
                "template": template.split()[0],
                "recall_at_k": 1.0,
                "query_lat_avg": lat_ms / 1e3,
                "algorithm": "Pre-Filter",
            }
        )

    return pd.DataFrame(rows)


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
    colors = sns.color_palette("tab10", n_colors=6)
    custom_palette = {
        "Curator": colors[0],
        "Curator (Idx)": colors[0],
        "Shared HNSW": colors[1],
        "Per-Pred HNSW": colors[1],
        "Shared IVF": colors[2],
        "Per-Pred IVF": colors[2],
        "Parlay IVF": colors[3],
        "ACORN": colors[4],
        "Pg-HNSW": colors[5],
        "Pg-IVF": colors[5]
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
    subplot_size: tuple = (2, 2),  # Make subplots more compact
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

    # Inject Pre-Filter single points computed from linear model
    pf_df = _compute_prefilter_points(output_dir)
    if not pf_df.empty:
        all_results[ALGORITHM_MAPPING["pre_filtering"]] = pf_df

    if not all_results:
        raise ValueError("No baseline results found")

    # Filter to only available algorithms in the desired order
    available_algorithms = [alg for alg in ALGORITHM_ORDER if alg in all_results]

    # Custom color palette
    colors = sns.color_palette("tab10", n_colors=6)
    custom_palette = {
        "Curator": colors[0],
        "Curator (Idx)": colors[0],
        "Shared HNSW": colors[1],
        "Per-Pred HNSW": colors[1],
        "Shared IVF": colors[2],
        "Per-Pred IVF": colors[2],
        "Parlay IVF": colors[3],
        "ACORN": colors[4],
        "Pg-HNSW": colors[5],
        "Pg-IVF": colors[5],
        "Pre-Filter": "tab:red",
    }

    # Define markers for each algorithm to ensure consistency
    marker_map = {
        "Curator": "^",
        "Curator (Idx)": "v",
        "Shared HNSW": "h",
        "Per-Pred HNSW": "o",
        "Shared IVF": "s",
        "Per-Pred IVF": "*",
        "Parlay IVF": "d",
        "ACORN": "p",
        "Pg-HNSW": "<",
        "Pg-IVF": ">",
        "Pre-Filter": "X",
    }

    # Define slow algorithms for AND template
    slow_algorithms = {"Shared HNSW", "Shared IVF", "ACORN", "Parlay IVF", "Pre-Filter", "Pg-HNSW", "Pg-IVF"}

    # Set up plotting - 3 subplots with custom spacing to group AND subplots
    plt.rcParams.update({"font.size": font_size})
    fig_width = subplot_size[0] * 3
    fig_height = subplot_size[1]

    # Use GridSpec to customize spacing between subplots
    # Group AND subplots together with reduced spacing
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3, width_ratios=[1, 1, 1])
    gs.update(left=0.08, right=0.98, top=0.88, bottom=0.15)

    # Create subplots with custom spacing: normal gap after OR, tight gap between AND subplots
    axes = [
        fig.add_subplot(gs[0, 0]),  # OR
        fig.add_subplot(gs[0, 1]),  # AND Fast
        fig.add_subplot(gs[0, 2]),  # AND Slow
    ]

    # Manually adjust positions to group AND subplots
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()

    # Keep OR position, move AND subplots closer together
    gap_normal = 0.16  # Normal gap between OR and AND-Fast
    gap_tight = 0.08  # Tight gap between AND-Fast and AND-Slow

    axes[1].set_position([pos0.x1 + gap_normal, pos1.y0, pos1.width, pos1.height])
    pos1_new = axes[1].get_position()
    axes[2].set_position([pos1_new.x1 + gap_tight, pos2.y0, pos2.width, pos2.height])

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

        # Convert latency to QPS (queries per second)
        combined_df["qps"] = 1000 / combined_df["query_lat_avg"]

        # Use sns.lineplot with custom markers
        # Create legend on first subplot to extract later
        sns.lineplot(
            data=combined_df,
            x="recall_at_k",
            y="qps",
            hue="algorithm",
            hue_order=available_algorithms,
            style="algorithm",
            style_order=available_algorithms,
            ax=ax,
            markers=marker_map,  # Pass custom marker mapping
            dashes=False,
            palette=custom_palette,
            legend=True,  # Create legend to extract for figure-level legend
        )

        ax.set_xlabel("Recall@10")
        ax.set_ylabel("QPS")
        ax.set_yscale("log")

        ax.set_yticks([10, 1000])
        ax.set_yticklabels(["10¹", "10³"])

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

        # Convert latency to QPS (queries per second)
        combined_df["qps"] = 1000 / combined_df["query_lat_avg"]

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
                x="recall_at_k",
                y="qps",
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

            ax_fast.set_xlabel("Recall@10")
            ax_fast.set_ylabel("QPS")
            ax_fast.set_yscale("log")

            # Set only two major ticks, enable minor ticks without labels
            ax_fast.set_yticks([10000, 100000])
            ax_fast.set_yticklabels(["10⁴", "10⁵"])
            ax_fast.minorticks_on()
            ax_fast.tick_params(axis="y", which="minor", left=True, labelleft=False)
            # Add minor grid lines for both axes
            ax_fast.grid(
                visible=True, which="minor", axis="both", linestyle=":", alpha=0.3
            )

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
                x="recall_at_k",
                y="qps",
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

            ax_slow.set_xlabel("Recall@10")
            ax_slow.set_ylabel("QPS")
            ax_slow.set_yscale("log")

            ax_slow.set_yticks([1, 100, 10000])
            ax_slow.set_yticklabels(["10⁰", "10²", "10⁴"])

            subplot_titles.append(f"{base_title} - Slow")
        else:
            subplot_titles.append(f"{base_title} - Slow (No Data)")
    else:
        subplot_titles.append(f"{template} - Fast (No Data)")
        subplot_titles.append(f"{template} - Slow (No Data)")

    # Set titles and styling
    for i, (ax, title) in enumerate(zip(axes, subplot_titles)):
        # Only set title for OR subplot (i=0)
        # AND subplots will use spanning title
        if i == 0:
            ax.set_title(title, fontsize=font_size - 1)
        ax.set_ylabel("QPS" if i == 0 else None)
        # Turn off minor ticks for y-axis, except for middle subplot (i=1) which has minor ticks enabled
        if i != 1:
            ax.tick_params(axis="y", which="minor", left=False)
        ax.grid(visible=True, which="major", axis="both", linestyle="-", alpha=0.6)

    # Create shared legend using Seaborn's default legend handles
    # This provides better marker styling (white borders, proper edge widths)
    # matching the style used in Figures 7-8
    if available_algorithms and axes[0].get_legend():
        legend_handles = axes[0].get_legend().legend_handles
        legend_labels = available_algorithms

        # Two-row legend, positioned to avoid overlap
        ncol = min(len(legend_labels), 5)  # Max 5 columns
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.55, 1.30),
            ncol=ncol,
            fontsize=font_size - 3,
            columnspacing=0.8,  # Reduced from 1.0
            handletextpad=0.5,
        )

        # Remove individual legends from subplots
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()

    # Add spanning title for AND subplots (middle and right)
    # Align with OR subplot title by using the same vertical position
    # Get final positions after manual adjustment
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    pos2 = axes[2].get_position()

    center_x = (pos1.x0 + pos2.x1) / 2
    # Use same y-position as OR title for horizontal alignment
    top_y = pos0.y1 + 0.02  # Match OR title position

    # Add shared AND title
    selectivity = template_selectivities.get("AND", None)
    if selectivity:
        shared_title = f"AND (Sel = {selectivity:.1e})"
    else:
        shared_title = "AND"

    fig.text(
        center_x,
        top_y,
        shared_title,
        ha="center",
        va="bottom",
        fontsize=font_size - 1,
        transform=fig.transFigure,
    )

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
