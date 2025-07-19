"""
Extracts optimal construction and search parameters for complex predicate algorithms.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm


def select_pareto_front(
    df: pd.DataFrame,
    x_key: str = "query_lat_avg",
    y_key: str = "recall_at_k",
    min_x: bool = True,  # minimize x
    min_y: bool = False,  # maximize y
) -> pd.DataFrame:
    """Select Pareto front from results DataFrame.

    Args:
        df: DataFrame with results
        x_key: Column name for x-axis metric
        y_key: Column name for y-axis metric
        min_x: Whether to minimize x-axis metric
        min_y: Whether to minimize y-axis metric

    Returns:
        DataFrame containing only Pareto-optimal points
    """

    def is_dominated(r1, r2):
        x_worse = r1[x_key] > r2[x_key] if min_x else r1[x_key] < r2[x_key]
        y_worse = r1[y_key] > r2[y_key] if min_y else r1[y_key] < r2[y_key]
        return x_worse and y_worse

    pareto_front = []
    for i, r1 in df.iterrows():
        if any(is_dominated(r1, r2) for _, r2 in df.iterrows()):
            continue
        pareto_front.append(r1)

    return pd.DataFrame(pareto_front) if pareto_front else pd.DataFrame()


def interpolate_latency_at_recall(
    df: pd.DataFrame,
    target_recall: float = 0.95,
    recall_col: str = "recall_at_k",
    latency_col: str = "query_lat_avg",
) -> Optional[float]:
    """Interpolate latency at a specific recall threshold using linear interpolation.

    Args:
        df: DataFrame with recall and latency data
        target_recall: Target recall threshold to interpolate at
        recall_col: Column name for recall values
        latency_col: Column name for latency values

    Returns:
        Interpolated latency at target recall, or None if interpolation not possible
    """
    if len(df) < 2:
        return None

    # Sort by recall to enable interpolation
    df_sorted = df.sort_values(recall_col)
    recalls = np.array(df_sorted[recall_col])
    latencies = np.array(df_sorted[latency_col])

    # Check if target recall is within the range
    if target_recall < recalls.min() or target_recall > recalls.max():
        return None

    # If exact match exists, return it
    exact_matches = df_sorted[df_sorted[recall_col] == target_recall]
    if len(exact_matches) > 0:
        return float(exact_matches[latency_col].values[0])

    # Linear interpolation
    try:
        interpolated_latency = np.interp(target_recall, recalls, latencies)
        return float(interpolated_latency)
    except Exception:
        return None


def interpolate_recall_at_latency(
    df: pd.DataFrame,
    target_latency: float,
    recall_col: str = "recall_at_k",
    latency_col: str = "query_lat_avg",
) -> Optional[float]:
    """Interpolate recall at a specific latency threshold using linear interpolation.

    Args:
        df: DataFrame with recall and latency data
        target_latency: Target latency threshold to interpolate at
        recall_col: Column name for recall values
        latency_col: Column name for latency values

    Returns:
        Interpolated recall at target latency, or None if interpolation not possible
    """
    if len(df) < 2:
        return None

    # Sort by latency to enable interpolation
    df_sorted = df.sort_values(latency_col)
    latencies = np.array(df_sorted[latency_col])
    recalls = np.array(df_sorted[recall_col])

    # Check if target latency is within the range
    if target_latency < latencies.min() or target_latency > latencies.max():
        return None

    # If exact match exists, return it
    exact_matches = df_sorted[df_sorted[latency_col] == target_latency]
    if len(exact_matches) > 0:
        return float(exact_matches[recall_col].values[0])

    # Linear interpolation
    try:
        interpolated_recall = np.interp(target_latency, latencies, recalls)
        return float(interpolated_recall)
    except Exception:
        return None


def find_adaptive_recall_threshold(
    df: pd.DataFrame,
    construction_params: list,
) -> Tuple[float, bool]:
    """Find an adaptive recall threshold that works for all construction parameter sets.

    Args:
        df: DataFrame with results
        construction_params: List of construction parameter names

    Returns:
        Tuple of (threshold, success_flag)
    """
    construction_groups = df.groupby(construction_params)

    recall_lbs = [
        float(group_df["recall_at_k"].min()) for _, group_df in construction_groups
    ]
    recall_ubs = [
        float(group_df["recall_at_k"].max()) for _, group_df in construction_groups
    ]

    if max(recall_lbs) > min(recall_ubs):
        return 0.0, False
    else:
        return (max(recall_lbs) + min(recall_ubs)) / 2, True


def find_adaptive_latency_threshold(
    df: pd.DataFrame,
    construction_params: list,
) -> Tuple[float, bool]:
    """Find an adaptive latency threshold that works for all construction parameter sets.

    Args:
        df: DataFrame with results
        construction_params: List of construction parameter names

    Returns:
        Tuple of (threshold, success_flag)
    """
    construction_groups = df.groupby(construction_params)

    latency_lbs = [
        float(group_df["query_lat_avg"].min()) for _, group_df in construction_groups
    ]
    latency_ubs = [
        float(group_df["query_lat_avg"].max()) for _, group_df in construction_groups
    ]

    if max(latency_lbs) > min(latency_ubs):
        return 0.0, False
    else:
        return (max(latency_lbs) + min(latency_ubs)) / 2, True


def create_param_label(row, construction_params, search_params):
    """Create abbreviated parameter label for plotting."""
    # Abbreviate parameter names
    abbrev_map = {
        "construction_ef": "c_ef",
        "max_sl_size": "max_sl",
        "search_ef": "s_ef",
        "beam_size": "beam",
        "variance_boost": "var_b",
        "ivf_cluster_size": "ivf_cs",
        "graph_degree": "gr_deg",
        "ivf_max_iter": "ivf_mi",
        "ivf_search_radius": "ivf_sr",
        "graph_search_L": "gr_sL",
        "m_beta": "m_β",
        "nprobe": "npr",
        "nlist": "nl",
        "gamma": "γ",
    }

    # Construction params
    const_parts = []
    for param in construction_params:
        short_name = abbrev_map.get(param, param)
        value = row[param]
        if hasattr(value, "item"):
            value = value.item()
        const_parts.append(f"{short_name}={value}")

    # Search params
    search_parts = []
    for param in search_params:
        short_name = abbrev_map.get(param, param)
        value = row[param]
        if hasattr(value, "item"):
            value = value.item()
        search_parts.append(f"{short_name}={value}")

    # Combine with separator
    const_str = ",".join(const_parts) if const_parts else ""
    search_str = ",".join(search_parts) if search_parts else ""

    if const_str and search_str:
        return f"C[{const_str}] S[{search_str}]"
    elif const_str:
        return f"C[{const_str}]"
    elif search_str:
        return f"S[{search_str}]"
    else:
        return "default"


def plot_recall_latency_by_baseline(
    base_output_dir: str | Path = "output/complex_predicate2",
    output_dir: str | Path = "/tmp",
):
    """Plot recall vs latency for different parameter sets for each baseline.

    Creates one figure per baseline showing recall-latency curves for different
    construction parameter combinations.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    dataset_key = DATASET_CONFIG["dataset_key"]
    test_size = DATASET_CONFIG["test_size"]

    for baseline in BASELINE_INFO:
        baseline_name = baseline["name"]
        algorithm_name = baseline["algorithm_name"]
        construction_params = baseline["construction_params"]
        search_params = baseline["search_params"]

        print(f"Creating plot for {baseline_name}...")

        try:
            # Load algorithm results
            df = load_algo_results(
                base_output_dir=base_output_dir,
                algorithm_name=algorithm_name,
                dataset_key=dataset_key,
                test_size=test_size,
            )

            # Select template based on algorithm (special case for Parlay IVF)
            template_name = "AND" if algorithm_name == "parlay_ivf" else "OR"
            template_df = df[df["template"] == template_name].copy()
            if len(template_df) == 0:
                print(
                    f"  No {template_name} template results for {baseline_name}, skipping..."
                )
                continue

            # Create figure
            plt.figure(figsize=(12, 8))

            # Group by construction parameters and plot each group
            construction_groups = template_df.groupby(construction_params)
            colors = getattr(cm, "tab10")(np.linspace(0, 1, len(construction_groups)))

            for i, (construction_values, group_df) in enumerate(construction_groups):
                # Create construction param label
                if len(construction_params) == 1:
                    const_label = f"{construction_params[0]}={construction_values}"
                else:
                    try:
                        const_pairs = [
                            f"{param}={val}"
                            for param, val in zip(
                                construction_params, list(construction_values)
                            )
                        ]
                        const_label = ",".join(const_pairs)
                    except TypeError:
                        # Handle case where construction_values is not iterable
                        const_label = f"{construction_params[0]}={construction_values}"

                # Sort by recall for better line plotting
                group_df_sorted = group_df.sort_values("recall_at_k").reset_index(
                    drop=True
                )

                # Plot line for this construction config
                plt.plot(
                    group_df_sorted["recall_at_k"],
                    group_df_sorted["query_lat_avg"],
                    "o-",
                    color=colors[i],
                    label=const_label,
                    alpha=0.7,
                    linewidth=2,
                )

            plt.xlabel("Recall@k")
            plt.ylabel("Query Latency (s)")
            plt.title(
                f"{baseline_name} - Recall vs Latency\n({template_name} Template, {dataset_key})"
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            safe_name = (
                baseline_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
            )
            plot_path = output_dir / f"recall_latency_{safe_name.lower()}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"  Saved plot to {plot_path}")
            plt.close()

        except Exception as e:
            print(f"  Error creating plot for {baseline_name}: {e}")
            continue

    print(f"\nAll plots saved to {output_dir}")


def load_algo_results(
    base_output_dir: str | Path,
    algorithm_name: str,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
) -> pd.DataFrame:
    """Load results from algorithm output directory.

    Args:
        base_output_dir: Base output directory containing algorithm results
        algorithm_name: Algorithm name (e.g., 'curator', 'shared_hnsw', etc.)
        dataset_key: Dataset key (e.g., 'yfcc100m')
        test_size: Test size fraction

    Returns:
        DataFrame with results including template, recall, latency, and other metrics
    """
    algo_dir = Path(base_output_dir) / algorithm_name
    results_dir = algo_dir / f"{dataset_key}_test{test_size}" / "results"

    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    all_results = []
    for json_file in results_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

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
                        # Extract configuration parameters from filename or result
                        config_info = {
                            "filename": json_file.name,
                            "template": template,
                            **per_template_result,
                        }

                        # Add any global parameters from the result (without global_ prefix)
                        for key, value in result.items():
                            if key != "per_template_results" and not key.startswith(
                                "_"
                            ):
                                config_info[key] = value

                        all_results.append(config_info)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue

    if not all_results:
        raise ValueError(f"No valid results found in {results_dir}")

    all_results_df = pd.DataFrame(all_results)

    # Clean up template names for better display
    if "template" in all_results_df.columns:
        all_results_df["template"] = all_results_df["template"].str.split(" ").str[0]

    return all_results_df


# Define algorithm information with parameter categorization
# Updated: beam_size and variance_boost are now treated as construction params
#  for curator algorithms since we do not intend to modify them at runtime.
BASELINE_INFO = [
    {
        "name": "Curator",
        "algorithm_name": "curator",
        "construction_params": ["nlist", "max_sl_size", "beam_size", "variance_boost"],
        "search_params": ["search_ef"],
    },
    {
        "name": "Curator (Indexed)",
        "algorithm_name": "curator_with_index",
        "construction_params": ["nlist", "max_sl_size", "beam_size", "variance_boost"],
        "search_params": ["search_ef"],
    },
    {
        "name": "Shared HNSW",
        "algorithm_name": "shared_hnsw",
        "construction_params": ["construction_ef", "m"],
        "search_params": ["search_ef"],
    },
    {
        "name": "Per-Predicate HNSW",
        "algorithm_name": "per_predicate_hnsw",
        "construction_params": ["construction_ef", "m"],
        "search_params": ["search_ef"],
    },
    {
        "name": "Shared IVF",
        "algorithm_name": "shared_ivf",
        "construction_params": ["nlist"],
        "search_params": ["nprobe"],
    },
    {
        "name": "Per-Predicate IVF",
        "algorithm_name": "per_predicate_ivf",
        "construction_params": ["nlist"],
        "search_params": ["nprobe"],
    },
    {
        "name": "Parlay IVF",
        "algorithm_name": "parlay_ivf",
        "construction_params": [
            "ivf_cluster_size",
            "graph_degree",
            "ivf_max_iter",
        ],
        "search_params": ["ivf_search_radius", "graph_search_L"],
    },
    {
        "name": "ACORN",
        "algorithm_name": "acorn",
        "construction_params": ["m", "gamma", "m_beta"],
        "search_params": ["search_ef"],
    },
]


# Configuration for dataset
DATASET_CONFIG = {
    "dataset_key": "yfcc100m",
    "test_size": 0.01,
    "templates": ["OR"],  # default template, but Parlay IVF uses AND
}


def extract_optimal_complex_predicate_params(
    base_output_dir: str | Path = "output/complex_predicate2",
    output_path: (
        str | Path
    ) = "benchmark/complex_predicate/optimal_baseline_params.json.in",
):
    """Extract optimal construction parameters for each algorithm on complex predicate datasets.

    Uses linear interpolation to estimate search latency at target recall threshold.
    Falls back to latency-based selection if recall threshold approach fails.
    """
    optimal_params = {}

    for baseline in BASELINE_INFO:
        baseline_name = baseline["name"]
        algorithm_name = baseline["algorithm_name"]
        optimal_params[baseline_name] = {}

        dataset_key = DATASET_CONFIG["dataset_key"]
        test_size = DATASET_CONFIG["test_size"]

        # Special handling for Parlay IVF: use AND template instead of OR
        if algorithm_name == "parlay_ivf":
            templates = ["AND"]
        else:
            templates = DATASET_CONFIG["templates"]

        print(f"Processing {baseline_name} on {dataset_key}...")

        try:
            # Load algorithm results
            df = load_algo_results(
                base_output_dir=base_output_dir,
                algorithm_name=algorithm_name,
                dataset_key=dataset_key,
                test_size=test_size,
            )

            print(f"  Loaded {len(df)} result rows")

            optimal_params[baseline_name] = {dataset_key: {}}

            # Process each template separately
            for template in templates:
                template_df = df[df["template"] == template].copy()

                if len(template_df) == 0:
                    print(f"  Warning: No results for template {template}")
                    continue

                print(f"  Processing template {template} ({len(template_df)} rows)")

                # Try recall-based approach first
                adaptive_threshold, threshold_success = find_adaptive_recall_threshold(
                    template_df, baseline["construction_params"]
                )

                optimal_construction_params = None
                approach_used = "recall"

                if threshold_success:
                    print(
                        f"  Using adaptive recall threshold: {adaptive_threshold:.3f}"
                    )
                    optimal_construction_params = find_optimal_construction_params(
                        template_df, baseline, adaptive_threshold
                    )

                # Fall back to latency-based approach if recall approach fails
                if optimal_construction_params is None:
                    print(
                        "  Recall-based approach failed, trying latency-based approach..."
                    )
                    latency_threshold, latency_success = (
                        find_adaptive_latency_threshold(
                            template_df, baseline["construction_params"]
                        )
                    )

                    if latency_success:
                        print(
                            f"  Using adaptive latency threshold: {latency_threshold:.6f}"
                        )
                        optimal_construction_params = (
                            find_optimal_construction_params_by_latency(
                                template_df, baseline, latency_threshold
                            )
                        )
                        approach_used = "latency"
                        adaptive_threshold = latency_threshold

                if optimal_construction_params is None:
                    print(
                        f"  Warning: Could not find optimal construction params for template {template}"
                    )
                    continue

                # Extract all search parameter combinations for this construction config on this template
                best_config_df = template_df.copy()
                for param in baseline["construction_params"]:
                    best_config_df = best_config_df[
                        best_config_df[param] == optimal_construction_params[param]
                    ]

                if len(best_config_df) == 0:
                    print(
                        f"  Warning: No results found for optimal construction params on template {template}"
                    )
                    # Use Pareto front selection as fallback
                    pareto_df = select_pareto_front(
                        template_df, x_key="query_lat_avg", y_key="recall_at_k"
                    )
                    if len(pareto_df) > 0:
                        best_config_df = pareto_df
                    else:
                        continue

                # Extract unique search parameter values for each parameter
                search_param_combinations = {}
                for param in baseline["search_params"]:
                    unique_values = []
                    for _, row in best_config_df.iterrows():
                        value = row[param]
                        # Convert numpy types to native Python types
                        if hasattr(value, "item"):
                            value = value.item()
                        if value not in unique_values:
                            unique_values.append(value)
                    # Sort values for consistency
                    if len(unique_values) > 0 and isinstance(
                        unique_values[0], (int, float)
                    ):
                        unique_values.sort()
                    search_param_combinations[param] = unique_values

                # Store results for this template
                result_info = {
                    "construction_params": optimal_construction_params,
                    "search_param_combinations": search_param_combinations,
                    "approach_used": approach_used,
                }

                if approach_used == "recall":
                    result_info["adaptive_recall_threshold"] = adaptive_threshold
                else:
                    result_info["adaptive_latency_threshold"] = adaptive_threshold

                optimal_params[baseline_name][dataset_key][template] = result_info

                print(
                    f"  Optimal construction params for {template} (via {approach_used}): {optimal_construction_params}"
                )
                print(
                    f"  Found {len(search_param_combinations)} search parameter arrays"
                )

        except Exception as e:
            print(f"Error processing {baseline_name} on {dataset_key}: {e}")
            continue

    # Save to JSON file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(optimal_params, f, indent=2)

    print(f"\nOptimal parameters saved to {output_path}")
    return optimal_params


def find_optimal_construction_params(
    df: pd.DataFrame,
    baseline: Dict,
    target_recall: float = 0.95,
) -> Optional[Dict]:
    """Find optimal construction parameters by comparing interpolated latency at target recall.

    For baselines with multiple search params, first selects Pareto optimal search configs
    for each build config, then does linear interpolation on the Pareto front.

    Args:
        df: DataFrame with results
        baseline: Baseline information including construction and search params
        target_recall: Target recall threshold for comparison

    Returns:
        Dictionary of optimal construction parameters, or None if not found
    """
    construction_params = baseline["construction_params"]
    search_params = baseline["search_params"]

    # Group by construction parameters
    construction_groups = df.groupby(construction_params)

    best_latency = float("inf")
    best_construction = None

    for construction_values, group_df in construction_groups:
        try:
            const_dict = dict(zip(construction_params, list(construction_values)))
            print(f"    Evaluating construction config: {const_dict}")
        except TypeError:
            # Handle single parameter case
            const_dict = {construction_params[0]: construction_values}
            print(f"    Evaluating construction config: {const_dict}")

        # If multiple search parameters exist, find Pareto front first
        if len(search_params) > 1:
            pareto_df = select_pareto_front(
                group_df, x_key="query_lat_avg", y_key="recall_at_k"
            )
            if len(pareto_df) == 0:
                print(f"      No Pareto optimal points found")
                continue
            eval_df = pareto_df
        else:
            eval_df = group_df

        # Interpolate latency at target recall
        interpolated_latency = interpolate_latency_at_recall(
            eval_df, target_recall=target_recall
        )

        if interpolated_latency is None:
            print(f"      Could not interpolate latency at recall {target_recall}")
            continue

        print(
            f"      Interpolated latency at recall {target_recall}: {interpolated_latency:.4f}"
        )

        # Track the best construction config
        if interpolated_latency < best_latency:
            best_latency = interpolated_latency
            best_construction = construction_values

    if best_construction is None:
        return None

    # Convert to dictionary
    optimal_construction_params = {}
    if isinstance(best_construction, (list, tuple)):
        for param, value in zip(construction_params, best_construction):
            # Convert numpy types to native Python types for JSON serialization
            if hasattr(value, "item"):
                value = value.item()
            optimal_construction_params[param] = value
    else:
        # Single parameter case
        param = construction_params[0]
        value = best_construction
        if hasattr(value, "item"):
            value = value.item()
        optimal_construction_params[param] = value

    print(
        f"    Best construction config: {optimal_construction_params} (latency: {best_latency:.4f})"
    )
    return optimal_construction_params


def find_optimal_construction_params_by_latency(
    df: pd.DataFrame,
    baseline: Dict,
    target_latency: float,
) -> Optional[Dict]:
    """Find optimal construction parameters by comparing interpolated recall at target latency.

    For baselines with multiple search params, first selects Pareto optimal search configs
    for each build config, then does linear interpolation on the Pareto front.

    Args:
        df: DataFrame with results
        baseline: Baseline information including construction and search params
        target_latency: Target latency threshold for comparison

    Returns:
        Dictionary of optimal construction parameters, or None if not found
    """
    construction_params = baseline["construction_params"]
    search_params = baseline["search_params"]

    # Group by construction parameters
    construction_groups = df.groupby(construction_params)

    best_recall = -1.0
    best_construction = None

    for construction_values, group_df in construction_groups:
        try:
            const_dict = dict(zip(construction_params, list(construction_values)))
            print(f"    Evaluating construction config: {const_dict}")
        except TypeError:
            # Handle single parameter case
            const_dict = {construction_params[0]: construction_values}
            print(f"    Evaluating construction config: {const_dict}")

        # If multiple search parameters exist, find Pareto front first
        if len(search_params) > 1:
            pareto_df = select_pareto_front(
                group_df, x_key="query_lat_avg", y_key="recall_at_k"
            )
            if len(pareto_df) == 0:
                print(f"      No Pareto optimal points found")
                continue
            eval_df = pareto_df
        else:
            eval_df = group_df

        # Interpolate recall at target latency
        interpolated_recall = interpolate_recall_at_latency(
            eval_df, target_latency=target_latency
        )

        if interpolated_recall is None:
            print(f"      Could not interpolate recall at latency {target_latency}")
            continue

        print(
            f"      Interpolated recall at latency {target_latency}: {interpolated_recall:.4f}"
        )

        # Track the best construction config (highest recall)
        if interpolated_recall > best_recall:
            best_recall = interpolated_recall
            best_construction = construction_values

    if best_construction is None:
        return None

    # Convert to dictionary
    optimal_construction_params = {}
    if isinstance(best_construction, (list, tuple)):
        for param, value in zip(construction_params, best_construction):
            # Convert numpy types to native Python types for JSON serialization
            if hasattr(value, "item"):
                value = value.item()
            optimal_construction_params[param] = value
    else:
        # Single parameter case
        param = construction_params[0]
        value = best_construction
        if hasattr(value, "item"):
            value = value.item()
        optimal_construction_params[param] = value

    print(
        f"    Best construction config: {optimal_construction_params} (recall: {best_recall:.4f})"
    )
    return optimal_construction_params


if __name__ == "__main__":
    fire.Fire(
        {
            "extract_optimal_params": extract_optimal_complex_predicate_params,
            "plot_recall_latency": plot_recall_latency_by_baseline,
        }
    )
