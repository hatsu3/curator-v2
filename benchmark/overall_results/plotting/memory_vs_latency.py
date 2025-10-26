"""
Memory vs latency plotting for analyzing build time and memory footprint relationships.
"""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Union

import fire
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import yaml

from benchmark.overall_results.plotting.build_time import load_algorithm_build_time
from benchmark.overall_results.plotting.memory_footprint import (
    load_algorithm_memory_usage,
)
from benchmark.overall_results.plotting.recall_vs_latency import (
    discover_baseline_results,
    preprocess_all_baselines,
)
from benchmark.profiler import Dataset

# Cache directory for process_dataset_data results
CACHE_DIR = Path("/tmp/curator_plot_cache")


def get_cache_key(
    results_dir: str, dataset_name: str, selectivity_threshold: float, target_recall: float
) -> str:
    """Generate cache key based on processing parameters."""
    key_string = f"{results_dir}_{dataset_name}_{selectivity_threshold}_{target_recall}"
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cache_path(
    results_dir: str, dataset_name: str, selectivity_threshold: float, target_recall: float
) -> Path:
    """Get cache file path for process_dataset_data results."""
    cache_key = get_cache_key(results_dir, dataset_name, selectivity_threshold, target_recall)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{dataset_name}_{cache_key}.pkl"


def load_from_cache(cache_path: Path) -> pd.DataFrame:
    """Load DataFrame from cache."""
    print(f"Loading from cache: {cache_path}")
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def save_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save DataFrame to cache."""
    print(f"Saving to cache: {cache_path}")
    with open(cache_path, "wb") as f:
        pickle.dump(df, f)


# Dataset-specific configuration (reused from recall_vs_latency.py)
DATASET_LABELS_PER_GROUP = {
    "yfcc100m": {
        "labels_per_group": 100,  # 1000 unique labels
        "default_test_size": 0.001,
    },
    "yfcc100m-10m": {
        "labels_per_group": 100,  # 1000 unique labels
        "default_test_size": 0.001,
    },
    "arxiv-large-10": {
        "labels_per_group": 10,  # 100 unique labels
        "default_test_size": 0.005,
    },
}


def filter_queries_by_selectivity(
    per_config_results: dict,
    dataset: Dataset,
    selectivity_threshold: float,
    label_selectivities_cache: dict[int, float] | None = None,
) -> dict:
    """Filter queries by absolute selectivity threshold.

    Args:
        per_config_results: Configuration results with query_recalls and query_latencies
        dataset: Dataset object containing label selectivities
        selectivity_threshold: Absolute selectivity threshold (e.g., 0.15)
        label_selectivities_cache: Cached label selectivities to avoid recomputation

    Returns:
        Dictionary with filtered recalls and latencies for queries under the threshold
    """
    # Use cached selectivities if provided, otherwise compute and cache
    if label_selectivities_cache is None:
        label_selectivities_cache = dataset.label_selectivities

    recall_gen = iter(per_config_results["query_recalls"])
    latency_gen = iter(per_config_results["query_latencies"])

    filtered_recalls = []
    filtered_latencies = []

    # Iterate through test data and filter by selectivity
    for __, label_list in zip(dataset.test_vecs, dataset.test_mds):
        for label in label_list:
            recall, latency = next(recall_gen), next(latency_gen)

            # Check if this label's selectivity is <= threshold
            label_selectivity = label_selectivities_cache.get(label, 1.0)
            if label_selectivity <= selectivity_threshold:
                filtered_recalls.append(recall)
                filtered_latencies.append(latency)

    return {
        "query_recalls": filtered_recalls,
        "query_latencies": filtered_latencies,
    }


def load_recall_latency_results(
    results_path: Path | str,
    dataset_key: str,
    test_size: float,
    selectivity_threshold: float,
    dataset: Dataset | None = None,
) -> pd.DataFrame:
    """Load and process recall vs latency results for queries under a selectivity threshold.

    For each search configuration, filter queries by selectivity and compute average recall and latency.
    Returns a DataFrame where each row represents one search configuration.
    """
    results_path = Path(results_path)

    if dataset is None:
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    # Cache label selectivities to avoid recomputation
    print(f"  Computing label selectivities...")
    label_selectivities_cache = dataset.label_selectivities

    # Load results
    if results_path.suffix == ".csv":
        results_df = pd.read_csv(results_path)
        for col in ["query_latencies", "query_recalls"]:
            results_df[col] = results_df[col].apply(json.loads)
        results = results_df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file extension {results_path.suffix}")

    # Process each search configuration separately
    config_results = []

    for res in results:
        filtered_res = filter_queries_by_selectivity(
            res, dataset, selectivity_threshold, label_selectivities_cache
        )

        # Skip configurations with no qualifying queries
        if not filtered_res["query_recalls"]:
            continue

        # Compute average recall and latency for this configuration
        avg_recall = np.mean(filtered_res["query_recalls"])
        avg_latency = np.mean(filtered_res["query_latencies"])

        config_results.append(
            {
                "recall": avg_recall,
                "latency": avg_latency,
            }
        )

    if not config_results:
        return pd.DataFrame(columns=["recall", "latency"])

    # Create DataFrame where each row represents one search configuration
    return pd.DataFrame(config_results)


def interpolate_latency_at_recall(
    recall_latency_df: pd.DataFrame, target_recall: float = 0.9
) -> float:
    """Use linear interpolation to estimate latency at target recall.

    Args:
        recall_latency_df: DataFrame containing recall and latency data
        target_recall: Target recall to interpolate to

    Returns:
        Interpolated latency at target recall
    """
    # Sort by recall to ensure proper interpolation
    df_sorted = recall_latency_df.sort_values("recall")

    recalls = df_sorted["recall"].to_numpy()
    latencies = df_sorted["latency"].to_numpy()

    # Check if we have valid data for interpolation
    if len(recalls) == 0:
        raise ValueError("No recall data available")

    min_recall = float(recalls.min())
    max_recall = float(recalls.max())

    # Check if all results achieve > target_recall
    if min_recall > target_recall:
        # Use the latency of the lowest recall configuration and print warning
        lowest_recall_idx = recalls.argmin()
        lowest_recall_latency = float(latencies[lowest_recall_idx])
        print(
            f"  WARNING: All results achieve > {target_recall} recall (min: {min_recall:.3f}). "
            f"Using latency at lowest recall: {lowest_recall_latency*1000:.2f} ms"
        )
        return lowest_recall_latency

    # Check if no results achieve > target_recall
    if max_recall <= target_recall:
        raise ValueError(
            f"No results achieve > {target_recall} recall (max: {max_recall:.3f})"
        )

    # Use linear interpolation
    interpolated_latency = np.interp(target_recall, recalls, latencies)

    return float(interpolated_latency)


def process_dataset_data(
    results_dir: Union[Path, str],
    dataset_name: str,
    selectivity_threshold: float,
    target_recall: float,
    dataset_config: dict,
    algorithm_dir_mapping: dict,
    short_names: dict,
) -> pd.DataFrame:
    """
    Process data for a single dataset and return DataFrame with plotting data.

    Args:
        results_dir: Root directory containing baseline results
        dataset_name: Dataset name ("yfcc100m" or "arxiv")
        selectivity_threshold: Selectivity percentile to analyze (0.0 to 1.0)
        target_recall: Target recall for latency interpolation
        dataset_config: Dataset configuration dictionary
        algorithm_dir_mapping: Mapping from display names to algorithm directories
        short_names: Mapping from display names to short names for annotations

    Returns:
        DataFrame with columns: baseline, short_name, latency_ms, memory_gb, build_time_sec
    """
    dataset_key = dataset_config[dataset_name]["dataset_key"]
    test_size = dataset_config[dataset_name]["test_size"]
    dataset_info = dataset_config[dataset_name]["info"]

    print(f"=== Processing {dataset_config[dataset_name]['display_name']} ===")
    print(f"Dataset key: {dataset_key}")
    print(f"Test size: {test_size}")
    print(f"Selectivity threshold: {selectivity_threshold}")
    print(f"Target recall: {target_recall}")
    print()

    # Discover available baselines
    baseline_results = discover_baseline_results(results_dir, dataset_key, test_size)

    # Load dataset once
    print(f"Loading dataset {dataset_key} with test_size={test_size} ...")
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    # Calculate base vector storage size for index overhead calculation
    # Calculate size of base vectors: num_vectors * embedding_dim * sizeof(float32)
    num_vectors = len(dataset.train_vecs)
    embedding_dim = (
        dataset.train_vecs.shape[1]
        if hasattr(dataset.train_vecs, "shape")
        else len(dataset.train_vecs[0])
    )
    base_vector_storage_bytes = num_vectors * embedding_dim * 4  # 4 bytes per float32
    base_vector_storage_gb = base_vector_storage_bytes / (1024**3)
    print(
        f"Base vector storage: {base_vector_storage_gb:.2f} GB ({num_vectors} vectors × {embedding_dim}D × 4 bytes)"
    )

    # Process each baseline
    plot_data = []

    for baseline_name, result_path in baseline_results.items():
        if result_path is None:
            print(f"Skipping {baseline_name}: no results found")
            continue

        try:
            print(f"Processing {baseline_name}...")

            # Load recall vs latency results
            recall_latency_df = load_recall_latency_results(
                results_path=result_path,
                dataset_key=dataset_key,
                test_size=test_size,
                selectivity_threshold=selectivity_threshold,
                dataset=dataset,
            )

            if recall_latency_df.empty:
                print(f"  No data found for {baseline_name}")
                continue

            # Interpolate latency at target recall
            try:
                interpolated_latency = interpolate_latency_at_recall(
                    recall_latency_df, target_recall
                )
                print(
                    f"  Latency at recall {target_recall}: {interpolated_latency*1000:.2f} ms"
                )
            except ValueError as e:
                print(f"  Cannot interpolate latency for {baseline_name}: {e}")
                continue

            # Load memory footprint
            algo_dir = algorithm_dir_mapping.get(baseline_name)
            if algo_dir is None:
                print(f"  No algorithm directory mapping for {baseline_name}")
                continue

            memory_kb = load_algorithm_memory_usage(
                output_dir=str(results_dir),
                algorithm=algo_dir,
                dataset_key=dataset_key,
                test_size=test_size,
            )

            if memory_kb is None:
                print(f"  No memory data found for {baseline_name}")
                continue

            total_memory_gb = memory_kb / 1024 / 1024

            # Calculate index overhead by subtracting base vector storage
            index_overhead_gb = total_memory_gb - base_vector_storage_gb
            print(f"  Index overhead: {index_overhead_gb:.2f} GB")

            # Load build time
            build_time_sec = load_algorithm_build_time(
                output_dir=str(results_dir),
                algorithm=algo_dir,
                dataset_key=dataset_key,
                test_size=test_size,
                dataset_info=dataset_info,
            )

            if build_time_sec is not None:
                print(f"  Build time: {build_time_sec:.2f} s")
            else:
                print(f"  Build time: Not available")

            plot_data.append(
                {
                    "baseline": baseline_name,
                    "short_name": short_names.get(baseline_name, baseline_name),
                    "latency_ms": interpolated_latency * 1000,  # Convert to ms
                    "memory_gb": index_overhead_gb,  # Always use index overhead
                    "build_time_sec": build_time_sec,
                }
            )

        except Exception as e:
            print(f"Error processing {baseline_name}: {e}")

    if not plot_data:
        print(f"Warning: No valid data found for {dataset_name}")
        return pd.DataFrame(
            columns=[
                "baseline",
                "short_name",
                "latency_ms",
                "memory_gb",
                "build_time_sec",
            ]
        )

    return pd.DataFrame(plot_data)


# Map dataset names to dataset keys and test sizes
DATASET_CONFIG = {
    "yfcc100m": {
        "display_name": "YFCC100M",
        "dataset_key": "yfcc100m-10m",
        "test_size": 0.001,
        "info": {
            "num_vecs": int(10_000_000 * (1 - 0.001)),  # 9,990,000
            "num_mds": 55293233,
        },
    },
    "arxiv": {
        "display_name": "arXiv",
        "dataset_key": "arxiv-large-10",
        "test_size": 0.005,
        "info": {
            "num_vecs": int(2_000_000 * (1 - 0.005)),  # 1,990,000
            "num_mds": 19755960,
        },
    },
}

# Algorithm directory mapping for memory loading
ALGORITHM_DIR_MAPPING = {
    "Curator": "curator",
    "Per-Label HNSW": "per_label_hnsw",
    "Per-Label IVF": "per_label_ivf",
    "Shared HNSW": "shared_hnsw",
    "Shared IVF": "shared_ivf",
    "Parlay IVF": "parlay_ivf",
    "Filtered DiskANN": "filtered_diskann",
    "ACORN-1": "acorn_1",
    r"ACORN-$\gamma$": "acorn_gamma",
    "Pg-HNSW": "pgvector_hnsw",
    "Pg-IVF": "pgvector_ivf",
}

# Short names for annotations
SHORT_NAMES = {
    "Curator": "Curator",
    "Per-Label HNSW": "P-H",
    "Per-Label IVF": "P-I",
    "Shared HNSW": "S-H",
    "Shared IVF": "S-I",
    "Parlay IVF": "Par",
    "Filtered DiskANN": "Disk",
    "ACORN-1": "A-1",
    r"ACORN-$\gamma$": "A-γ",
    "Pg-HNSW": "Pg-H",
    "Pg-IVF": "Pg-I",
}

# Consistent color mapping for baselines across all plots
BASELINE_COLORS = {
    "Curator": "#1f77b4",  # Blue
    "Per-Label HNSW": "#ff7f0e",  # Orange
    "Per-Label IVF": "#2ca02c",  # Green
    "Shared HNSW": "#d62728",  # Red
    "Shared IVF": "#9467bd",  # Purple
    "Parlay IVF": "#8c564b",  # Brown
    "Filtered DiskANN": "#e377c2",  # Pink
    "ACORN-1": "#7f7f7f",  # Gray
    r"ACORN-$\gamma$": "#bcbd22",  # Olive
    "Pg-HNSW": "#17becf",  # Teal
    "Pg-IVF": "#aec7e8",  # Light blue
}


def load_annotation_config(config_path: Union[Path, str]) -> dict:
    """Load annotation offset configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary with annotation offsets per baseline per dataset
    """
    config_path = Path(config_path)
    if not config_path.exists():
        print(
            f"Warning: Annotation config file {config_path} not found. Using default offsets."
        )
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        print(f"Loaded annotation configuration from {config_path}")
        return config.get("annotation_offsets", {})
    except Exception as e:
        print(f"Error loading annotation config from {config_path}: {e}")
        return {}


def generate_sample_annotation_config(
    dataset_dataframes: dict,
    config_path: Union[Path, str],
    default_offset: tuple = (5, 2),
) -> None:
    """Generate a sample annotation configuration file based on current data.

    Args:
        dataset_dataframes: Dictionary of dataset DataFrames with baseline data
        config_path: Path to save the sample configuration file
        default_offset: Default (x, y) offset for annotations
    """
    config_path = Path(config_path)

    # Create annotation offsets structure
    annotation_offsets = {}

    for ds_name, df in dataset_dataframes.items():
        dataset_display_name = DATASET_CONFIG[ds_name]["display_name"]
        annotation_offsets[dataset_display_name] = {}

        for _, row in df.iterrows():
            baseline_name = row["baseline"]
            annotation_offsets[dataset_display_name][baseline_name] = {
                "x_offset": default_offset[0],
                "y_offset": default_offset[1],
                "ha": "left",  # horizontal alignment
                "va": "bottom",  # vertical alignment
            }

    # Create full configuration structure
    config = {
        "annotation_offsets": annotation_offsets,
        "description": "Configuration file for annotation offsets in memory vs latency plots",
        "usage": "Modify x_offset and y_offset values to adjust annotation positions. "
        "Positive x_offset moves annotations right, positive y_offset moves them up. "
        "ha (horizontal alignment) can be: left, center, right. "
        "va (vertical alignment) can be: bottom, center, top.",
    }

    # Save configuration
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated sample annotation configuration at {config_path}")


def get_annotation_offset(
    annotation_config: dict,
    dataset_display_name: str,
    baseline_name: str,
    default_offset: tuple = (5, 2),
) -> dict:
    """Get annotation offset and alignment for a specific baseline and dataset.

    Args:
        annotation_config: Loaded annotation configuration
        dataset_display_name: Display name of the dataset
        baseline_name: Name of the baseline
        default_offset: Default (x, y) offset if not configured

    Returns:
        Dictionary with offset and alignment parameters
    """
    if (
        dataset_display_name in annotation_config
        and baseline_name in annotation_config[dataset_display_name]
    ):

        config = annotation_config[dataset_display_name][baseline_name]
        return {
            "x_offset": config.get("x_offset", default_offset[0]),
            "y_offset": config.get("y_offset", default_offset[1]),
            "ha": config.get("ha", "left"),
            "va": config.get("va", "bottom"),
        }
    else:
        return {
            "x_offset": default_offset[0],
            "y_offset": default_offset[1],
            "ha": "left",
            "va": "bottom",
        }


def plot_memory_vs_latency_vs_build_time(
    results_dir: Union[Path, str],
    output_path: Union[Path, str],
    annotation_config_path: Union[Path, str],
    dataset_names: list[str] | None = None,
    selectivity_threshold: float = 0.15,
    target_recall: float = 0.9,
    figsize: tuple = (7, 3),
    font_size: int = 14,
    ignore_cache: bool = False,
    skip_annotation_lines: bool = False,
):
    """
    Plot build time vs index overhead with marker size representing search latency.
    Creates side-by-side subplots for YFCC100M and arXiv datasets.

    Args:
        results_dir: Root directory containing baseline results
        output_path: Path to save the plot
        annotation_config_path: Path to YAML configuration file for annotation offsets
        dataset_names: List of dataset names to process (default: ["yfcc100m", "arxiv"])
        selectivity_threshold: Selectivity percentile to analyze (0.0 to 1.0)
        target_recall: Target recall for latency interpolation (default: 0.9)
        figsize: Figure size (width, height)
        font_size: Font size for the plot
    """
    # Determine which datasets to process
    if dataset_names is None:
        datasets_to_process = ["yfcc100m", "arxiv"]
    else:
        if not all(ds_name in DATASET_CONFIG for ds_name in dataset_names):
            raise ValueError(
                f"Some datasets are not supported: {dataset_names}. Must be one of {list(DATASET_CONFIG.keys())}"
            )
        datasets_to_process = dataset_names

    print(f"=== Build Time vs Index Overhead Analysis ===")
    print(f"Results directory: {results_dir}")
    print(f"Datasets: {datasets_to_process}")
    print(f"Selectivity threshold: {selectivity_threshold}")
    print(f"Target recall: {target_recall}")
    print(f"Output path: {output_path}")
    print()

    # Process data for each dataset
    dataset_dataframes = {}
    for ds_name in datasets_to_process:
        # Try to load from cache first
        cache_path = get_cache_path(
            str(results_dir), ds_name, selectivity_threshold, target_recall
        )
        use_cache = cache_path.exists() and not ignore_cache

        if use_cache:
            print(f"Using cache for {ds_name}. Pass ignore_cache=True to recompute.")
            df = load_from_cache(cache_path)
        else:
            # Process data and refresh cache
            df = process_dataset_data(
                results_dir=results_dir,
                dataset_name=ds_name,
                selectivity_threshold=selectivity_threshold,
                target_recall=target_recall,
                dataset_config=DATASET_CONFIG,
                algorithm_dir_mapping=ALGORITHM_DIR_MAPPING,
                short_names=SHORT_NAMES,
            )
            if not df.empty:
                save_to_cache(df, cache_path)

        if not df.empty:
            dataset_dataframes[ds_name] = df
        print()

    if not dataset_dataframes:
        raise ValueError("No valid data found for any dataset")

    # Load annotation configuration
    annotation_config = {}
    if annotation_config_path:
        annotation_config = load_annotation_config(annotation_config_path)

    # Create the plot
    plt.rcParams.update({"font.size": font_size})

    if len(datasets_to_process) == 1:
        fig, ax = plt.subplots(figsize=figsize)
        axes = [ax]
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Get global data ranges for consistent scaling across subplots
    all_build_times = []
    all_memory_values = []
    all_latencies = []

    for df in dataset_dataframes.values():
        # Check for missing build time data
        missing_build_times = [
            row["baseline"] for _, row in df.iterrows() if row["build_time_sec"] is None
        ]
        if missing_build_times:
            raise ValueError(
                f"Build time information is missing for the following algorithms: {missing_build_times}. "
                "Please ensure all algorithms have valid build time data before plotting."
            )

        all_build_times.extend([bt for bt in df["build_time_sec"] if bt is not None])
        all_memory_values.extend([mv for mv in df["memory_gb"] if mv is not None])
        all_latencies.extend([lat for lat in df["latency_ms"] if lat is not None])

    if not all_build_times:
        raise ValueError("No build time data available for any algorithm.")

    # Use fixed legend range for consistent sizing across plots
    fixed_legend_latencies = [1, 10, 100]  # 1ms, 10ms, 100ms
    fixed_legend_labels = ["1ms", "10ms", "100ms"]

    legend_min_latency = min(fixed_legend_latencies)
    legend_max_latency = max(fixed_legend_latencies)

    # Scale sizes from 30 to 300 based on latency (log scale)
    min_size, max_size = 30, 300

    # Plot each dataset in its subplot
    for i, (ds_name, df) in enumerate(dataset_dataframes.items()):
        ax = axes[i] if len(axes) > 1 else axes[0]
        dataset_display_name = DATASET_CONFIG[ds_name]["display_name"]

        # Calculate point sizes based on latency (log scale)
        sizes = []
        for latency_ms in df["latency_ms"]:
            # Log scale for better visualization using legend range
            # Clamp values to legend range to avoid extreme sizes
            clamped_latency = max(
                legend_min_latency, min(legend_max_latency, latency_ms)
            )
            if legend_max_latency == legend_min_latency:
                norm_size = 0.5  # Middle size if all latencies are the same
            else:
                norm_size = (np.log(clamped_latency) - np.log(legend_min_latency)) / (
                    np.log(legend_max_latency) - np.log(legend_min_latency)
                )
            size = min_size + norm_size * (max_size - min_size)
            sizes.append(size)

        # Create scatter plot with build time on x-axis, index overhead on y-axis
        # Use consistent color mapping based on baseline names
        colors = [
            BASELINE_COLORS.get(baseline, "#000000") for baseline in df["baseline"]
        ]
        scatter = ax.scatter(
            df["build_time_sec"],
            df["memory_gb"],
            s=sizes,
            alpha=0.7,
            c=colors,
        )

        # Add annotations with configurable offsets
        for _, row in df.iterrows():
            # Get annotation configuration for this baseline and dataset
            annotation_params = get_annotation_offset(
                annotation_config,
                dataset_display_name,
                row["baseline"],
                default_offset=(5, 2),
            )

            kwargs = dict(
                text=row["short_name"],
                xy=(row["build_time_sec"], row["memory_gb"]),
                xytext=(annotation_params["x_offset"], annotation_params["y_offset"]),
                textcoords="offset points",
                fontsize=font_size - 4,
                ha=annotation_params["ha"],
                va=annotation_params["va"],
            )
            if not skip_annotation_lines:
                kwargs["arrowprops"] = dict(
                    arrowstyle="-",
                    color="gray",
                    linewidth=0.8,
                    alpha=0.6,
                )
            ax.annotate(**kwargs)

        # Set scales and labels
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Build Time (s)")
        # Only show y-axis label for the leftmost subplot
        if i == 0:
            ax.set_ylabel("Index Overhead (GB)")
        ax.set_title(f"{dataset_display_name} (Sel ≤ {selectivity_threshold})")

        # Set ticks - use only powers of 10 for y-axis to avoid intermediate values like 6x10^0
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs="auto", numticks=10))

        # For y-axis, use only integer powers of 10 and ensure at least 2 ticks
        y_min, y_max = ax.get_ylim()
        y_min_log = int(np.floor(np.log10(y_min)))
        y_max_log = int(np.ceil(np.log10(y_max)))

        # Ensure we have at least 2 ticks by expanding range if needed
        if y_max_log <= y_min_log:
            y_max_log = y_min_log + 1

        y_ticks = [10**i for i in range(y_min_log, y_max_log + 1)]
        ax.set_yticks(y_ticks)
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs="auto", numticks=10))

        # Add grid
        ax.grid(True, alpha=0.3)

        # Add margin around the data points (use multiplication for log scales)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        margin_factor = 1.5  # 50% margin on each side for log scale
        x_margin_factor = 2.0  # Larger margin on x-axis to prevent annotation overflow
        ax.set_xlim(x_min / margin_factor, x_max * x_margin_factor)
        ax.set_ylim(y_min / margin_factor, y_max * margin_factor)

    # Create legend at the top of the figure
    # Always show all legend entries
    legend_latencies = fixed_legend_latencies
    legend_labels = fixed_legend_labels

    # Create legend with all fixed values
    legend_sizes = []
    for latency in legend_latencies:
        if legend_max_latency == legend_min_latency:
            size = (min_size + max_size) / 2
        else:
            norm_size = (np.log(latency) - np.log(legend_min_latency)) / (
                np.log(legend_max_latency) - np.log(legend_min_latency)
            )
            size = min_size + norm_size * (max_size - min_size)
        legend_sizes.append(size)

    # Create legend elements
    legend_elements = []
    for size, label in zip(legend_sizes, legend_labels):
        legend_elements.append(
            plt.scatter([], [], s=size, c="gray", alpha=0.7, label=label)
        )

        # Add legend to the right side with two-line title and multiple rows
    legend = fig.legend(
        handles=legend_elements,
        title="Latency\n(R=0.9)",  # Two-line title
        loc="center left",
        bbox_to_anchor=(1.00, 0.5),  # Position closer to the plots
        ncol=1,  # Single column (multiple rows)
        frameon=True,
        fontsize=font_size - 2,
        title_fontsize=font_size - 1,
        borderaxespad=0,  # Remove padding between legend and axes
        handletextpad=0.5,  # Space between legend marker and text
        labelspacing=0.8,  # Increase vertical space between legend entries
    )

    fig.tight_layout()

    # Save the figure
    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved successfully!")

    # Print summary
    print(f"\nSummary (Recall {target_recall}, Selectivity {selectivity_threshold}):")
    for ds_name, df in dataset_dataframes.items():
        print(f"\n{DATASET_CONFIG[ds_name]['display_name']}:")
        for _, row in df.iterrows():
            print(
                f"  {row['baseline']}: {row['build_time_sec']:.2f}s build, "
                f"{row['memory_gb']:.2f}GB overhead, {row['latency_ms']:.2f}ms latency"
            )


def generate_annotation_config(
    results_dir: Union[Path, str],
    output_path: Union[Path, str],
    dataset_names: list[str] | None = None,
    selectivity_threshold: float = 0.15,
    target_recall: float = 0.9,
) -> None:
    """
    Generate a sample annotation configuration file based on current data.

    Args:
        results_dir: Root directory containing baseline results
        output_path: Path to save the sample configuration file
        dataset_names: List of dataset names to process (default: ["yfcc100m", "arxiv"])
        selectivity_threshold: Selectivity percentile to analyze (0.0 to 1.0)
        target_recall: Target recall for latency interpolation (default: 0.9)
    """
    # Determine which datasets to process
    if dataset_names is None:
        datasets_to_process = ["yfcc100m", "arxiv"]
    else:
        if not all(ds_name in DATASET_CONFIG for ds_name in dataset_names):
            raise ValueError(
                f"Some datasets are not supported: {dataset_names}. Must be one of {list(DATASET_CONFIG.keys())}"
            )
        datasets_to_process = dataset_names

    print(f"=== Generating Annotation Configuration ===")
    print(f"Results directory: {results_dir}")
    print(f"Datasets: {datasets_to_process}")
    print(f"Output path: {output_path}")
    print()

    # Process data for each dataset
    dataset_dataframes = {}
    for ds_name in datasets_to_process:
        df = process_dataset_data(
            results_dir=results_dir,
            dataset_name=ds_name,
            selectivity_threshold=selectivity_threshold,
            target_recall=target_recall,
            dataset_config=DATASET_CONFIG,
            algorithm_dir_mapping=ALGORITHM_DIR_MAPPING,
            short_names=SHORT_NAMES,
        )
        if not df.empty:
            dataset_dataframes[ds_name] = df
        print()

    if not dataset_dataframes:
        raise ValueError("No valid data found for any dataset")

    # Generate sample configuration
    generate_sample_annotation_config(dataset_dataframes, output_path)


def plot_memory_vs_latency_per_selectivity(
    results_dir: Union[Path, str],
    dataset_name: str,
    output_path: Union[Path, str],
    percentiles: list[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    labels_per_group: int | None = None,
    target_recall: float = 0.9,
    force_reprocess: bool = False,
    figsize_per_subplot: tuple = (3, 3),
    font_size: int = 14,
):
    """
    Plot memory vs build time with latency represented by marker size for each selectivity group.
    Creates subplots for each selectivity percentile, similar to recall_vs_latency.py.

    Args:
        results_dir: Root directory containing baseline results
        dataset_name: Dataset name ("yfcc100m" or "arxiv")
        output_path: Path to save the plot
        percentiles: Selectivity percentiles to analyze
        labels_per_group: Number of labels per selectivity group
        target_recall: Target recall for latency interpolation
        force_reprocess: Whether to force reprocessing of raw results
        figsize_per_subplot: Size of each subplot (width, height)
        font_size: Font size for the plot
    """
    # Map dataset names to dataset keys and test sizes
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of {list(DATASET_CONFIG.keys())}"
        )

    dataset_key = DATASET_CONFIG[dataset_name]["dataset_key"]
    test_size = DATASET_CONFIG[dataset_name]["test_size"]
    dataset_info = DATASET_CONFIG[dataset_name]["info"]

    print(f"=== Memory vs Build Time Per Selectivity for {dataset_name} ===")
    print(f"Results directory: {results_dir}")
    print(f"Dataset key: {dataset_key}")
    print(f"Test size: {test_size}")
    print(f"Percentiles: {percentiles}")
    print(f"Target recall: {target_recall}")
    print(f"Output path: {output_path}")
    print()

    # Preprocess all available baseline results using imported function
    preprocessed_results = preprocess_all_baselines(
        results_dir=results_dir,
        dataset_key=dataset_key,
        test_size=test_size,
        labels_per_group=labels_per_group,
        percentiles=percentiles,
        force_reprocess=force_reprocess,
    )

    if not preprocessed_results:
        raise ValueError(f"No baseline results found in {results_dir}")

    print(f"\nFound {len(preprocessed_results)} baselines to process:")
    for baseline_name in preprocessed_results.keys():
        print(f"  - {baseline_name}")
    print()

    # Load dataset for base vector storage calculation
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    # Calculate base vector storage size for index overhead calculation
    num_vectors = len(dataset.train_vecs)
    embedding_dim = (
        dataset.train_vecs.shape[1]
        if hasattr(dataset.train_vecs, "shape")
        else len(dataset.train_vecs[0])
    )
    base_vector_storage_bytes = num_vectors * embedding_dim * 4  # 4 bytes per float32
    base_vector_storage_gb = base_vector_storage_bytes / (1024**3)
    print(
        f"Base vector storage: {base_vector_storage_gb:.2f} GB ({num_vectors} vectors × {embedding_dim}D × 4 bytes)"
    )

    # Load all preprocessed results and process per selectivity
    all_results = {}
    for baseline_name, preproc_path in preprocessed_results.items():
        try:
            df = pd.read_csv(preproc_path)
            df["baseline"] = baseline_name
            all_results[baseline_name] = df
            print(f"Loaded {len(df)} rows from {baseline_name}")
        except Exception as e:
            print(f"Error loading {baseline_name}: {e}")

    # Process each baseline to add memory and build time data
    processed_results = {}
    for baseline_name, df in all_results.items():
        # Get algorithm directory mapping
        algo_dir = ALGORITHM_DIR_MAPPING.get(baseline_name)
        if algo_dir is None:
            print(f"Warning: No algorithm directory mapping for {baseline_name}")
            continue

        # Load memory footprint
        memory_kb = load_algorithm_memory_usage(
            output_dir=str(results_dir),
            algorithm=algo_dir,
            dataset_key=dataset_key,
            test_size=test_size,
        )

        if memory_kb is None:
            print(f"Warning: No memory data found for {baseline_name}")
            continue

        total_memory_gb = memory_kb / 1024 / 1024
        index_overhead_gb = total_memory_gb - base_vector_storage_gb

        # Load build time
        build_time_sec = load_algorithm_build_time(
            output_dir=str(results_dir),
            algorithm=algo_dir,
            dataset_key=dataset_key,
            test_size=test_size,
            dataset_info=dataset_info,
        )

        if build_time_sec is None:
            print(f"Warning: No build time data found for {baseline_name}")
            continue

        # Add memory and build time data to each row
        df_copy = df.copy()
        df_copy["memory_gb"] = index_overhead_gb
        df_copy["build_time_sec"] = build_time_sec
        df_copy["short_name"] = SHORT_NAMES.get(baseline_name, baseline_name)

        # Interpolate latency at target recall for each selectivity group
        latency_data = []
        for percentile in percentiles:
            pct_data = df_copy[df_copy["percentile"] == percentile]
            if not pct_data.empty:
                # Use the recall and latency data to interpolate
                recalls = pct_data["recall"].values
                latencies = pct_data["latency"].values * 1000  # Convert to ms

                if len(recalls) > 0:
                    # Simple interpolation - use the latency at the recall closest to target
                    target_idx = np.argmin(np.abs(recalls - target_recall))
                    interpolated_latency = latencies[target_idx]
                    latency_data.append(interpolated_latency)
                else:
                    latency_data.append(np.nan)
            else:
                latency_data.append(np.nan)

        # Create summary row for this baseline with one data point per percentile
        summary_rows = []
        for i, percentile in enumerate(percentiles):
            pct_data = df_copy[df_copy["percentile"] == percentile]
            if not pct_data.empty:
                selectivity = pct_data["selectivity"].iloc[0]
                summary_rows.append(
                    {
                        "baseline": baseline_name,
                        "short_name": SHORT_NAMES.get(baseline_name, baseline_name),
                        "percentile": percentile,
                        "selectivity": selectivity,
                        "memory_gb": index_overhead_gb,
                        "build_time_sec": build_time_sec,
                        "latency_ms": (
                            latency_data[i] if i < len(latency_data) else np.nan
                        ),
                    }
                )

        if summary_rows:
            processed_results[baseline_name] = pd.DataFrame(summary_rows)

    if not processed_results:
        raise ValueError("No valid processed results found")

    # Aggregate results by percentile
    per_pct_results = {}
    for percentile in percentiles:
        res_list = []
        for baseline_name, df in processed_results.items():
            pct_data = df[df["percentile"] == percentile]
            if not pct_data.empty:
                res_list.append(pct_data)

        if res_list:
            res = pd.concat(res_list, ignore_index=True)
            per_pct_results[percentile] = res

    if not per_pct_results:
        raise ValueError("No data found for any percentile")

    # Get selectivity values for titles
    pct_to_sel = {}
    for percentile, res in per_pct_results.items():
        if not res.empty:
            pct_to_sel[percentile] = res["selectivity"].iloc[0]

    # Set up the plot
    plt.rcParams.update({"font.size": font_size})
    fig_width = figsize_per_subplot[0] * len(percentiles)
    fig_height = figsize_per_subplot[1]
    fig, axes = plt.subplots(1, len(percentiles), figsize=(fig_width, fig_height))

    # Handle case where there's only one subplot
    if len(percentiles) == 1:
        axes = [axes]

    # Define the desired order of baselines for plotting
    desired_order = [
        "Per-Label HNSW",
        "Per-Label IVF",
        "Parlay IVF",
        "Filtered DiskANN",
        "Shared HNSW",
        "Shared IVF",
        "Pg-HNSW",
        "Pg-IVF",
        "ACORN-1",
        r"ACORN-$\gamma$",
        "Curator",
    ]

    # Use only baselines that are available, in the desired order
    baseline_names = [name for name in desired_order if name in processed_results]

    # Add any additional baselines that weren't in the desired order
    additional_baselines = []
    for name in processed_results.keys():
        if name not in baseline_names:
            additional_baselines.append(name)
            baseline_names.append(name)

    # Use fixed legend range for consistent sizing across plots
    fixed_legend_latencies = [1, 10, 100]  # 1ms, 10ms, 100ms
    fixed_legend_labels = ["1ms", "10ms", "100ms"]

    legend_min_latency = min(fixed_legend_latencies)
    legend_max_latency = max(fixed_legend_latencies)

    # Scale sizes from 30 to 300 based on latency (log scale)
    min_size, max_size = 30, 300

    # Plot each percentile
    for i, (ax, percentile) in enumerate(zip(axes, percentiles)):
        if percentile not in per_pct_results:
            ax.set_title(f"No data for {int(percentile * 100)}p")
            continue

        per_pct_df = per_pct_results[percentile]

        # Remove rows with NaN latency
        per_pct_df = per_pct_df.dropna(subset=["latency_ms"])

        if per_pct_df.empty:
            ax.set_title(f"No valid data for {int(percentile * 100)}p")
            continue

        # Calculate point sizes based on latency (log scale)
        sizes = []
        for latency_ms in per_pct_df["latency_ms"]:
            # Log scale for better visualization using legend range
            # Clamp values to legend range to avoid extreme sizes
            clamped_latency = max(
                legend_min_latency, min(legend_max_latency, latency_ms)
            )
            if legend_max_latency == legend_min_latency:
                norm_size = 0.5  # Middle size if all latencies are the same
            else:
                norm_size = (np.log(clamped_latency) - np.log(legend_min_latency)) / (
                    np.log(legend_max_latency) - np.log(legend_min_latency)
                )
            size = min_size + norm_size * (max_size - min_size)
            sizes.append(size)

        # Create scatter plot with build time on x-axis, memory on y-axis
        # Use consistent color mapping based on baseline names
        colors = [
            BASELINE_COLORS.get(baseline, "#000000")
            for baseline in per_pct_df["baseline"]
        ]

        scatter = ax.scatter(
            per_pct_df["build_time_sec"],
            per_pct_df["memory_gb"],
            s=sizes,
            alpha=0.7,
            c=colors,
        )

        # Add simple annotations (no config for now)
        for _, row in per_pct_df.iterrows():
            ax.annotate(
                row["short_name"],
                (row["build_time_sec"], row["memory_gb"]),
                xytext=(5, 2),
                textcoords="offset points",
                fontsize=font_size - 4,
                ha="left",
                va="bottom",
            )

        # Set scales and labels
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Build Time (s)" if i == len(axes) // 2 else "")
        ax.set_ylabel("Index Overhead (GB)" if i == 0 else "")

        # Set title with selectivity info
        if percentile in pct_to_sel:
            selectivity = pct_to_sel[percentile]
            display_percentile = 99 if percentile == 1.0 else int(percentile * 100)
            ax.set_title(f"{display_percentile}p Sel ({selectivity:.4f})")
        else:
            ax.set_title(f"{int(percentile * 100)}p")

        # Add grid
        ax.grid(True, alpha=0.3)

    # Set consistent axis limits across all subplots
    if len(axes) > 1:
        valid_axes = [ax for ax in axes if len(ax.collections) > 0]
        if valid_axes:
            global_xlim = (
                min(ax.get_xlim()[0] for ax in valid_axes if ax.get_xlim()[0] > 0),
                max(ax.get_xlim()[1] for ax in valid_axes),
            )
            global_ylim = (
                min(ax.get_ylim()[0] for ax in valid_axes if ax.get_ylim()[0] > 0),
                max(ax.get_ylim()[1] for ax in valid_axes),
            )

            for ax in valid_axes:
                ax.set_xlim(global_xlim)
                ax.set_ylim(global_ylim)

    # Create legend for latency (marker size)
    legend_sizes = []
    for latency in fixed_legend_latencies:
        if legend_max_latency == legend_min_latency:
            size = (min_size + max_size) / 2
        else:
            norm_size = (np.log(latency) - np.log(legend_min_latency)) / (
                np.log(legend_max_latency) - np.log(legend_min_latency)
            )
            size = min_size + norm_size * (max_size - min_size)
        legend_sizes.append(size)

    # Create legend elements
    legend_elements = []
    for size, label in zip(legend_sizes, fixed_legend_labels):
        legend_elements.append(
            plt.scatter([], [], s=size, c="gray", alpha=0.7, label=label)
        )

    # Add legend
    legend = fig.legend(
        handles=legend_elements,
        title=f"Latency (R={target_recall})",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=len(fixed_legend_labels),
        frameon=True,
        fontsize=font_size - 2,
        title_fontsize=font_size - 1,
    )

    fig.tight_layout()

    # Save the figure
    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")
    print(f"Plot saved successfully!")

    # Print summary
    print(f"\nSummary (Dataset: {dataset_name}, Target Recall: {target_recall}):")
    for percentile in percentiles:
        if percentile in per_pct_results:
            print(f"\n{int(percentile * 100)}p Selectivity:")
            for _, row in per_pct_results[percentile].iterrows():
                print(
                    f"  {row['baseline']}: {row['build_time_sec']:.2f}s build, "
                    f"{row['memory_gb']:.2f}GB overhead, {row['latency_ms']:.2f}ms latency"
                )


if __name__ == "__main__":
    fire.Fire()
