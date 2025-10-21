"""
Memory footprint plotting script using evaluation results.
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_algorithm_memory_usage(
    output_dir: str,
    algorithm: str,
    dataset_key: str,
    test_size: float,
) -> float | None:
    """Load memory usage (index_size_kb) for a specific algorithm and dataset.

    Args:
        output_dir: Base output directory containing experiment results
        algorithm: Algorithm name (e.g., 'curator', 'shared_hnsw', etc.)
        dataset_key: Dataset key (e.g., 'yfcc100m-10m', 'arxiv-large-10')
        test_size: Test size fraction

    Returns:
        Memory usage in KB, or None if not found
    """
    results_path = (
        Path(output_dir) / algorithm / f"{dataset_key}_test{test_size}" / "results.csv"
    )

    if not results_path.exists():
        print(f"Warning: Results file not found: {results_path}")
        return None

    try:
        # Read the CSV file and get the index_size_kb from the first row
        # (all rows should have the same index size for a given configuration)
        df = pd.read_csv(results_path)

        if "index_size_kb" not in df.columns:
            print(f"Warning: 'index_size_kb' column not found in {results_path}")
            print(f"Available columns: {list(df.columns)}")
            return None

        # Get the index size from the first row (should be consistent across all rows)
        index_size_kb = df["index_size_kb"].iloc[0]

        if pd.isna(index_size_kb):
            print(f"Warning: index_size_kb is NaN in {results_path}")
            return None

        return float(index_size_kb)

    except Exception as e:
        print(f"Error reading {results_path}: {e}")
        return None


def load_memory_footprint_results(
    output_dir: str = "output/overall_results2",
    datasets: list[str] = ["yfcc100m", "arxiv"],
    optimal_params_file: str = "benchmark/overall_results/optimal_baseline_params.json",
) -> list[dict]:
    """Load memory footprint results from experiment outputs.

    Args:
        output_dir: Directory containing experiment results
        datasets: List of dataset names to process
        optimal_params_file: Path to optimal parameters JSON file

    Returns:
        List of result dictionaries with algorithm names and memory usage
    """
    # Load optimal parameters to get the mapping of algorithm names
    with open(optimal_params_file, "r") as f:
        optimal_params = json.load(f)

    # Map dataset names to dataset keys and test sizes
    dataset_config = {
        "yfcc100m": {
            "dataset_key": "yfcc100m-10m",
            "test_size": 0.001,
            "display_name": "YFCC-10M",
        },
        "arxiv": {
            "dataset_key": "arxiv-large-10",
            "test_size": 0.005,
            "display_name": "arXiv",
        },
    }

    # Map algorithm names from JSON to directory names
    algorithm_mapping = {
        "Curator": "curator",
        "Per-Label HNSW": "per_label_hnsw",
        "Per-Label IVF": "per_label_ivf",
        "Shared HNSW": "shared_hnsw",
        "Shared IVF": "shared_ivf",
        "Parlay IVF": "parlay_ivf",
        "Filtered DiskANN": "filtered_diskann",
        "ACORN-1": "acorn_1",
        "ACORN-gamma": "acorn_gamma",
    }

    # Map to display names for plotting
    display_name_mapping = {
        "Curator": "Curator",
        "Per-Label HNSW": "P-HNSW",
        "Per-Label IVF": "P-IVF",
        "Shared HNSW": "S-HNSW",
        "Shared IVF": "S-IVF",
        "Parlay IVF": "Parlay",
        "Filtered DiskANN": "DiskANN",
        "ACORN-1": "ACORN-1",
        "ACORN-gamma": r"ACORN-$\gamma$",
    }

    results = []

    for algorithm_json_name, dir_name in algorithm_mapping.items():
        result = {
            "name": display_name_mapping.get(algorithm_json_name, algorithm_json_name),
            "index_size_kb": {},
        }

        for dataset in datasets:
            if dataset not in dataset_config:
                print(f"Warning: Unknown dataset {dataset}")
                continue

            config = dataset_config[dataset]
            display_name = config["display_name"]

            # Check if this algorithm has results for this dataset in optimal params
            if algorithm_json_name not in optimal_params:
                print(f"Warning: {algorithm_json_name} not found in optimal params")
                result["index_size_kb"][display_name] = None
                continue

            if dataset not in optimal_params[algorithm_json_name]:
                print(f"Info: {algorithm_json_name} not available for {dataset}")
                result["index_size_kb"][display_name] = None
                continue

            # Load the memory usage from results
            memory_usage = load_algorithm_memory_usage(
                output_dir=output_dir,
                algorithm=dir_name,
                dataset_key=config["dataset_key"],
                test_size=config["test_size"],
            )

            result["index_size_kb"][display_name] = memory_usage

            if memory_usage is not None:
                print(
                    f"Loaded {algorithm_json_name} on {dataset}: {memory_usage/1024/1024:.2f} GB"
                )
            else:
                print(f"Failed to load {algorithm_json_name} on {dataset}")

        results.append(result)

    return results


def plot_memory_footprint(
    output_dir: str = "output/overall_results2",
    datasets: list[str] = ["yfcc100m", "arxiv"],
    optimal_params_file: str = "benchmark/overall_results/optimal_baseline_params.json",
    output_path: str = "output/overall_results2/figs/memory_footprint.pdf",
    subtract_vectors: bool = True,
):
    """Plot memory footprint comparison using real evaluation results.

    Args:
        output_dir: Directory containing experiment results
        datasets: List of dataset names to process
        optimal_params_file: Path to optimal parameters JSON file
        output_path: Path to save the output plot
        subtract_vectors: If True, subtract raw vector storage size from memory footprint
    """
    print("Loading memory footprint results from experiment outputs...")
    profile_results = load_memory_footprint_results(
        output_dir=output_dir,
        datasets=datasets,
        optimal_params_file=optimal_params_file,
    )

    # Convert dataset names to display names
    dataset_display_mapping = {"yfcc100m": "YFCC-10M", "arxiv": "arXiv"}
    dataset_display_names = [dataset_display_mapping[d] for d in datasets]

    # Dataset specifications for vector size calculation
    dataset_specs = {
        "YFCC-10M": {
            "num_vectors": 10_000_000 * (1 - 0.001),  # 10M vectors
            "dimensions": 192,  # dimensions
            "bytes_per_float": 4,  # float32
        },
        "arXiv": {
            "num_vectors": 2_000_000 * (1 - 0.005),  # 2M vectors
            "dimensions": 384,  # dimensions
            "bytes_per_float": 4,  # float32
        },
    }

    def calculate_vector_size_kb(dataset_display: str) -> float:
        """Calculate raw vector storage size in KB for a dataset."""
        if dataset_display not in dataset_specs:
            return 0.0

        spec = dataset_specs[dataset_display]
        size_bytes = spec["num_vectors"] * spec["dimensions"] * spec["bytes_per_float"]
        return size_bytes / 1024  # Convert to KB

    # Create DataFrame for plotting
    df_rows = []
    for res in profile_results:
        for dataset_display in dataset_display_names:
            if (
                dataset_display in res["index_size_kb"]
                and res["index_size_kb"][dataset_display] is not None
            ):
                raw_size_kb = res["index_size_kb"][dataset_display]

                # Optionally subtract vector storage size
                if subtract_vectors:
                    vector_size_kb = calculate_vector_size_kb(dataset_display)
                    adjusted_size_kb = max(0, raw_size_kb - vector_size_kb)
                    if vector_size_kb > 0:
                        print(
                            f"  {res['name']} ({dataset_display}): {raw_size_kb/1024/1024:.2f} GB raw -> {adjusted_size_kb/1024/1024:.2f} GB (vectors: {vector_size_kb/1024/1024:.2f} GB)"
                        )
                else:
                    adjusted_size_kb = raw_size_kb

                df_rows.append(
                    {
                        "index_key": res["name"],
                        "index_size_gb": adjusted_size_kb / 1024 / 1024,
                        "dataset": dataset_display,
                    }
                )

    if not df_rows:
        print("Error: No valid data found for plotting")
        return

    df = pd.DataFrame(df_rows)

    # Define the desired ordering of baselines (same as original script)
    baseline_order = [
        "P-HNSW",
        "P-IVF",
        "Parlay",
        "DiskANN",
        r"ACORN-$\gamma$",
        "ACORN-1",
        "S-HNSW",
        "S-IVF",
        "Curator",
    ]

    print(f"Plotting {len(df)} data points...")
    print(f"Datasets: {dataset_display_names}")

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 3))

    # Only include algorithms that have at least one valid data point and are in the desired order
    available_algorithms = set(df["index_key"].unique())
    filtered_index_keys = [key for key in baseline_order if key in available_algorithms]

    print(f"Algorithms (ordered): {filtered_index_keys}")

    sns.barplot(
        data=df,
        x="index_key",
        y="index_size_gb",
        hue="dataset",
        order=filtered_index_keys,
        hue_order=dataset_display_names,
        ax=ax,
    )

    # Add red crosses for missing data
    # Create a set of available (algorithm, dataset) combinations
    available_data = set((row["index_key"], row["dataset"]) for _, row in df.iterrows())

    # Get bar positions and widths
    bar_width = 0.8 / len(
        dataset_display_names
    )  # Total width divided by number of datasets

    # Place crosses at small positive value above zero
    max_y = df["index_size_gb"].max() if not df.empty else 1.0
    cross_y = 0.04 * max_y  # Place crosses at 4% of max value

    for i, algorithm in enumerate(filtered_index_keys):
        for j, dataset in enumerate(dataset_display_names):
            if (algorithm, dataset) not in available_data:
                # Calculate x position for this bar
                x_pos = i + (j - (len(dataset_display_names) - 1) / 2) * bar_width

                # Add red cross
                ax.plot(x_pos, cross_y, "rx", markersize=10, markeredgewidth=3)
                print(
                    f"Adding red cross for missing data: {algorithm} on {dataset} at ({x_pos:.2f}, {cross_y:.4f})"
                )

    # ax.set_yscale("log")
    ax.set_xlabel("")
    ylabel = "Index Overhead (GB)" if subtract_vectors else "Memory Footprint (GB)"
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    ax.legend(title="", fontsize="small", ncol=len(dataset_display_names))

    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)

    # Print summary
    summary_title = (
        "Index overhead summary:" if subtract_vectors else "Memory footprint summary:"
    )
    print(f"\n{summary_title}")
    if subtract_vectors:
        print("(Raw vector storage has been subtracted)")
    for _, row in df.iterrows():
        print(f"  {row['index_key']} ({row['dataset']}): {row['index_size_gb']:.2f} GB")


if __name__ == "__main__":
    fire.Fire()
