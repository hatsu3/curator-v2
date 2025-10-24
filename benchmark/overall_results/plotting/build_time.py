"""
Build time plotting script using evaluation results.
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_algorithm_build_time(
    output_dir: str,
    algorithm: str,
    dataset_key: str,
    test_size: float,
    dataset_info: dict,
) -> float | None:
    """Load build time for a specific algorithm and dataset.

    Args:
        output_dir: Base output directory containing experiment results
        algorithm: Algorithm name (e.g., 'curator', 'shared_hnsw', etc.)
        dataset_key: Dataset key (e.g., 'yfcc100m-10m', 'arxiv-large-10')
        test_size: Test size fraction
        dataset_info: Dataset information with num_vecs and num_mds

    Returns:
        Build time in seconds, or None if not found
    """
    base_dir = Path(output_dir) / algorithm / f"{dataset_key}_test{test_size}"

    # Resolve thread count from parameters.json or env
    params_path = base_dir / "parameters.json"
    assert params_path.exists(), f"Parameters file not found: {params_path}"
    num_threads = 1
    with open(params_path, "r") as f:
        params = json.load(f)
        if (
            "experiment_config" in params
            and "num_threads" in params["experiment_config"]
        ):
            num_threads = params["experiment_config"]["num_threads"]
            print(f"  Found thread count: {num_threads}")
        else:
            print("  Thread count not in parameters; using default: 1")

    # Special handling for pgvector baselines: read build artifacts
    # We do not support incremental builds for pgvector baselines for now
    if algorithm in {"pgvector_hnsw", "pgvector_ivf"}:
        build_path = base_dir / "build.csv"
        df = pd.read_csv(build_path)
        assert len(df) > 0, f"Empty pgvector build file: {build_path}"
        row = df.iloc[0]
        assert (
            "build_time_seconds" in df.columns
        ), "build_time_seconds missing in pgvector build.csv"
        total_build_time = float(row["build_time_seconds"]) * num_threads
        print(
            f"  pgvector build_time_seconds: {float(row['build_time_seconds']):.2f} * {num_threads} threads = {total_build_time:.2f}s"
        )
        return total_build_time

    results_path = base_dir / "results.csv"
    assert results_path.exists(), f"Results file not found: {results_path}"

    try:
        # Read the CSV file
        df = pd.read_csv(results_path)

        if len(df) == 0:
            print(f"Warning: Empty results file: {results_path}")
            return None

        # Method 1: Try to use batch_insert_latency (for most baselines)
        if "batch_insert_latency" in df.columns:
            build_time = df["batch_insert_latency"].iloc[0]
            if not pd.isna(build_time):
                # batch_insert_latency uses multi-threading, so adjust for thread count
                adjusted_build_time = float(build_time) * num_threads
                print(
                    f"  Using batch_insert_latency: {build_time:.2f}s * {num_threads} threads = {adjusted_build_time:.2f}s"
                )

                # Also add training time if available (training uses multi-threading)
                row = df.iloc[0]
                train_time = 0.0
                if "train_latency" in df.columns and not pd.isna(row["train_latency"]):
                    raw_train_time = float(row["train_latency"])
                    train_time = (
                        raw_train_time * num_threads
                    )  # Training uses multi-threading
                    print(
                        f"  Adding train_latency: {raw_train_time:.2f}s * {num_threads} threads = {train_time:.2f}s"
                    )

                total_time = adjusted_build_time + train_time
                print(
                    f"  Total build time: {adjusted_build_time:.2f} + {train_time:.2f} = {total_time:.2f}s"
                )
                return total_time

        # Method 2: Compute from individual components (for curator-like algorithms)
        required_cols = ["insert_lat_avg", "access_grant_lat_avg"]
        if all(col in df.columns for col in required_cols):
            # Get data from first row
            row = df.iloc[0]

            # Start with training time if available (training uses multi-threading)
            train_time = 0.0
            if "train_latency" in df.columns and not pd.isna(row["train_latency"]):
                raw_train_time = float(row["train_latency"])
                train_time = (
                    raw_train_time * num_threads
                )  # Training uses multi-threading
                print(
                    f"  Found train_latency: {raw_train_time:.2f}s * {num_threads} threads = {train_time:.2f}s"
                )
            else:
                print(f"  No train_latency found")

            # Add insertion time: num_vecs * insert_lat_avg (single-threaded)
            insert_lat_avg = float(row["insert_lat_avg"])  # in seconds
            insert_time = dataset_info["num_vecs"] * insert_lat_avg
            print(
                f"  Insert time (single-threaded): {dataset_info['num_vecs']} * {insert_lat_avg:.6f} = {insert_time:.2f}s"
            )

            # Add access grant time: num_mds * access_grant_lat_avg (single-threaded)
            access_grant_lat_avg = float(row["access_grant_lat_avg"])  # in seconds
            access_grant_time = dataset_info["num_mds"] * access_grant_lat_avg
            print(
                f"  Access grant time (single-threaded): {dataset_info['num_mds']} * {access_grant_lat_avg:.6f} = {access_grant_time:.2f}s"
            )

            total_time = train_time + insert_time + access_grant_time
            print(
                f"  Total build time: {train_time:.2f} + {insert_time:.2f} + {access_grant_time:.2f} = {total_time:.2f}s"
            )

            return total_time

        print(f"Warning: No suitable build time columns found in {results_path}")
        print(f"Available columns: {list(df.columns)}")
        print(
            f"Expected columns: 'batch_insert_latency' OR ['insert_lat_avg', 'access_grant_lat_avg']"
        )
        return None

    except Exception as e:
        print(f"Error reading {results_path}: {e}")
        return None


def load_build_time_results(
    output_dir: str = "output/overall_results2",
    datasets: list[str] = ["yfcc100m", "arxiv"],
    optimal_params_file: str = "benchmark/overall_results/optimal_baseline_params.json",
) -> list[dict]:
    """Load build time results from experiment outputs.

    Args:
        output_dir: Directory containing experiment results
        datasets: List of dataset names to process
        optimal_params_file: Path to optimal parameters JSON file

    Returns:
        List of result dictionaries with algorithm names and build times
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
            "info": {
                "num_vecs": int(10_000_000 * (1 - 0.001)),  # 9,990,000
                "num_mds": 55293233,
            },
        },
        "arxiv": {
            "dataset_key": "arxiv-large-10",
            "test_size": 0.005,
            "display_name": "arXiv",
            "info": {
                "num_vecs": int(2_000_000 * (1 - 0.005)),  # 1,990,000
                "num_mds": 19755960,
            },
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
        "pgvector HNSW": "pgvector_hnsw",
        "pgvector IVF": "pgvector_ivf",
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
        "pgvector HNSW": "Pg-HNSW",
        "pgvector IVF": "Pg-IVF",
    }

    results = []

    for algorithm_json_name, dir_name in algorithm_mapping.items():
        result = {
            "name": display_name_mapping.get(algorithm_json_name, algorithm_json_name),
            "build_time_sec": {},
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
                result["build_time_sec"][display_name] = None
                continue

            if dataset not in optimal_params[algorithm_json_name]:
                print(f"Info: {algorithm_json_name} not available for {dataset}")
                result["build_time_sec"][display_name] = None
                continue

            print(f"\nProcessing {algorithm_json_name} on {dataset}:")

            # Load the build time from results
            build_time = load_algorithm_build_time(
                output_dir=output_dir,
                algorithm=dir_name,
                dataset_key=config["dataset_key"],
                test_size=config["test_size"],
                dataset_info=config["info"],
            )

            result["build_time_sec"][display_name] = build_time

            if build_time is not None:
                print(f"✓ Loaded {algorithm_json_name} on {dataset}: {build_time:.2f}s")
            else:
                print(f"✗ Failed to load {algorithm_json_name} on {dataset}")

        results.append(result)

    return results


def plot_construction_time(
    output_dir: str = "output/overall_results2",
    datasets: list[str] = ["yfcc100m", "arxiv"],
    optimal_params_file: str = "benchmark/overall_results/optimal_baseline_params.json",
    output_path: str = "output/overall_results2/figs/build_time.pdf",
):
    """Plot build time comparison using real evaluation results.

    Args:
        output_dir: Directory containing experiment results
        datasets: List of dataset names to process
        optimal_params_file: Path to optimal parameters JSON file
        output_path: Path to save the output plot
    """
    print("Loading build time results from experiment outputs...")
    profile_results = load_build_time_results(
        output_dir=output_dir,
        datasets=datasets,
        optimal_params_file=optimal_params_file,
    )

    # Convert dataset names to display names
    dataset_display_mapping = {"yfcc100m": "YFCC-10M", "arxiv": "arXiv"}
    dataset_display_names = [dataset_display_mapping[d] for d in datasets]

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
        "Pg-HNSW",
        "Pg-IVF",
        "Curator",
    ]

    # Create DataFrame for plotting
    df_rows = []
    for res in profile_results:
        for dataset_display in dataset_display_names:
            if (
                dataset_display in res["build_time_sec"]
                and res["build_time_sec"][dataset_display] is not None
            ):
                df_rows.append(
                    {
                        "index_key": res["name"],
                        "construction_time": res["build_time_sec"][dataset_display],
                        "dataset": dataset_display,
                    }
                )

    if not df_rows:
        print("Error: No valid data found for plotting")
        return

    df = pd.DataFrame(df_rows)

    print(f"\nPlotting {len(df)} data points...")
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
        y="construction_time",
        hue="dataset",
        hue_order=dataset_display_names,
        order=filtered_index_keys,
        ax=ax,
    )

    # Add red crosses for missing data
    # Create a set of available (algorithm, dataset) combinations
    available_data = set((row["index_key"], row["dataset"]) for _, row in df.iterrows())

    # Get bar positions and widths
    bar_width = 0.8 / len(
        dataset_display_names
    )  # Total width divided by number of datasets

    # Use minimum y value across all bars for cross position
    cross_y = df["construction_time"].min() if not df.empty else 1.0

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

    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Construction Time (s)")
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    ax.legend(title="", fontsize="small", ncol=len(dataset_display_names))

    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    fig.tight_layout()

    print(f"\nSaving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)

    # Print summary
    print("\nBuild time summary:")
    for _, row in df.iterrows():
        print(
            f"  {row['index_key']} ({row['dataset']}): {row['construction_time']:.2f} s"
        )


if __name__ == "__main__":
    fire.Fire()
