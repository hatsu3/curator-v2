"""
Update performance plotting script using evaluation results.
"""

import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# CURATOR DELETION PERFORMANCE DATA PATHS
# NOTE: Both delete_lat_avg and revoke_access_lat_avg are loaded from previous results
# because vector deletion is currently disabled in the unified_curator branch
CURATOR_DELETION_PATHS = {
    "YFCC-10M": "output/overall_results/curator/yfcc100m-10m_test0.001/results/nlist32_sl256.csv",
    "arXiv": "output/overall_results/curator/arxiv-large-10_test0.005/results/nlist32_sl256.csv",
}

# SHARED IVF MISSING RESULTS FALLBACK
# NOTE: Shared IVF results are missing for YFCC100M in the new results, so we use old results
# This should be updated when new results are available
SHARED_IVF_FALLBACK_PATHS = {
    "YFCC-10M": "output/overall_results/shared_ivf/yfcc100m-10m_test0.001/results/nlist32768_nprobe32.csv",
}


def load_algorithm_update_performance(
    output_dir: str,
    algorithm: str,
    dataset_key: str,
    test_size: float,
    dataset_display_name: str = "",
) -> dict | None:
    """Load update performance metrics for a specific algorithm and dataset.

    Args:
        output_dir: Base output directory containing experiment results
        algorithm: Algorithm name (e.g., 'curator', 'shared_hnsw', etc.)
        dataset_key: Dataset key (e.g., 'yfcc100m-10m', 'arxiv-large-10')
        test_size: Test size fraction
        dataset_display_name: Display name for dataset (for fallback handling)

    Returns:
        Dictionary with update performance metrics, or None if not found
    """
    base_dir = Path(output_dir) / algorithm / f"{dataset_key}_test{test_size}"
    results_path = base_dir / "results.csv"

    # Special handling for Shared IVF on YFCC100M - use old results if new ones are missing
    if (
        algorithm == "shared_ivf"
        and dataset_display_name == "YFCC-10M"
        and dataset_display_name in SHARED_IVF_FALLBACK_PATHS
    ):
        fallback_path = SHARED_IVF_FALLBACK_PATHS[dataset_display_name]
        print(
            f"Using fallback results for Shared IVF on {dataset_display_name}: {fallback_path}"
        )
        results_path = Path(fallback_path)

    # Special handling for pgvector baselines: read insert_non_durable.csv
    if algorithm in {"pgvector_hnsw", "pgvector_ivf"}:
        insert_path = base_dir / "insert_non_durable.csv"
        if not insert_path.exists():
            print(f"Warning: pgvector insert file not found: {insert_path}")
            return None
        try:
            df = pd.read_csv(insert_path)
            if len(df) == 0:
                print(f"Warning: Empty pgvector insert file: {insert_path}")
                return None
            row = df.iloc[0]
            result = {
                "insert_lat_avg": float(row["insert_lat_avg"]) * 1000 if not pd.isna(row.get("insert_lat_avg")) else None,
                "insert_lat_p99": float(row["insert_lat_p99"]) * 1000 if not pd.isna(row.get("insert_lat_p99")) else None,
                "access_grant_lat_avg": None,
                "delete_lat_avg": None,
                "revoke_access_lat_avg": None,
            }
            print("✓ Loaded pgvector update performance from insert_non_durable.csv")
            return result
        except Exception as e:
            print(f"Error reading pgvector insert file: {e}")
            return None

    if not results_path.exists():
        print(f"Warning: Results file not found: {results_path}")
        return None

    try:
        # Read the CSV file
        df = pd.read_csv(results_path)

        if len(df) == 0:
            print(f"Warning: Empty results file: {results_path}")
            return None

        # Get data from first row (or best performing row if multiple)
        row = df.iloc[0]

        # Extract update performance metrics
        result = {}

        # Vector insertion latency (required for most algorithms)
        if "insert_lat_avg" in df.columns and not pd.isna(row["insert_lat_avg"]):
            result["insert_lat_avg"] = (
                float(row["insert_lat_avg"]) * 1000
            )  # Convert to ms
        else:
            result["insert_lat_avg"] = None

        # Label insertion latency (required for most algorithms)
        if "access_grant_lat_avg" in df.columns and not pd.isna(
            row["access_grant_lat_avg"]
        ):
            result["access_grant_lat_avg"] = (
                float(row["access_grant_lat_avg"]) * 1000
            )  # Convert to ms
        else:
            result["access_grant_lat_avg"] = None

        # Vector deletion latency (Curator only)
        # NOTE: Vector deletion is currently disabled in the unified_curator branch
        # due to IndexFlat storage not supporting vector removal. Using hardcoded
        # values from previous experiments until this is reimplemented.
        if "delete_lat_avg" in df.columns and not pd.isna(row["delete_lat_avg"]):
            result["delete_lat_avg"] = (
                float(row["delete_lat_avg"]) * 1000
            )  # Convert to ms
        else:
            result["delete_lat_avg"] = None

        # Label deletion latency (Curator only)
        if "revoke_access_lat_avg" in df.columns and not pd.isna(
            row["revoke_access_lat_avg"]
        ):
            result["revoke_access_lat_avg"] = (
                float(row["revoke_access_lat_avg"]) * 1000
            )  # Convert to ms
        else:
            result["revoke_access_lat_avg"] = None

        print(f"✓ Loaded {algorithm} update performance:")
        for metric, value in result.items():
            if value is not None:
                print(f"    {metric}: {value:.3f} ms")
            else:
                print(f"    {metric}: N/A")

        return result

    except Exception as e:
        print(f"Error reading {results_path}: {e}")
        return None


def load_update_performance_results(
    output_dir: str = "output/overall_results2",
    datasets: list[str] = ["yfcc100m", "arxiv"],
    optimal_params_file: str = "benchmark/overall_results/optimal_baseline_params.json",
) -> list[dict]:
    """Load update performance results from experiment outputs.

    Args:
        output_dir: Directory containing experiment results
        datasets: List of dataset names to process
        optimal_params_file: Path to optimal parameters JSON file

    Returns:
        List of result dictionaries with algorithm names and update performance metrics
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
            "update_performance": {},
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
                result["update_performance"][display_name] = None
                continue

            if dataset not in optimal_params[algorithm_json_name]:
                print(f"Info: {algorithm_json_name} not available for {dataset}")
                result["update_performance"][display_name] = None
                continue

            print(f"\nProcessing {algorithm_json_name} on {dataset}:")

            # Load the update performance from results
            update_perf = load_algorithm_update_performance(
                output_dir=output_dir,
                algorithm=dir_name,
                dataset_key=config["dataset_key"],
                test_size=config["test_size"],
                dataset_display_name=display_name,
            )

            result["update_performance"][display_name] = update_perf

            if update_perf is not None:
                print(f"✓ Loaded {algorithm_json_name} on {dataset}")
            else:
                print(f"✗ Failed to load {algorithm_json_name} on {dataset}")

        results.append(result)

        # LOAD DELETION PERFORMANCE FOR CURATOR FROM EXISTING RESULTS
    # NOTE: Both delete_lat_avg and revoke_access_lat_avg are loaded from previous results
    # because vector deletion is currently disabled in the unified_curator branch
    # due to IndexFlat storage not supporting individual vector removal.
    # Loading values from previous experiments until this is reimplemented.

    for result in results:
        if result["name"] == "Curator":
            for dataset_display, perf_data in result["update_performance"].items():
                if perf_data is not None and dataset_display in CURATOR_DELETION_PATHS:
                    deletion_file = CURATOR_DELETION_PATHS[dataset_display]

                    try:
                        if Path(deletion_file).exists():
                            deletion_df = pd.read_csv(deletion_file)
                            if len(deletion_df) > 0:
                                deletion_row = deletion_df.iloc[0]

                                # Load deletion latencies from the CSV
                                if (
                                    "delete_lat_avg" in deletion_df.columns
                                    and not pd.isna(deletion_row["delete_lat_avg"])
                                ):
                                    perf_data["delete_lat_avg"] = (
                                        float(deletion_row["delete_lat_avg"]) * 1000
                                    )  # Convert to ms

                                if (
                                    "revoke_access_lat_avg" in deletion_df.columns
                                    and not pd.isna(
                                        deletion_row["revoke_access_lat_avg"]
                                    )
                                ):
                                    perf_data["revoke_access_lat_avg"] = (
                                        float(deletion_row["revoke_access_lat_avg"])
                                        * 1000
                                    )  # Convert to ms

                                print(
                                    f"✓ Loaded deletion performance for Curator on {dataset_display} from {deletion_file}:"
                                )
                                if perf_data.get("delete_lat_avg") is not None:
                                    print(
                                        f"    delete_lat_avg: {perf_data['delete_lat_avg']:.3f} ms"
                                    )
                                if perf_data.get("revoke_access_lat_avg") is not None:
                                    print(
                                        f"    revoke_access_lat_avg: {perf_data['revoke_access_lat_avg']:.3f} ms"
                                    )
                            else:
                                print(
                                    f"Warning: Empty deletion results file: {deletion_file}"
                                )
                        else:
                            print(
                                f"Warning: Deletion results file not found: {deletion_file}"
                            )
                    except Exception as e:
                        print(
                            f"Error loading deletion performance from {deletion_file}: {e}"
                        )

    return results


def plot_update_results(
    output_dir: str = "output/overall_results2",
    datasets: list[str] = ["yfcc100m", "arxiv"],
    optimal_params_file: str = "benchmark/overall_results/optimal_baseline_params.json",
    output_path: str = "output/overall_results2/figs/update_perf.pdf",
):
    """Plot update performance comparison using real evaluation results.

    Args:
        output_dir: Directory containing experiment results
        datasets: List of dataset names to process
        optimal_params_file: Path to optimal parameters JSON file
        output_path: Path to save the output plot
    """
    print("Loading update performance results from experiment outputs...")
    profile_results = load_update_performance_results(
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
        "Pg-HNSW",
        "Pg-IVF",
        "S-HNSW",
        "S-IVF",
        "Curator",
    ]

    # Create DataFrame for plotting
    df_rows = []
    for res in profile_results:
        for dataset_display in dataset_display_names:
            if (
                dataset_display in res["update_performance"]
                and res["update_performance"][dataset_display] is not None
            ):
                perf_data = res["update_performance"][dataset_display]

                # Add row for each available metric
                if perf_data.get("insert_lat_avg") is not None:
                    df_rows.append(
                        {
                            "index_key": res["name"],
                            "dataset": dataset_display,
                            "insert_lat_avg": perf_data["insert_lat_avg"],
                            "access_grant_lat_avg": perf_data.get(
                                "access_grant_lat_avg"
                            ),
                            "delete_lat_avg": perf_data.get("delete_lat_avg"),
                            "revoke_access_lat_avg": perf_data.get(
                                "revoke_access_lat_avg"
                            ),
                        }
                    )

    if not df_rows:
        print("Error: No valid data found for plotting")
        return

    df = pd.DataFrame(df_rows)

    print(f"\nPlotting {len(df)} data points...")
    print(f"Datasets: {dataset_display_names}")

    # Only include algorithms that have at least one valid data point and are in the desired order
    available_algorithms = set(df["index_key"].unique())
    filtered_index_keys = [key for key in baseline_order if key in available_algorithms]

    print(f"Algorithms (ordered): {filtered_index_keys}")

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(2, 5)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :3])
    ax3 = fig.add_subplot(gs[1, 3:])

    # Plot 1: Label Insertion (access_grant_lat_avg)
    label_insertion_df = df[df["access_grant_lat_avg"].notna()].copy()
    assert isinstance(label_insertion_df, pd.DataFrame)
    if not label_insertion_df.empty:
        available_label_algorithms = [
            k for k in filtered_index_keys if k in set(label_insertion_df["index_key"])
        ]
        sns.barplot(
            data=label_insertion_df,
            x="index_key",
            y="access_grant_lat_avg",
            hue="dataset",
            order=available_label_algorithms,
            hue_order=dataset_display_names,
            ax=ax1,
        )
    ax1.set_title("Label Insertion")
    ax1.set_yscale("log")
    ax1.set_xlabel("")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    # Plot 2: Vector Insertion (insert_lat_avg) - exclude P-* and Pg-* algorithms
    vector_insertion_df = df[
        (~df["index_key"].str.startswith("P-"))
        & (~df["index_key"].str.startswith("Pg-"))
        & (df["insert_lat_avg"].notna())
    ].copy()
    assert isinstance(vector_insertion_df, pd.DataFrame)
    if not vector_insertion_df.empty:
        non_p_algorithms = [k for k in filtered_index_keys if not k.startswith("P-") and not k.startswith("Pg-")]
        available_non_p = [
            k for k in non_p_algorithms if k in set(vector_insertion_df["index_key"])
        ]

        sns.barplot(
            data=vector_insertion_df,
            x="index_key",
            y="insert_lat_avg",
            hue="dataset",
            order=available_non_p,
            hue_order=dataset_display_names,
            ax=ax2,
        )
    ax2.set_title("Vector Insertion")
    ax2.set_yscale("log")
    ax2.set_xlabel("")
    ax2.set_ylabel("Latency (ms)")
    ax2.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    # Plot 3: Deletion (Curator only)
    curator_df = df[df["index_key"] == "Curator"]
    if not curator_df.empty:
        # Create deletion DataFrame
        delete_rows = []
        for _, row in curator_df.iterrows():
            if row["revoke_access_lat_avg"] is not None:
                delete_rows.append(
                    {
                        "dataset": row["dataset"],
                        "index_key": row["index_key"],
                        "op": "Label",
                        "latency": row["revoke_access_lat_avg"],
                    }
                )
            if row["delete_lat_avg"] is not None:
                delete_rows.append(
                    {
                        "dataset": row["dataset"],
                        "index_key": row["index_key"],
                        "op": "Vector",
                        "latency": row["delete_lat_avg"],
                    }
                )

        if delete_rows:
            delete_df = pd.DataFrame(delete_rows)
            sns.barplot(
                data=delete_df,
                x="op",
                y="latency",
                hue="dataset",
                hue_order=dataset_display_names,
                ax=ax3,
            )

    ax3.set_title("Deletion (Curator)")
    ax3.set_yscale("log")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    # Add red crosses for missing data
    def add_red_crosses_for_missing_data(
        ax, available_data, algorithms, datasets, metric_name
    ):
        """Add red crosses for missing data points in a subplot."""
        if not available_data:
            return

        # Get bar width for positioning
        bar_width = 0.8 / len(datasets)

        # Use minimum y value for cross position
        min_y = min(row[1] for row in available_data)
        cross_y = min_y

        for i, algorithm in enumerate(algorithms):
            for j, dataset in enumerate(datasets):
                # Check if this combination has data
                has_data = any(
                    row[0] == algorithm and row[2] == dataset for row in available_data
                )

                if not has_data:
                    # Calculate x position for this bar
                    x_pos = i + (j - (len(datasets) - 1) / 2) * bar_width

                    # Add red cross
                    ax.plot(x_pos, cross_y, "rx", markersize=10, markeredgewidth=3)
                    print(
                        f"Adding red cross for missing {metric_name}: {algorithm} on {dataset} at ({x_pos:.2f}, {cross_y:.4f})"
                    )

    # Add red crosses for Plot 1: Label Insertion
    if not label_insertion_df.empty:
        label_data = [
            (row["index_key"], row["access_grant_lat_avg"], row["dataset"])
            for _, row in label_insertion_df.iterrows()
        ]
        add_red_crosses_for_missing_data(
            ax1,
            label_data,
            available_label_algorithms,
            dataset_display_names,
            "label insertion",
        )

    # Add red crosses for Plot 2: Vector Insertion (non-P and non-Pg algorithms only)
    if not vector_insertion_df.empty:
        vector_data = [
            (row["index_key"], row["insert_lat_avg"], row["dataset"])
            for _, row in vector_insertion_df.iterrows()
        ]
        add_red_crosses_for_missing_data(
            ax2,
            vector_data,
            available_non_p,
            dataset_display_names,
            "vector insertion",
        )

    # Set consistent y-axis limits
    ylim_min = float("inf")
    ylim_max = 0
    for ax in [ax1, ax2, ax3]:
        if ax.get_ylim()[0] > 0:  # Only consider valid limits
            ylim_min = min(ylim_min, ax.get_ylim()[0])
            ylim_max = max(ylim_max, ax.get_ylim()[1])

    if ylim_min != float("inf"):
        for ax in [ax1, ax2, ax3]:
            ax.set_ylim(ylim_min, ylim_max)

    # Handle legends
    if ax1.get_legend():
        ax1.get_legend().set_title("")
    if ax2.get_legend():
        ax2.get_legend().remove()
    if ax3.get_legend():
        ax3.get_legend().remove()

    fig.tight_layout()

    print(f"\nSaving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)

    # Print summary
    print("\nUpdate performance summary:")
    for _, row in df.iterrows():
        print(f"  {row['index_key']} ({row['dataset']}):")
        if row["access_grant_lat_avg"] is not None:
            print(f"    Label insertion: {row['access_grant_lat_avg']:.3f} ms")
        if row["insert_lat_avg"] is not None:
            print(f"    Vector insertion: {row['insert_lat_avg']:.3f} ms")
        if row["delete_lat_avg"] is not None:
            print(f"    Vector deletion: {row['delete_lat_avg']:.3f} ms")
        if row["revoke_access_lat_avg"] is not None:
            print(f"    Label deletion: {row['revoke_access_lat_avg']:.3f} ms")


if __name__ == "__main__":
    fire.Fire()
