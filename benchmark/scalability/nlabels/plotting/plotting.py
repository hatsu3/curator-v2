from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import parse
import seaborn as sns
from matplotlib.axes import Axes

from benchmark.scalability.nlabels.data.dataset import ScalabilityDataset


def precompute_n_queries(
    n_labels_max: int = 10000,
    n_labels_n_steps: int = 10,
    dataset_cache_dir: str = "data/scalability/random_yfcc100m",
    seed: int = 42,
    output_path: str = "benchmark/scalability/num_queries.csv",
):
    n_labels_min = n_labels_max // (2 ** (n_labels_n_steps - 1))
    n_labels_space = [n_labels_min * (2**i) for i in range(n_labels_n_steps)]
    dataset = ScalabilityDataset.load(dataset_cache_dir)

    results = list()
    for n_labels in n_labels_space:
        split = dataset.get_random_split(n_labels, seed=seed)
        n_queries = sum(len(access_list) for access_list in split.test_mds)
        results.append({"n_labels": n_labels, "n_queries": n_queries})

    print(f"Saving results to {output_path} ...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


def load_all_results(
    output_dir: str = "output/scalability/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    n_queries_path: str = "benchmark/scalability/num_queries.csv",
):
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    if not Path(n_queries_path).exists():
        raise ValueError(
            f"{n_queries_path} does not exist. Call precompute_n_queries first."
        )

    n_queries = pd.read_csv(n_queries_path).to_dict(orient="records")
    n_queries = {result["n_labels"]: result["n_queries"] for result in n_queries}

    all_results = list()
    for csv_file in results_dir.glob("*.csv"):
        parse_res = parse.parse("n_labels{n_labels}_{}", csv_file.stem)
        assert isinstance(parse_res, parse.Result), f"Failed to parse {csv_file.stem}"

        n_labels = int(parse_res["n_labels"])
        df = pd.read_csv(csv_file).assign(n_labels=n_labels)

        if "batch_query_latency" in df.columns:
            df["query_lat_avg"] = df["batch_query_latency"] / n_queries[n_labels]

        all_results.append(df)

    all_results_df = pd.concat(all_results)
    all_results_df = all_results_df.reset_index(drop=True)
    return all_results_df


def load_nvec_results(results_dir: str | Path) -> pd.DataFrame:
    return pd.concat(
        [pd.read_csv(csv_path) for csv_path in Path(results_dir).glob("*.csv")]
    )


def select_best_config(
    df: pd.DataFrame, recall_threshold: float = 0.88
) -> pd.DataFrame:
    """For each dataset size, select the config that achieves >= recall_threshold with lowest latency"""

    def select_best_config_for_group(group: pd.DataFrame) -> pd.DataFrame:
        return (
            group.loc[group["recall_at_k"] >= recall_threshold]
            .sort_values("query_lat_avg")
            .iloc[:1]
        )

    return (
        df.groupby("subset_size")
        .apply(select_best_config_for_group)
        .reset_index(drop=True)
    )


def select_best_result(
    output_dir: str = "output/scalability/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    min_recall: float | None = 0.9,
    max_latency: float | None = 1e-3,
    n_queries_path: str = "benchmark/scalability/num_queries.csv",
):
    all_results_df = load_all_results(
        output_dir=output_dir,
        dataset_key=dataset_key,
        test_size=test_size,
        n_queries_path=n_queries_path,
    )
    if "recall_at_k" in all_results_df.columns and min_recall is not None:
        all_results_df = all_results_df[all_results_df["recall_at_k"] >= min_recall]
    if "query_lat_avg" in all_results_df.columns and max_latency is not None:
        all_results_df = all_results_df[all_results_df["query_lat_avg"] <= max_latency]
    all_results_df = all_results_df.reset_index(drop=True)

    best_results_df = (
        all_results_df.groupby("n_labels")
        .apply(lambda group: group.loc[group["index_size_kb"].idxmin()])
        .reset_index(drop=True)
    )

    return best_results_df


def load_acorn_csv(
    csv_path: str | None,
    index_type: str,
    description: str = "",
) -> pd.DataFrame | None:
    """Load ACORN profiling results from CSV.

    Args:
        csv_path: Path to CSV file with ACORN profiling data, or None to skip
        index_type: Index type to assign (e.g., "acorn-1", "acorn-gamma")
        description: Optional description for log messages (e.g., "nlabels", "nvecs")

    Returns:
        DataFrame with profiling results and index_type column, or None if csv_path is None
    """
    if csv_path is None:
        return None

    assert Path(csv_path).exists(), f"{index_type} CSV not found at {csv_path}"
    df = pd.read_csv(csv_path).assign(index_type=index_type)
    desc_suffix = f" {description}" if description else ""
    print(f"Loaded {index_type}{desc_suffix} data from {csv_path}")
    return df


def plot_memory_vs_nlabel_at_ax(
    ax: Axes,
    index_keys: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "parlay_ivf_graph",
        "filtered_diskann",
        "shared_hnsw",
        "curator",
        "acorn-1",
        "acorn-gamma",
    ],
    output_dir: str = "output/scalability",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    n_queries_path: str = "benchmark/scalability/num_queries.csv",
    min_recall: float | None = 0.9,
    max_latency: float | None = 2e-3,
    color_map: dict[str, str] | str = "tab10",
    marker_map: dict[str, str] | bool = True,
    marker_size: int = 6,
    line_width: float = 1,
    acorn_1_nlabels_csv: str | None = None,
    acorn_gamma_nlabels_csv: str | None = None,
):
    """Plot memory vs number of labels at the given axis.

    Args:
        ax: Matplotlib axis to plot on
        index_keys: List of index types to include in the plot
        output_dir: Base output directory containing scalability results
        dataset_key: Dataset identifier (e.g., "yfcc100m")
        test_size: Test size fraction
        n_queries_path: Path to CSV with number of queries per label count
        min_recall: Minimum recall threshold for config selection
        max_latency: Maximum latency threshold for config selection
        color_map: Color mapping for different index types
        marker_map: Marker mapping for different index types
        marker_size: Size of markers in plot
        line_width: Width of plot lines
        acorn_1_nlabels_csv: Path to ACORN-1 profiling CSV. If not provided, ACORN-1 will be excluded.
                             Expected format: CSV with columns 'n_labels' and 'index_size_gb'
                             Example: output/scalability/acorn_1/yfcc100m_test0.01/profiling_results.csv
                             Note: ACORN results must be manually collected (no automated benchmark scripts exist)
        acorn_gamma_nlabels_csv: Path to ACORN-gamma profiling CSV. If not provided, ACORN-gamma will be excluded.
                                 Expected format: CSV with columns 'n_labels' and 'index_size_gb'
                                 Example: output/scalability/acorn_gamma/yfcc100m_test0.01/profiling_results.csv
                                 Note: ACORN results must be manually collected (no automated benchmark scripts exist)
    """
    best_results: dict[str, pd.DataFrame] = dict()
    for index_key in index_keys:
        if index_key in ["acorn-1", "acorn-gamma"]:
            continue

        min_recall = None if index_key == "shared_hnsw" else min_recall
        max_latency = None if index_key == "shared_hnsw" else max_latency
        best_results[index_key] = select_best_result(
            output_dir=str(Path(output_dir) / index_key),
            dataset_key=dataset_key,
            test_size=test_size,
            min_recall=min_recall,
            max_latency=max_latency,
            n_queries_path=n_queries_path,
        )

    df = pd.concat(
        [
            results.assign(index_type=index_type)
            for index_type, results in best_results.items()
        ]
    )
    df["index_size_gb"] = df["index_size_kb"] / 1024 / 1024

    # Load ACORN data from CSV files if provided
    acorn_dfs = [
        acorn_df
        for acorn_df in [
            load_acorn_csv(acorn_1_nlabels_csv, "acorn-1"),
            load_acorn_csv(acorn_gamma_nlabels_csv, "acorn-gamma"),
        ]
        if acorn_df is not None
    ]
    if acorn_dfs:
        df = pd.concat([df, *acorn_dfs], ignore_index=True)

    # Filter index_keys to only include loaded data
    loaded_index_types = set(df["index_type"].unique())
    index_keys = [k for k in index_keys if k in loaded_index_types]

    sns.lineplot(
        data=df,
        x="n_labels",
        y="index_size_gb",
        hue="index_type",
        style="index_type",
        palette=color_map,
        markers=marker_map,
        dashes=False,
        hue_order=index_keys,
        errorbar=None,
        markersize=marker_size,
        linewidth=line_width,
        ax=ax,
    )
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    ax.set_xlabel("Number of Labels")
    ax.set_ylabel("Memory Usage (GB)")


def plot_memory_vs_dataset_size_at_ax(
    ax: Axes,
    results_dir: str,
    index_keys: list[str] = [
        "curator",
        "parlay_ivf",
        "filtered_diskann",
        "shared_hnsw",
        "acorn-gamma",
        "acorn-1",
    ],
    recall_threshold: float = 0.88,
    color_map: dict[str, str] | str = "tab10",
    marker_map: dict[str, str] | bool = True,
    marker_size: int = 6,
    line_width: float = 1,
    acorn_1_nvecs_csv: str | None = None,
    acorn_gamma_nvecs_csv: str | None = None,
):
    """Plot memory vs dataset size at the given axis.

    Args:
        ax: Matplotlib axis to plot on
        results_dir: Directory containing nvecs scalability results
        index_keys: List of index types to include in the plot
        recall_threshold: Recall threshold for config selection
        color_map: Color mapping for different index types
        marker_map: Marker mapping for different index types
        marker_size: Size of markers in plot
        line_width: Width of plot lines
        acorn_1_nvecs_csv: Path to ACORN-1 nvecs profiling CSV. If not provided, ACORN-1 will be excluded.
                           Expected format: CSV with columns 'subset_size' (in millions) and 'index_size_gb'
                           Example: output/scalability_nvec/acorn_1/yfcc100m-10m_test0.01/profiling_results.csv
                           Note: ACORN results must be manually collected (no automated benchmark scripts exist)
        acorn_gamma_nvecs_csv: Path to ACORN-gamma nvecs profiling CSV. If not provided, ACORN-gamma will be excluded.
                               Expected format: CSV with columns 'subset_size' (in millions) and 'index_size_gb'
                               Example: output/scalability_nvec/acorn_gamma/yfcc100m-10m_test0.01/profiling_results.csv
                               Note: ACORN results must be manually collected (no automated benchmark scripts exist)
    """
    index_dfs = []
    for index_key in index_keys:
        if index_key in ["acorn-1", "acorn-gamma"]:
            continue

        index_df = load_nvec_results(
            Path(results_dir) / index_key / "yfcc100m-10m_test0.01" / "results"
        )
        index_df = select_best_config(index_df, recall_threshold)
        index_dfs.append(index_df.assign(index_type=index_key))

    combined_df = pd.concat(index_dfs)
    combined_df["subset_size"] = combined_df["subset_size"] / 1e6
    combined_df["index_size_gb"] = combined_df["index_size_kb"] / 1024 / 1024

    combined_df.loc[len(combined_df)] = {
        "index_type": "filtered_diskann",
        "subset_size": 10,
        "index_size_gb": 26.9,
    }

    # Load ACORN data from CSV files if provided
    acorn_dfs = [
        acorn_df
        for acorn_df in [
            load_acorn_csv(acorn_1_nvecs_csv, "acorn-1", "nvecs"),
            load_acorn_csv(acorn_gamma_nvecs_csv, "acorn-gamma", "nvecs"),
        ]
        if acorn_df is not None
    ]
    if acorn_dfs:
        combined_df = pd.concat([combined_df, *acorn_dfs], ignore_index=True)

    # Filter index_keys to only include loaded data
    loaded_index_types = set(combined_df["index_type"].unique())
    index_keys = [k for k in index_keys if k in loaded_index_types]

    sns.lineplot(
        data=combined_df,
        x="subset_size",
        y="index_size_gb",
        hue="index_type",
        hue_order=index_keys,
        style="index_type",
        style_order=index_keys,
        palette=color_map,
        markers=marker_map,
        dashes=False,
        markersize=marker_size,
        linewidth=line_width,
        ax=ax,
    )
    ax.plot(
        combined_df["subset_size"],
        combined_df["subset_size"] * 1e6 * 192 * 4 / 1024 / 1024 / 1024,
        color=(0.5, 1, 0.5, 0.5),
        linestyle="-",
        linewidth=3,
        label="Vector Storage",
    )
    ax.set_xlabel("Number of Vectors (M)")
    ax.set_ylabel("Memory Usage (GB)")
    ax.grid(visible=True, which="major", axis="y", linestyle="-", alpha=0.6)
    ax.set_xticks(combined_df["subset_size"].unique())


def plot_memory_vs_nvec_nlabel_combined(
    output_path: str = "output/scalability_nvec/figs/resivion_memory_vs_nvec_nlabel_combined.pdf",
    nlabel_index_keys: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "parlay_ivf_graph",
        "filtered_diskann",
        "shared_hnsw",
        "acorn-1",
        "acorn-gamma",
        "curator",
    ],
    nlabel_index_keys_readable: list[str] = [
        "Per-Label HNSW",
        "Per-Label IVF",
        "Parlay IVF",
        "Filtered DiskANN",
        "Shared HNSW",
        "ACORN-1",
        r"ACORN-$\gamma$",
        "Curator",
    ],
    nlabel_output_dir: str = "output/scalability",
    nlabel_dataset_key: str = "yfcc100m",
    nlabel_test_size: float = 0.01,
    nlabel_n_queries_path: str = "benchmark/scalability/num_queries.csv",
    nlabel_min_recall: float | None = 0.9,
    nlabel_max_latency: float | None = 2e-3,
    acorn_1_nlabels_csv: str | None = None,
    acorn_gamma_nlabels_csv: str | None = None,
    nvec_results_dir: str = "output/scalability_nvec",
    nvec_index_keys: list[str] = [
        "curator",
        "parlay_ivf",
        "filtered_diskann",
        "shared_hnsw",
        "acorn-1",
        "acorn-gamma",
    ],
    nvec_recall_threshold: float = 0.88,
    acorn_1_nvecs_csv: str | None = None,
    acorn_gamma_nvecs_csv: str | None = None,
    marker_size: int = 6,
    line_width: float = 1.5,
):
    """Plot memory usage vs number of labels and dataset size (combined figure).

    Creates a two-subplot figure:
    - Left subplot: Memory vs number of labels
    - Right subplot: Memory vs dataset size (number of vectors)

    Args:
        output_path: Path to save the combined figure
        nlabel_index_keys: Index types to include in nlabels plot
        nlabel_index_keys_readable: Readable names for legend
        nlabel_output_dir: Base directory containing nlabels scalability results
        nlabel_dataset_key: Dataset name for nlabels experiments
        nlabel_test_size: Test set fraction for nlabels experiments
        nlabel_n_queries_path: Path to CSV with number of queries per nlabels setting
        nlabel_min_recall: Minimum recall threshold for nlabels filtering (None = no filter)
        nlabel_max_latency: Maximum latency threshold for nlabels filtering (None = no filter)
        acorn_1_nlabels_csv: Path to ACORN-1 nlabels profiling CSV. If not provided, ACORN-1
                            will be excluded from nlabels plot.
                            Expected format: CSV with columns 'n_labels' and 'index_size_gb'
                            Example: output/scalability/acorn_1/yfcc100m_test0.01/profiling_results.csv
        acorn_gamma_nlabels_csv: Path to ACORN-gamma nlabels profiling CSV. If not provided,
                                ACORN-gamma will be excluded from nlabels plot.
                                Expected format: CSV with columns 'n_labels' and 'index_size_gb'
                                Example: output/scalability/acorn_gamma/yfcc100m_test0.01/profiling_results.csv
        nvec_results_dir: Base directory containing nvecs scalability results
        nvec_index_keys: Index types to include in nvecs plot
        nvec_recall_threshold: Recall threshold for filtering nvecs results
        acorn_1_nvecs_csv: Path to ACORN-1 nvecs profiling CSV. If not provided, ACORN-1
                          will be excluded from nvecs plot.
                          Expected format: CSV with columns 'subset_size' and 'index_size_gb'
                          Example: output/scalability_nvec/acorn_1/yfcc100m-10m_test0.01/profiling_results.csv
        acorn_gamma_nvecs_csv: Path to ACORN-gamma nvecs profiling CSV. If not provided,
                              ACORN-gamma will be excluded from nvecs plot.
                              Expected format: CSV with columns 'subset_size' and 'index_size_gb'
                              Example: output/scalability_nvec/acorn_gamma/yfcc100m-10m_test0.01/profiling_results.csv
        marker_size: Size of markers in plots
        line_width: Width of lines in plots

    Note:
        ACORN scalability results are manually profiled (no automated benchmark scripts exist).
        Users must manually collect ACORN profiling data and provide CSV files with the expected format.
        If CSV files are not provided, ACORN will be excluded from the respective plots.
    """
    plt.rcParams.update({"font.size": 16})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    color_map = {
        "per_label_hnsw": "tab:blue",
        "per_label_ivf": "tab:orange",
        "parlay_ivf_graph": "tab:green",
        "parlay_ivf": "tab:green",
        "filtered_diskann": "tab:red",
        "shared_hnsw": "tab:purple",
        "acorn-1": "tab:pink",
        "acorn-gamma": "tab:gray",
        "curator": "tab:olive",
    }
    marker_map = {
        "per_label_hnsw": "o",
        "per_label_ivf": "X",
        "parlay_ivf_graph": "d",
        "parlay_ivf": "d",
        "filtered_diskann": "P",
        "shared_hnsw": "h",
        "acorn-1": "*",
        "acorn-gamma": "p",
        "curator": "^",
    }

    plot_memory_vs_nlabel_at_ax(
        ax=ax1,
        index_keys=nlabel_index_keys,
        output_dir=nlabel_output_dir,
        dataset_key=nlabel_dataset_key,
        test_size=nlabel_test_size,
        n_queries_path=nlabel_n_queries_path,
        min_recall=nlabel_min_recall,
        max_latency=nlabel_max_latency,
        acorn_1_nlabels_csv=acorn_1_nlabels_csv,
        acorn_gamma_nlabels_csv=acorn_gamma_nlabels_csv,
        color_map=color_map,
        marker_map=marker_map,
        marker_size=marker_size,
        line_width=line_width,
    )

    plot_memory_vs_dataset_size_at_ax(
        ax=ax2,
        results_dir=nvec_results_dir,
        index_keys=nvec_index_keys,
        recall_threshold=nvec_recall_threshold,
        acorn_1_nvecs_csv=acorn_1_nvecs_csv,
        acorn_gamma_nvecs_csv=acorn_gamma_nvecs_csv,
        color_map=color_map,
        marker_map=marker_map,
        marker_size=marker_size,
        line_width=line_width,
    )
    ax2.set_ylabel("")

    legend = fig.legend(
        ax1.get_legend().legend_handles,  # type: ignore
        nlabel_index_keys_readable,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncols=(len(nlabel_index_keys_readable) + 1) // 2,
        fontsize="small",
        columnspacing=0.5,
        handletextpad=0.5,
    )
    ax1.get_legend().remove()
    ax2.get_legend().remove()

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire()
