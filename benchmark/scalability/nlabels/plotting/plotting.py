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
):
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

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 19,
                        "index_size_gb": 1.6,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 38,
                        "index_size_gb": 1.8,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 76,
                        "index_size_gb": 2.1,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 152,
                        "index_size_gb": 2.4,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 304,
                        "index_size_gb": 2.6,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 608,
                        "index_size_gb": 2.8,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 1216,
                        "index_size_gb": 3.0,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 2432,
                        "index_size_gb": 3.4,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 4864,
                        "index_size_gb": 4.4,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "n_labels": 9728,
                        "index_size_gb": 6.3,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 19,
                        "index_size_gb": 1.2,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 38,
                        "index_size_gb": 1.4,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 76,
                        "index_size_gb": 1.7,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 152,
                        "index_size_gb": 2.0,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 304,
                        "index_size_gb": 2.2,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 608,
                        "index_size_gb": 2.4,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 1216,
                        "index_size_gb": 2.6,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 2432,
                        "index_size_gb": 3.0,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 4864,
                        "index_size_gb": 4.0,
                    },
                    {
                        "index_type": "acorn-1",
                        "n_labels": 9728,
                        "index_size_gb": 5.9,
                    },
                ]
            ),
        ]
    )

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
):
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

    combined_df = pd.concat(
        [
            combined_df,
            pd.DataFrame(
                [
                    {
                        "index_type": "acorn-gamma",
                        "subset_size": 2,
                        "index_size_gb": 3.1,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "subset_size": 4,
                        "index_size_gb": 6.5,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "subset_size": 6,
                        "index_size_gb": 9.8,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "subset_size": 8,
                        "index_size_gb": 13.1,
                    },
                    {
                        "index_type": "acorn-gamma",
                        "subset_size": 10,
                        "index_size_gb": 16.4,
                    },
                    {
                        "index_type": "acorn-1",
                        "subset_size": 2,
                        "index_size_gb": 2.7,
                    },
                    {
                        "index_type": "acorn-1",
                        "subset_size": 4,
                        "index_size_gb": 5.9,
                    },
                    {
                        "index_type": "acorn-1",
                        "subset_size": 6,
                        "index_size_gb": 9.2,
                    },
                    {
                        "index_type": "acorn-1",
                        "subset_size": 8,
                        "index_size_gb": 12.5,
                    },
                    {
                        "index_type": "acorn-1",
                        "subset_size": 10,
                        "index_size_gb": 15.7,
                    },
                ]
            ),
        ]
    )

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
    marker_size: int = 6,
    line_width: float = 1.5,
):
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
