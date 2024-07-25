from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import parse
import seaborn as sns

from benchmark.scalability.dataset import ScalabilityDataset


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


def plot_overall_results(
    index_keys: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "parlay_ivf_graph",
        "filtered_diskann",
        "shared_hnsw",
        "curator",
    ],
    index_keys_readable: list[str] = [
        "Per-Label HNSW",
        "Per-Label IVF",
        "Parlay IVF",
        "Filtered DiskANN",
        "Shared HNSW",
        "Curator",
    ],
    output_dir: str = "output/scalability",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    n_queries_path: str = "benchmark/scalability/num_queries.csv",
    min_recall: float | None = 0.9,
    max_latency: float | None = 2e-3,
    output_path: str = "output/scalability/figs/overall_results.pdf",
):
    best_results: dict[str, pd.DataFrame] = dict()
    for index_key in index_keys:
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
    df["index_type"] = df["index_type"].map(dict(zip(index_keys, index_keys_readable)))

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(7, 4.5))
    ax = sns.lineplot(
        data=df,
        x="n_labels",
        y="index_size_gb",
        hue="index_type",
        style="index_type",
        markers=True,
        dashes=False,
        hue_order=index_keys_readable,
        errorbar=None,
        markersize=10,
        linewidth=2,
    )
    ax.grid(axis="y", which="major", linestyle="-")
    ax.set_xlabel("Number of Labels")
    ax.set_ylabel("Memory Footprint (GB)")

    legend = plt.legend(
        ax.get_legend().legend_handles,
        index_keys_readable,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.3),
        ncols=(len(index_keys_readable) + 1) // 2,
        columnspacing=1.0,
        fontsize="small",
    )

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


def plot_filtered_diskann_results(
    output_dir: str = "output/scalability/filtered_diskann",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    n_queries_path: str = "benchmark/scalability/num_queries.csv",
    output_path: str = "output/scalability/figs/filtered_diskann_results.pdf",
):
    df = load_all_results(
        output_dir=output_dir,
        dataset_key=dataset_key,
        test_size=test_size,
        n_queries_path=n_queries_path,
    )
    df = df[df["recall_at_k"] >= 0.9]
    df["index_size_gb"] = df["index_size_kb"] / 1024 / 1024
    df["query_lat_avg"] *= 1000
    df = df[df["query_lat_avg"] <= 10]

    # plt.rcParams.update({"font.size": 14})
    # fig = plt.figure(figsize=(7, 4.5))
    ax = sns.scatterplot(data=df, x="query_lat_avg", y="index_size_gb", hue="n_labels")
    ax.grid(axis="both", which="major", linestyle="-")
    ax.set_xlabel("Average Query Latency (ms)")
    ax.set_ylabel("Memory Footprint (GB)")

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    fire.Fire()
