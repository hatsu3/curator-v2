from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark.profiler import Dataset


def select_results(
    output_dir: str = "output/overall_results/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    max_index_size_mb: int | None = None,
    min_recall: float | None = None,
    max_search_latency: float | None = None,
) -> list[dict]:
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    all_results = list()
    for csv_file in results_dir.glob("*.csv"):
        all_results.extend(pd.read_csv(csv_file).to_dict(orient="records"))

    if max_index_size_mb is not None:
        all_results = [
            result
            for result in all_results
            if result["index_size_kb"] / 1000 <= max_index_size_mb
        ]

    if min_recall is not None:
        all_results = [
            result for result in all_results if result["recall_at_k"] >= min_recall
        ]

    if max_search_latency is not None:
        all_results = [
            result
            for result in all_results
            if result["query_lat_avg"] <= max_search_latency
        ]

    return all_results


def select_pareto_front(
    df: pd.DataFrame,
    x_key: str = "query_lat_avg",
    y_key: str = "recall_at_k",
    min_x: bool = True,  # minimize x
    min_y: bool = False,  # maximize y
):
    def is_dominated(r1, r2):
        x_worse = r1[x_key] > r2[x_key] if min_x else r1[x_key] < r2[x_key]
        y_worse = r1[y_key] > r2[y_key] if min_y else r1[y_key] < r2[y_key]
        return x_worse and y_worse

    pareto_front = []
    for i, r1 in df.iterrows():
        if any(is_dominated(r1, r2) for _, r2 in df.iterrows()):
            continue
        pareto_front.append(r1)

    return pd.DataFrame(pareto_front)


def load_all_results(
    output_dir: str = "output/overall_results/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
):
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    all_results = list()
    for csv_file in results_dir.glob("*.csv"):
        for res in pd.read_csv(csv_file).to_dict(orient="records"):
            all_results.append(
                {
                    "filename": csv_file.name,
                    **res,
                }
            )

    all_results_df = pd.DataFrame(all_results)
    all_results_df = all_results_df.reset_index(drop=True)
    return all_results_df


def select_best_config(
    output_dir: str = "output/overall_results/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    alpha: float = 1.0,
    min_recall: float = 0.9,
):
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    all_results = list()
    for csv_file in results_dir.glob("*.csv"):
        for res in pd.read_csv(csv_file).to_dict(orient="records"):
            all_results.append(
                {
                    "filename": csv_file.name,
                    **res,
                }
            )

    best_result = min(
        (res for res in all_results if res["recall_at_k"] > min_recall),
        key=lambda res: res["query_lat_avg"] * 1000 - alpha * res["recall_at_k"],
    )

    return best_result


def plot_construction_time(
    index_keys: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "parlay_ivf_seq",
        "filtered_diskann",
        "shared_hnsw",
        "shared_ivf",
        "curator_opt",
    ],
    index_keys_readable: list[str] = [
        "P-HNSW",
        "P-IVF",
        "Parlay",
        "DiskANN",
        "S-HNSW",
        "S-IVF",
        "Curator",
    ],
    output_dir: str = "output/overall_results",
    min_recall: float = 0.88,
    output_path: str = "output/overall_results/figs/construction_time.pdf",
):
    def select_per_dataset_results(
        dataset_key: str,
        test_size: float,
        min_recall: float,
    ):
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
        num_train_vecs = dataset.train_vecs.shape[0]
        num_labels = sum(len(access_list) for access_list in dataset.train_mds)

        best_results = dict()
        for index_key, index_key_readable in zip(index_keys, index_keys_readable):
            best_result = pd.DataFrame(
                select_results(
                    output_dir=str(Path(output_dir) / index_key),
                    dataset_key=dataset_key,
                    test_size=test_size,
                    min_recall=min_recall,
                )
            )
            # select the best result based on query latency
            best_result = best_result.iloc[best_result["query_lat_avg"].idxmin()]  # type: ignore

            if best_result.empty:
                print(f"No results found for {index_key}")
                continue

            best_result["construction_time"] = 0.0
            if "train_latency" in best_result.index:
                best_result["construction_time"] += best_result["train_latency"]

            if "batch_insert_latency" in best_result.index:
                best_result["construction_time"] += best_result["batch_insert_latency"]
            else:
                assert "insert_lat_avg" in best_result.index
                assert "access_grant_lat_avg" in best_result.index
                best_result["construction_time"] += (
                    best_result["insert_lat_avg"] * num_train_vecs
                    + best_result["access_grant_lat_avg"] * num_labels
                )

            best_result["index_key"] = index_key_readable
            best_results[index_key] = best_result

        return pd.DataFrame([best_results[index_key] for index_key in index_keys])

    best_results_df = pd.concat(
        [
            select_per_dataset_results(
                dataset_key="yfcc100m",
                test_size=0.01,
                min_recall=min_recall,
            ).assign(Dataset="YFCC100M"),
            select_per_dataset_results(
                dataset_key="arxiv-large-10",
                test_size=0.005,
                min_recall=min_recall,
            ).assign(Dataset="arXiv"),
        ]
    )

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(7, 3.5))
    ax = sns.barplot(
        data=best_results_df,
        x="index_key",
        y="construction_time",
        hue="Dataset",
        order=index_keys_readable,
        ax=fig.gca(),
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Construction Time (s)")
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    ax.get_legend().set_title("")

    datasets = ["YFCC100M", "arXiv"]
    legend = fig.legend(
        ax.get_legend().legend_handles,  # type: ignore
        datasets,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncols=len(datasets),
    )
    ax.get_legend().remove()

    plt.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


def plot_overall_results(
    index_keys: list[str] = [
        "curator_opt",
        "filtered_diskann",
        "parlay_ivf_seq",
        "shared_hnsw",
        "shared_ivf",
        "per_label_hnsw",
        "per_label_ivf",
    ],
    index_keys_readable: list[str] = [
        "Curator",
        "Filtered DiskANN",
        "Parlay IVF",
        "Shared HNSW",
        "Shared IVF",
        "Per-Label HNSW",
        "Per-Label IVF",
    ],
    output_dir: str = "output/overall_results",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    max_index_size_mb: int = 2000,
    min_recall: float = 0.9,
    output_path: str = "output/overall_results/figs/overall_results.pdf",
):
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    num_queries = sum(len(access_list) for access_list in dataset.test_mds)

    def selected_results_df(
        max_index_size_mb: int | None = None,
        min_recall: float | None = None,
        pareto_x_key: str = "query_lat_avg",
        pareto_y_key: str = "recall_at_k",
        pareto_min_x: bool = True,
        pareto_min_y: bool = False,
    ):
        best_results = dict()
        for index_key in index_keys:
            best_results[index_key] = pd.DataFrame(
                select_results(
                    output_dir=str(Path(output_dir) / index_key),
                    dataset_key=dataset_key,
                    test_size=test_size,
                    max_index_size_mb=max_index_size_mb,
                    min_recall=min_recall,
                )
            )

            if "batch_query_latency" in best_results[index_key].columns:
                best_results[index_key]["query_lat_avg"] = (
                    best_results[index_key]["batch_query_latency"] / num_queries
                )

            best_results[index_key] = select_pareto_front(
                best_results[index_key],
                x_key=pareto_x_key,
                y_key=pareto_y_key,
                min_x=pareto_min_x,
                min_y=pareto_min_y,
            )

        return pd.concat(
            [
                best_results[index_key].assign(index_key=index_key)
                for index_key in index_keys
            ]
        )

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    df = selected_results_df(
        max_index_size_mb=max_index_size_mb,
        min_recall=None,
        pareto_x_key="query_lat_avg",
        pareto_y_key="recall_at_k",
        pareto_min_x=True,
        pareto_min_y=False,
    )
    df["query_lat_avg"] = df["query_lat_avg"] * 1000

    # df["index_size_gb"] = df["index_size_kb"] / 1024 / 1024
    # df["index_size_gb_bin"] = pd.cut(
    #     df["index_size_gb"],
    #     bins=np.arange(0, max(df["index_size_gb"]) + 0.5, 0.5).tolist(),
    #     right=False,
    #     labels=[
    #         f"{i}-{i + 0.5}GB" for i in np.arange(0, max(df["index_size_gb"]), 0.5)
    #     ],
    # )

    sns.scatterplot(
        data=df,
        x="query_lat_avg",
        y="recall_at_k",
        # hue="index_size_gb_bin",
        hue="index_key",
        hue_order=index_keys,
        style="index_key",
        style_order=index_keys,
        ax=axes[0],
        s=50,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Query Latency (ms)")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_title(rf"Memory Usage $\leq$ {max_index_size_mb} MiB")

    df = selected_results_df(
        max_index_size_mb=None,
        min_recall=min_recall,
        pareto_x_key="query_lat_avg",
        pareto_y_key="index_size_kb",
        pareto_min_x=True,
        pareto_min_y=True,
    )
    df["query_lat_avg"] = df["query_lat_avg"] * 1000
    df["index_size_gb"] = df["index_size_kb"] / 1024 / 1024

    sns.scatterplot(
        data=df,
        x="query_lat_avg",
        y="index_size_gb",
        hue="index_key",
        hue_order=index_keys,
        style="index_key",
        style_order=index_keys,
        ax=axes[1],
        s=100,
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Query Latency (ms)")
    axes[1].set_ylabel("Memory Footprint (GiB)")
    axes[1].set_title(rf"Recall@10 $\geq$ {min_recall}")

    legend = fig.legend(
        axes[1].get_legend().legend_handles,
        index_keys_readable,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncols=(len(index_keys_readable) + 1) // 2,
    )
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


def plot_memory_footprint(
    index_keys: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "parlay_ivf_seq",
        "filtered_diskann",
        "shared_hnsw",
        "shared_ivf",
        "curator_opt",
    ],
    index_keys_readable: list[str] = [
        "P-HNSW",
        "P-IVF",
        "Parlay",
        "DiskANN",
        "S-HNSW",
        "S-IVF",
        "Curator",
    ],
    output_dir: str = "output/overall_results",
    min_recall: float = 0.88,
    output_path: str = "output/overall_results/figs/memory_footprint.pdf",
):
    def select_per_dataset_results(
        dataset_key: str,
        test_size: float,
        min_recall: float,
    ):
        best_results = dict()
        for index_key, index_key_readable in zip(index_keys, index_keys_readable):
            best_res = pd.DataFrame(
                select_results(
                    output_dir=str(Path(output_dir) / index_key),
                    dataset_key=dataset_key,
                    test_size=test_size,
                    min_recall=min_recall,
                )
            )

            best_res = best_res.iloc[best_res["query_lat_avg"].idxmin()]  # type: ignore
            best_res["index_key"] = index_key_readable
            best_results[index_key] = best_res

        return pd.DataFrame([best_results[index_key] for index_key in index_keys])

    best_results_df = pd.concat(
        [
            select_per_dataset_results(
                dataset_key="yfcc100m",
                test_size=0.01,
                min_recall=min_recall,
            ).assign(Dataset="YFCC100M"),
            select_per_dataset_results(
                dataset_key="arxiv-large-10",
                test_size=0.005,
                min_recall=min_recall,
            ).assign(Dataset="arXiv"),
        ]
    )
    best_results_df["index_size_gb"] = best_results_df["index_size_kb"] / 1024 / 1024

    best_results_df["index_type"] = best_results_df["index_key"].map(
        {
            "P-HNSW": "Per-Label",
            "P-IVF": "Per-Label",
            "Parlay": "Per-Label",
            "S-HNSW": "Shared",
            "S-IVF": "Shared",
            "DiskANN": "Specialized",
            "Curator": "Specialized",
        }
    )

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(7, 3.5))
    ax = sns.barplot(
        data=best_results_df,
        x="index_key",
        y="index_size_gb",
        hue="Dataset",
        order=index_keys_readable,
        ax=fig.gca(),
    )

    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Memory Footprint (GB)")
    ax.set_ylim(0.7, None)
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    ax.get_legend().set_title("")

    datasets = ["YFCC100M", "arXiv"]
    legend = fig.legend(
        ax.get_legend().legend_handles,  # type: ignore
        datasets,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncols=len(datasets),
    )

    ax.get_legend().remove()

    plt.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


def plot_recall_vs_latency(
    index_keys: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "parlay_ivf_seq",
        "filtered_diskann",
        "shared_hnsw",
        "shared_ivf",
        "curator_opt",
    ],
    index_keys_readable: list[str] = [
        "Per-Label HNSW",
        "Per-Label IVF",
        "Parlay IVF",
        "Filtered DiskANN",
        "Shared HNSW",
        "Shared IVF",
        "Curator",
    ],
    min_recall: float = 0.88,
    output_dir: str = "output/overall_results",
    output_path: str = "output/overall_results/figs/recall_vs_latency.pdf",
):
    def select_per_dataset_memory_vs_latency_results(
        dataset_key: str,
        test_size: float,
        min_recall: float,
    ):
        best_results = dict()
        for index_key in index_keys:
            best_res = pd.DataFrame(
                select_results(
                    output_dir=str(Path(output_dir) / index_key),
                    dataset_key=dataset_key,
                    test_size=test_size,
                    min_recall=min_recall,
                )
            )

            best_res = select_pareto_front(
                best_res,
                x_key="query_lat_avg",
                y_key="index_size_kb",
                min_x=True,
                min_y=True,
            )

            best_results[index_key] = best_res

        best_results_df = pd.concat(
            [
                best_results[index_key].assign(index_key=index_key_readable)
                for index_key, index_key_readable in zip(
                    index_keys, index_keys_readable
                )
            ]
        )

        best_results_df["query_lat_avg"] = best_results_df["query_lat_avg"] * 1000
        best_results_df["index_size_gb"] = (
            best_results_df["index_size_kb"] / 1024 / 1024
        )
        best_results_df["index_type"] = best_results_df["index_key"].map(
            {
                "P-HNSW": "Per-Label",
                "P-IVF": "Per-Label",
                "Parlay": "Per-Label",
                "S-HNSW": "Shared",
                "S-IVF": "Shared",
                "DiskANN": "Specialized",
                "Curator": "Specialized",
            }
        )
        best_results_df.sort_values(
            ["index_type", "index_key", "query_lat_avg"], inplace=True
        )

        return best_results_df

    def select_per_dataset_recall_vs_latency_results(
        dataset_key: str,
        test_size: float,
    ):
        best_results = dict()
        for index_key in index_keys:
            best_res = pd.DataFrame(
                select_results(
                    output_dir=str(Path(output_dir) / index_key),
                    dataset_key=dataset_key,
                    test_size=test_size,
                )
            )

            best_res = select_pareto_front(
                best_res,
                x_key="query_lat_avg",
                y_key="recall_at_k",
                min_x=True,
                min_y=False,
            )

            best_results[index_key] = best_res

        best_results_df = pd.concat(
            [
                best_results[index_key].assign(index_key=index_key_readable)
                for index_key, index_key_readable in zip(
                    index_keys, index_keys_readable
                )
            ]
        )

        best_results_df["query_lat_avg"] = best_results_df["query_lat_avg"] * 1000
        best_results_df["index_type"] = best_results_df["index_key"].map(
            {
                "P-HNSW": "Per-Label",
                "P-IVF": "Per-Label",
                "Parlay": "Per-Label",
                "S-HNSW": "Shared",
                "S-IVF": "Shared",
                "DiskANN": "Specialized",
                "Curator": "Specialized",
            }
        )
        best_results_df.sort_values(
            ["index_type", "index_key", "query_lat_avg"], inplace=True
        )

        return best_results_df

    def plot_per_dataset_memory_vs_latency(
        dataset_key_readable: str,
        dataset_key: str,
        test_size: float,
        ax,
        xlim=None,
        ylim=None,
        ylabel=True,
    ):
        df = select_per_dataset_memory_vs_latency_results(
            dataset_key=dataset_key,
            test_size=test_size,
            min_recall=min_recall,
        )

        print("Dataset:", dataset_key_readable)
        df["score"] = df["query_lat_avg"] + df["index_size_gb"]
        df = df.sort_values("score").groupby("index_key").first().reset_index()
        print(df[["index_key", "query_lat_avg", "index_size_gb"]])

        colors = sns.color_palette("tab10", n_colors=3)
        palette = {
            "Per-Label HNSW": colors[0],
            "Per-Label IVF": colors[0],
            "Parlay IVF": colors[0],
            "Filtered DiskANN": colors[2],
            "Shared HNSW": colors[1],
            "Shared IVF": colors[1],
            "Curator": colors[2],
        }

        sns.lineplot(
            data=df,
            x="query_lat_avg",
            y="index_size_gb",
            hue="index_key",
            hue_order=index_keys_readable,
            style="index_key",
            style_order=index_keys_readable,
            ax=ax,
            markers=True,
            dashes=False,
            palette=palette,
        )

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xscale("log")
        ax.set_yscale("log")
        # ax.set_xlabel("Query Latency (ms)")
        ax.set_xlabel("")

        ax.set_ylabel("Memory Footprint (GB)" if ylabel else "")
        ax.set_title(dataset_key_readable)
        ax.grid(axis="x", which="major", linestyle="-", alpha=0.6)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    def plot_per_dataset_recall_vs_latency(
        dataset_key_readable: str,
        dataset_key: str,
        test_size: float,
        ax,
        xlim=None,
        ylim=None,
        ylabel=True,
    ):
        df = select_per_dataset_recall_vs_latency_results(
            dataset_key=dataset_key,
            test_size=test_size,
        )

        colors = sns.color_palette("tab10", n_colors=3)
        palette = {
            "Per-Label HNSW": colors[0],
            "Per-Label IVF": colors[0],
            "Parlay IVF": colors[0],
            "Filtered DiskANN": colors[2],
            "Shared HNSW": colors[1],
            "Shared IVF": colors[1],
            "Curator": colors[2],
        }

        sns.lineplot(
            data=df,
            x="query_lat_avg",
            y="recall_at_k",
            hue="index_key",
            hue_order=index_keys_readable,
            style="index_key",
            style_order=index_keys_readable,
            ax=ax,
            markers=True,
            dashes=False,
            palette=palette,
        )

        if xlim is not None:
            ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xscale("log")
        ax.set_xlabel("Query Latency (ms)")
        ax.set_ylabel("Recall@10" if ylabel else "")
        # ax.set_title(dataset_key_readable)
        ax.grid(axis="x", which="major", linestyle="-", alpha=0.6)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    xlim = (3e-2, 2e2)
    plot_per_dataset_memory_vs_latency(
        "YFCC100M", "yfcc100m", 0.01, axes[0, 0], xlim=xlim, ylim=(0.7, 50)
    )
    plot_per_dataset_memory_vs_latency(
        "arXiv",
        "arxiv-large-10",
        0.005,
        axes[0, 1],
        xlim=xlim,
        ylim=(0.7, 50),
        ylabel=False,
    )
    plot_per_dataset_recall_vs_latency(
        "YFCC100M", "yfcc100m", 0.01, axes[1, 0], xlim=xlim
    )
    plot_per_dataset_recall_vs_latency(
        "arXiv",
        "arxiv-large-10",
        0.005,
        axes[1, 1],
        xlim=xlim,
        ylim=axes[1, 0].get_ylim(),
        ylabel=False,
    )

    legend = fig.legend(
        axes[0, 0].get_legend().legend_handles,
        index_keys_readable,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncols=(len(index_keys_readable) + 1) // 2,
        columnspacing=1.0,
        fontsize="small",
    )

    for ax in axes.flat:
        ax.get_legend().remove()

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


def plot_filtered_diskann_mem_vs_latency(
    output_dir: str = "output/overall_results/filtered_diskann",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_path: str = "output/overall_results/figs/filtered_diskann.pdf",
):
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    all_results = list()
    for csv_file in results_dir.glob("*.csv"):
        for res in pd.read_csv(csv_file).to_dict(orient="records"):
            all_results.append(
                {
                    "filename": csv_file.name,
                    **res,
                }
            )

    df = pd.DataFrame(all_results)
    df = df[df["ef_construct"] == df["ef_construct"].max()]
    df["index_size_gb"] = df["index_size_kb"] / 1024 / 1024
    df["query_lat_avg"] = df["query_lat_avg"] * 1000

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    sns.lineplot(
        data=df,
        x="query_lat_avg",
        y="recall_at_k",
        hue="graph_degree",
        marker="o",
        markersize=8,
        linewidth=2,
        ax=axes[0],
    )

    sns.lineplot(
        data=df,
        x="graph_degree",
        y="index_size_gb",
        marker="o",
        markersize=8,
        linewidth=2,
        ax=axes[1],
    )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Query Latency (ms)")
    axes[0].set_ylabel("Recall@10")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles=handles,
        labels=labels,
        title="Graph Deg",
        fontsize=10,
        title_fontsize=10,
        loc="lower right",
    )
    axes[0].grid(axis="x", which="major", linestyle="-", alpha=0.6)
    axes[0].grid(axis="y", which="major", linestyle="-", alpha=0.6)

    axes[1].set_xlabel("Graph Degree")
    axes[1].set_ylabel("Memory Footprint (GB)")
    axes[1].set_xticks(df["graph_degree"].unique())
    axes[1].grid(axis="x", which="major", linestyle="-", alpha=0.6)
    axes[1].grid(axis="y", which="major", linestyle="-", alpha=0.6)

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


def plot_update_results(
    index_keys: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "shared_hnsw",
        "shared_ivf",
        "curator_opt",
    ],
    index_keys_readable: list[str] = [
        "P-HNSW",
        "P-IVF",
        "S-HNSW",
        "S-IVF",
        "Curator",
    ],
    output_dir: str = "output/overall_results",
    min_recall: float = 0.88,
    output_path: str = "output/update_results/figs/update_perf.pdf",
):
    def select_per_dataset_results(
        dataset_key: str,
        test_size: float,
        min_recall: float,
    ):
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
        num_queries = sum(len(access_list) for access_list in dataset.test_mds)

        best_results = dict()
        for index_key in index_keys:
            best_res = pd.DataFrame(
                select_results(
                    output_dir=str(Path(output_dir) / index_key),
                    dataset_key=dataset_key,
                    test_size=test_size,
                    min_recall=min_recall,
                )
            )

            if "batch_query_latency" in best_res.columns:
                best_res["query_lat_avg"] = (
                    best_res["batch_query_latency"] / num_queries
                )

            if "batch_insert_latency" in best_res.columns:
                best_res["train_latency"] = best_res["batch_insert_latency"]

            if best_res.empty:
                continue

            best_results[index_key] = best_res.iloc[best_res["query_lat_avg"].idxmin()]  # type: ignore

        return pd.DataFrame(best_results).T.reset_index(names="index_key")

    best_results_df = pd.concat(
        [
            select_per_dataset_results(
                dataset_key="yfcc100m",
                test_size=0.01,
                min_recall=min_recall,
            ).assign(Dataset="YFCC100M"),
            select_per_dataset_results(
                dataset_key="arxiv-large-10",
                test_size=0.005,
                min_recall=min_recall,
            ).assign(Dataset="arXiv"),
        ]
    )

    best_results_df["access_grant_lat_avg"] *= 1000
    best_results_df["insert_lat_avg"] *= 1000
    best_results_df["revoke_access_lat_avg"] *= 1000
    best_results_df["delete_lat_avg"] *= 1000
    best_results_df["index_key"] = best_results_df["index_key"].map(
        dict(zip(index_keys, index_keys_readable))
    )

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(7, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 5)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :3])
    ax3 = fig.add_subplot(gs[1, 3:])

    sns.barplot(
        data=best_results_df,
        x="index_key",
        y="access_grant_lat_avg",
        hue="Dataset",
        order=index_keys_readable,
        ax=ax1,
    )
    ax1.set_title("Label Insertion")
    ax1.set_yscale("log")
    ax1.set_xlabel("")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    sns.barplot(
        data=best_results_df[~best_results_df["index_key"].str.startswith("P-")],
        x="index_key",
        y="insert_lat_avg",
        hue="Dataset",
        order=[i for i in index_keys_readable if not i.startswith("P-")],
        ax=ax2,
    )
    ax2.set_title("Vector Insertion")
    ax2.set_yscale("log")
    ax2.set_xlabel("")
    ax2.set_ylabel("Latency (ms)")
    ax2.set_ylim(*ax1.get_ylim())
    ax2.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    delete_df = pd.melt(
        best_results_df[best_results_df["index_key"] == "Curator"],
        id_vars=["Dataset", "index_key"],
        value_vars=["revoke_access_lat_avg", "delete_lat_avg"],
        var_name="op",
        value_name="latency",
    )
    delete_df["op"] = delete_df["op"].map(
        {
            "revoke_access_lat_avg": "Label",
            "delete_lat_avg": "Vector",
        }
    )

    sns.barplot(
        data=delete_df,
        x="op",
        y="latency",
        hue="Dataset",
        ax=ax3,
    )
    ax3.set_title("Deletion (Curator)")
    ax3.set_yscale("log")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.set_ylim(*ax1.get_ylim())
    ax3.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    datasets = ["YFCC100M", "arXiv"]
    legend = fig.legend(
        ax1.get_legend().legend_handles,  # type: ignore
        datasets,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncols=len(datasets),
    )

    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


def plot_ablation_results(
    output_dir: str | Path = "output/ablation",
    metric: str = "beam_size",
    metric_readable: str = "Beam Size",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_path: str = "output/ablation/figs/ablation_beam_size.pdf",
):
    output_dir = Path(output_dir) / metric
    df = load_all_results(
        output_dir=str(output_dir),
        dataset_key=dataset_key,
        test_size=test_size,
    )
    df["query_lat_avg"] = df["query_lat_avg"] * 1000

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(4, 4))
    ax = sns.lineplot(
        data=df,
        x="query_lat_avg",
        y="recall_at_k",
        hue=metric,
        marker="o",
        markersize=8,
        linewidth=2,
    )
    ax.set_xlabel("Query Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.get_legend().set_title(metric_readable)
    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


def plot_ablation_results_combined(
    output_dir: str | Path = "output/ablation",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_path: str = "output/ablation/figs/ablation_combined.pdf",
):
    nlist_df = load_all_results(
        output_dir=str(Path(output_dir) / "nlist"),
        dataset_key=dataset_key,
        test_size=test_size,
    )
    bufcap_df = load_all_results(
        output_dir=str(Path(output_dir) / "max_sl_size"),
        dataset_key=dataset_key,
        test_size=test_size,
    )

    nlist_df["query_lat_avg"] = nlist_df["query_lat_avg"] * 1000
    bufcap_df["query_lat_avg"] = bufcap_df["query_lat_avg"] * 1000

    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    sns.lineplot(
        data=nlist_df,
        x="query_lat_avg",
        y="recall_at_k",
        hue="nlist",
        marker="o",
        markersize=8,
        linewidth=2,
        ax=ax1,
    )
    ax1.set_title("Branch Factor")
    ax1.set_xlabel("Query Latency (ms)")
    ax1.set_ylabel("Recall@10")
    ax1.get_legend().set_title("")

    sns.lineplot(
        data=bufcap_df,
        x="query_lat_avg",
        y="recall_at_k",
        hue="max_sl_size",
        marker="o",
        markersize=8,
        linewidth=2,
        ax=ax2,
    )
    ax2.set_title("Buffer Capacity")
    ax2.set_xlabel("Query Latency (ms)")
    ax2.set_ylabel("Recall@10")
    ax2.get_legend().set_title("")

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    fire.Fire()
