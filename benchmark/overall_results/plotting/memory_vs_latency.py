from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


baseline_info = [
    {
        "name": "P-HNSW",
        "result_path": {
            "YFCC100M": None,  # too large
            "arXiv": "output/overall_results/per_label_hnsw/arxiv-large-10_test0.005/results/*.csv",
        },
    },
    {
        "name": "P-IVF",
        "result_path": {
            "YFCC100M": None,  # too large
            "arXiv": "output/overall_results/per_label_ivf/arxiv-large-10_test0.005/results/*.csv",
        },
    },
    {
        "name": "Parlay",
        "result_path": {
            "YFCC100M": [
                "output/overall_results/parlay_ivf/yfcc100m-10m_test0.001/results/cutoff1000c40r8i10.csv",
                "output/overall_results/parlay_ivf/yfcc100m-10m_test0.001/results/cutoff1000c40r16i10.csv",
                "output/overall_results/parlay_ivf/yfcc100m-10m_test0.001/results/cutoff1000c40r32i10.csv",
            ],
            "arXiv": "output/overall_results/parlay_ivf/arxiv-large-10_test0.005/results/*.csv",
        },
    },
    {
        "name": "DiskANN",
        "result_path": {
            "YFCC100M": [
                "output/overall_results/filtered_diskann/yfcc100m-10m_test0.001/results/R512_L600_a1.2_avg_lat.csv",
            ],
            "arXiv": "output/overall_results/filtered_diskann/arxiv-large-10_test0.005/results/*.csv",
        },
    },
    {
        "name": "S-HNSW",
        "result_path": {
            "YFCC100M": [
                "output/overall_results/shared_hnsw/yfcc100m-10m_test0.001/results/ef128_m32.csv",
                "output/overall_results/shared_hnsw/yfcc100m-10m_test0.001/results/ef128_m64.csv",
                "output/overall_results/shared_hnsw/yfcc100m-10m_test0.001/results/ef128_m128.csv",
            ],
            "arXiv": "output/overall_results/shared_hnsw/arxiv-large-10_test0.005/results/*.csv",
        },
    },
    {
        "name": "S-IVF",
        "result_path": {
            "YFCC100M": [
                "output/overall_results/shared_ivf/yfcc100m-10m_test0.001/results/nlist32768.csv",
            ],
            "arXiv": [
                "output/overall_results/shared_ivf/arxiv-large-10_test0.005/results/nlist200.csv",
                "output/overall_results/shared_ivf/arxiv-large-10_test0.005/results/nlist400.csv",
                "output/overall_results/shared_ivf/arxiv-large-10_test0.005/results/nlist800.csv",
                "output/overall_results/shared_ivf/arxiv-large-10_test0.005/results/nlist1600.csv",
            ],
        },
    },
    {
        "name": "ACORN-1",
        "result_path": {
            "YFCC100M": [
                "output/overall_results/acorn/yfcc100m-10m_test0.001/results/m32_g1_b64.csv",
                "output/overall_results/acorn/yfcc100m-10m_test0.001/results/m64_g1_b128.csv",
                "output/overall_results/acorn/yfcc100m-10m_test0.001/results/m128_g1_b256.csv",
            ],
            "arXiv": [
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m32_g1_b64.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m64_g1_b128.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m128_g1_b256.csv",
            ],
        },
    },
    {
        "name": r"ACORN-$\gamma$",
        "result_path": {
            "YFCC100M": [
                "output/overall_results/acorn/yfcc100m-10m_test0.001/results/m32_g40_b64.csv",
                "output/overall_results/acorn/yfcc100m-10m_test0.001/results/m32_g80_b64.csv",
                "output/overall_results/acorn/yfcc100m-10m_test0.001/results/m64_g20_b128.csv",
            ],
            "arXiv": [
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m16_gamma10_m_beta32_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m16_gamma10_m_beta64_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m16_gamma10_m_beta16_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m16_gamma20_m_beta16_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m16_gamma20_m_beta32_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m16_gamma20_m_beta64_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m32_gamma10_m_beta128_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m32_gamma10_m_beta32_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m32_gamma10_m_beta64_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m32_gamma20_m_beta32_avg_lat.csv",
                "output/overall_results/acorn/arxiv-large-10_test0.005/results/m32_gamma20_m_beta64_avg_lat.csv",
            ],
        },
    },
    {
        "name": "Curator",
        "result_path": {
            "YFCC100M": [
                "output/overall_results/curator/yfcc100m-10m_test0.001/results/nlist16_sl128.csv",
                "output/overall_results/curator/yfcc100m-10m_test0.001/results/nlist16_sl256.csv",
                "output/overall_results/curator/yfcc100m-10m_test0.001/results/nlist32_sl128.csv",
                "output/overall_results/curator/yfcc100m-10m_test0.001/results/nlist32_sl256.csv",
            ],
            "arXiv": "output/overall_results/curator/arxiv-large-10_test0.005/results/*.csv",
        },
    },
]


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


def preproc_per_dataset_results(dataset: str, min_recall: float = 0.9):
    results = []
    for res in baseline_info:
        if res["result_path"][dataset] is None:
            continue

        if isinstance(res["result_path"][dataset], str):
            p = Path(res["result_path"][dataset])
            paths = p.parent.glob(p.name)
        else:
            paths = [Path(p) for p in res["result_path"][dataset]]

        df = pd.concat([pd.read_csv(path) for path in paths])
        df["index_key"] = res["name"]
        df["index_size_gb"] = df["index_size_kb"] / 1024 / 1024

        # filter by recall
        df = df.query(f"recall_at_k >= {min_recall}")

        # select pareto front (latency vs index size)
        df = select_pareto_front(
            df, x_key="query_lat_avg", y_key="index_size_gb", min_x=True, min_y=True
        )

        results.append(df)

    return pd.concat(results)


def plot_per_dataset_memory_vs_latency(
    ax,
    dataset: str,
    index_keys: list[str],
    min_recall: float = 0.88,
):
    df = preproc_per_dataset_results(dataset, min_recall=min_recall)

    colors = sns.color_palette("tab10", n_colors=3)
    palette = {
        "P-HNSW": colors[0],
        "P-IVF": colors[0],
        "Parlay": colors[0],
        "DiskANN": colors[2],
        "ACORN-1": colors[2],
        r"ACORN-$\gamma$": colors[2],
        "S-HNSW": colors[1],
        "S-IVF": colors[1],
        "Curator": colors[2],
    }

    sns.lineplot(
        data=df,
        x="query_lat_avg",
        y="index_size_gb",
        hue="index_key",
        hue_order=index_keys,
        style="index_key",
        style_order=index_keys,
        ax=ax,
        markers=True,
        dashes=False,
        palette=palette,
    )

    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_xlabel("Query Latency (ms)")

    ax.set_ylabel("Memory Footprint (GB)")
    ax.set_title(dataset)
    ax.grid(axis="x", which="major", linestyle="-", alpha=0.6)
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)


def plot_memory_vs_latency(
    output_path: str = "output/overall_results/figs/revision_memory_vs_latency.pdf",
):
    index_keys = [info["name"] for info in baseline_info]

    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    plot_per_dataset_memory_vs_latency(ax1, "YFCC100M", index_keys)
    plot_per_dataset_memory_vs_latency(ax2, "arXiv", index_keys)

    # Remove y-label from the second plot
    ax2.set_ylabel("")

    # Set common x and y limits
    # xlim = (
    #     min(ax.get_xlim()[0] for ax in [ax1, ax2]),
    #     max(ax.get_xlim()[1] for ax in [ax1, ax2]),
    # )
    # ylim = (
    #     min(ax.get_ylim()[0] for ax in [ax1, ax2]),
    #     max(ax.get_ylim()[1] for ax in [ax1, ax2]),
    # )
    # for ax in [ax1, ax2]:
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)

    ax2.set_yscale("log")

    # Move legend to the top
    legend = fig.legend(
        ax1.get_legend().legend_handles,
        index_keys,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncols=(len(index_keys) + 1) // 2,
        columnspacing=1.0,
        fontsize="small",
    )

    for ax in [ax1, ax2]:
        ax.get_legend().remove()

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


"""
def plot_per_selectivity_results_memory_vs_latency(
    index_keys: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "parlay_ivf",
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
    labels_per_group: int = 100,
    percentiles: list[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    recompute: bool = False,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    recall_threshold: float = 0.85,
    output_dir: str = "output/overall_results_v2",
    output_path: str = "output/overall_results_v2/figs/memory_vs_latency_yfcc100m.pdf",
):
    # load dataset
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    label_groups = group_labels_by_selectivity(
        labels_per_group, percentiles, dataset_key, test_size, dataset=dataset
    )

    # aggregate profiling results across all selectivity levels for each index
    all_results: dict[str, pd.DataFrame] = dict()
    for index_key in index_keys:
        all_res = aggregate_per_selectivity_results(
            output_dir=str(Path(output_dir) / index_key),
            labels_per_group=labels_per_group,
            percentiles=percentiles,
            dataset_key=dataset_key,
            test_size=test_size,
            dataset=dataset,
            label_groups=label_groups,
            recompute=recompute,
        )

        all_res = all_res[all_res["recall"] >= recall_threshold]
        all_results[index_key] = all_res

    # select pareto optimal results for each selectivity level
    # each dataframe in per_pct_best_results.keys() contains following columns:
    # [percentile, selectivity, recall, latency, memory_usage_kb, index_key, index_type]
    per_pct_best_results: dict[float, pd.DataFrame] = dict()
    for percentile in percentiles:
        best_results = dict()
        for index_key in index_keys:
            per_pct_results = all_results[index_key][
                all_results[index_key]["percentile"] == percentile
            ]

            best_res = select_pareto_front(
                per_pct_results,
                x_key="latency",
                y_key="memory_usage_kb",
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

        best_results_df["memory_usage_gb"] = (
            best_results_df["memory_usage_kb"] / 1024 / 1024
        )
        best_results_df["latency"] = best_results_df["latency"] * 1000
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
            ["index_type", "index_key", "latency"], inplace=True
        )

        per_pct_best_results[percentile] = best_results_df

    # shared plotting parameters
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

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, len(percentiles), figsize=(3 * len(percentiles), 3))

    for i, (ax, percentile) in enumerate(zip(axes, percentiles)):
        per_pct_df = per_pct_best_results[percentile]
        sns.lineplot(
            data=per_pct_df,
            x="latency",
            y="memory_usage_gb",
            hue="index_key",
            hue_order=index_keys_readable,
            style="index_key",
            style_order=index_keys_readable,
            ax=ax,
            markers=True,
            dashes=False,
            palette=palette,
        )

        ax.set_xscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("Memory Footprint (GB)" if i == 0 else "")
        ax.set_title(f"{percentile:.0%} Selectivity")
        ax.grid(axis="x", which="major", linestyle="-", alpha=0.6)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    legend = fig.legend(
        axes[0].get_legend().legend_handles,
        index_keys_readable,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncols=len(index_keys_readable),
        columnspacing=1.0,
    )

    global_xlim = (
        min(ax.get_xlim()[0] for ax in axes.flat),
        max(ax.get_xlim()[1] for ax in axes.flat),
    )
    global_ylim = (
        min(ax.get_ylim()[0] for ax in axes.flat),
        max(ax.get_ylim()[1] for ax in axes.flat),
    )

    for ax in axes.flat:
        ax.get_legend().remove()
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")
"""


if __name__ == "__main__":
    fire.Fire()
