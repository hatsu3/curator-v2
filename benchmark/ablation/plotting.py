from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import LogFormatter


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

    pareto_front = [
        row for _, row in df.iterrows()
        if not any(is_dominated(row, other) for _, other in df.iterrows())
    ]
    return pd.DataFrame(pareto_front) if pareto_front else pd.DataFrame()


def plot_skewness_results(
    ax: Axes,
    origin_results_path: str = "output/skewness/origin.csv",
    shuffled_results_path: str = "output/skewness/shuffled.csv",
):
    print(f"Loading results from {origin_results_path} and {shuffled_results_path} ...")
    origin_df = pd.read_csv(origin_results_path)
    origin_df = origin_df[["query_lat_avg", "recall_at_k"]]
    shuffled_df = pd.read_csv(shuffled_results_path)
    shuffled_df = shuffled_df[["query_lat_avg", "recall_at_k"]]

    df = pd.concat(
        [
            origin_df.assign(shuffled=False),
            shuffled_df.assign(shuffled=True),
        ]
    )
    df["query_lat_avg"] = df["query_lat_avg"] * 1000

    # Convert latency to QPS (queries per second)
    df["qps"] = 1000 / df["query_lat_avg"]

    sns.lineplot(
        data=df,
        x="recall_at_k",
        y="qps",
        hue="shuffled",
        hue_order=[False, True],
        marker="o",
        markersize=8,
        linewidth=2,
        ax=ax,
    )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")

    ax.legend(
        ax.get_legend_handles_labels()[0],
        ["Original", "Shuffled"],
        fontsize="small",
        loc="lower right",
    )


def plot_structural_constraint_results(
    ax: Axes,
    curator_results_dir: str,
    per_label_curator_results_dir: str,
    recall_threshold: float | None = 0.75,
    latency_threshold: float | None = 1.0,
    markers: bool = False,
):
    pl_curator_df = pd.concat(
        [
            pd.read_csv(csv_path)
            for csv_path in Path(per_label_curator_results_dir).glob("*.csv")
        ]
    )

    curator_df = pd.concat(
        [pd.read_csv(csv_path) for csv_path in Path(curator_results_dir).glob("*.csv")]
    )
    curator_df = curator_df[
        (curator_df["nlist"].isin(pl_curator_df["nlist"].unique()))
        & (curator_df["max_sl_size"].isin(pl_curator_df["max_sl_size"].unique()))
        & (curator_df["search_ef"].isin(pl_curator_df["search_ef"].unique()))
    ]

    if recall_threshold is not None:
        curator_df = curator_df[curator_df["recall_at_k"] >= recall_threshold]
        pl_curator_df = pl_curator_df[pl_curator_df["recall_at_k"] >= recall_threshold]

    if latency_threshold is not None:
        latency_threshold = latency_threshold / 1000
        curator_df = curator_df[curator_df["query_lat_avg"] <= latency_threshold]
        pl_curator_df = pl_curator_df[
            pl_curator_df["query_lat_avg"] <= latency_threshold
        ]

    curator_df = select_pareto_front(curator_df)
    pl_curator_df = select_pareto_front(pl_curator_df)

    combined_df = pd.concat(
        [
            curator_df.assign(index_type="Curator"),
            pl_curator_df.assign(index_type="Unconstrained"),
        ]
    )
    combined_df["query_lat_avg"] = combined_df["query_lat_avg"] * 1000

    # Convert latency to QPS (queries per second)
    combined_df["qps"] = 1000 / combined_df["query_lat_avg"]

    markers_kwargs: dict[str, Any] = (
        dict(
            marker="o",
            markersize=8,
        )
        if markers
        else {}
    )

    sns.lineplot(
        data=combined_df,
        x="recall_at_k",
        y="qps",
        hue="index_type",
        linewidth=2,
        ax=ax,
        **markers_kwargs,
    )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("QPS")
    ax.set_yscale("log")

    ax.legend(
        ax.get_legend_handles_labels()[0],
        ax.get_legend_handles_labels()[1],
        fontsize="small",
        loc="lower right",
    )


def plot_latency_breakdown(
    ax: Axes,
    results_path: str = "output/scalability_nvec/curator/yfcc100m-10m_test0.01/latency_breakdown.csv",
):
    df = pd.read_csv(results_path)
    df["Nodes"] = df["n_dists_node"] / df["subset_size"] * 100
    df["Vectors"] = df["n_dists_vec"] / df["subset_size"] * 100
    df["subset_size"] = (df["subset_size"] / 1e6).astype(int)
    df = df.drop(columns=["n_dists_node", "n_dists_vec"])

    ax = df.plot(
        kind="bar",
        x="subset_size",
        y=["Vectors", "Nodes"],
        stacked=True,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_xlabel("Number of Vectors (M)")
    ax.set_ylabel("% Indexed Vectors")
    ax.legend(fontsize="small")


def plot_latency_breakdown_v2(
    output_path: str,
    results_path: str = "output/scalability_nvec/curator/yfcc100m-10m_test0.01/latency_breakdown.csv",
):
    df = pd.read_csv(results_path)
    df["node_pct"] = df["n_dists_node"] / df["subset_size"] * 100
    df["vec_pct"] = df["n_dists_vec"] / df["subset_size"] * 100
    df["subset_size"] = (df["subset_size"] / 1e6).astype(int)

    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    df.plot(
        x="subset_size",
        y=["n_dists_vec", "n_dists_node"],
        ax=ax1,
        logx=True,
        marker="o",
    )
    ax1.set_xlabel("Number of Vectors (M)")
    ax1.set_ylabel("Number of Dist Comp")

    ax1.xaxis.set_major_formatter(LogFormatter(10, labelOnlyBase=False))
    ax1.xaxis.set_minor_formatter(LogFormatter(10, labelOnlyBase=False))

    ax1.legend(["Vec", "Node"], fontsize="small")

    df.plot(
        kind="bar",
        x="subset_size",
        y=["vec_pct", "node_pct"],
        stacked=True,
        ax=ax2,
    )
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    ax2.set_xlabel("Number of Vectors (M)")
    ax2.set_ylabel("% Indexed Vectors")
    ax2.legend(["Vec", "Node"], fontsize="small")

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


def plot_combined(
    output_path: str,
    structure_curator_results_dir: str = "output/overall_results/curator_opt/yfcc100m_test0.01/results",
    structure_per_label_curator_results_dir: str = "output/structural/per_label_curator/yfcc100m_test0.01/results",
    skewness_origin_results_path: str = "output/skewness/origin.csv",
    skewness_shuffled_results_path: str = "output/skewness/shuffled.csv",
    structure_recall_threshold: float | None = 0.75,
    structure_latency_threshold: float | None = 1.0,
    structure_markers: bool = False,
    breakdown_results_path: str = "output/scalability_nvec/curator/yfcc100m-10m_test0.01/latency_breakdown.csv",
    share_xylim: bool = True,
):
    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3))

    plot_structural_constraint_results(
        ax1,
        structure_curator_results_dir,
        structure_per_label_curator_results_dir,
        recall_threshold=structure_recall_threshold,
        latency_threshold=structure_latency_threshold,
        markers=structure_markers,
    )
    plot_skewness_results(
        ax2, skewness_origin_results_path, skewness_shuffled_results_path
    )
    plot_latency_breakdown(ax3, breakdown_results_path)

    if share_xylim:
        shared_xlim = (
            min(ax1.get_xlim()[0], ax2.get_xlim()[0]),
            max(ax1.get_xlim()[1], ax2.get_xlim()[1]),
        )
        shared_ylim = (
            min(ax1.get_ylim()[0], ax2.get_ylim()[0]),
            max(ax1.get_ylim()[1], ax2.get_ylim()[1]),
        )
        ax1.set_xlim(shared_xlim)
        ax2.set_xlim(shared_xlim)
        ax1.set_ylim(shared_ylim)
        ax2.set_ylim(shared_ylim)

    fig.tight_layout(pad=0.5)

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    fire.Fire()
