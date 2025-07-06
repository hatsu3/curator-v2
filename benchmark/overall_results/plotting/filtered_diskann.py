from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
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


if __name__ == "__main__":
    fire.Fire(plot_filtered_diskann_mem_vs_latency)
