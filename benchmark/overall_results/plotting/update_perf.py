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
            "arXiv": "output/overall_results_v2/per_label_hnsw/arxiv-large-10_test0.005/results/ef128_m16.csv",
        },
    },
    {
        "name": "P-IVF",
        "result_path": {
            "YFCC100M": None,  # too large
            "arXiv": "output/overall_results_v2/per_label_ivf/arxiv-large-10_test0.005/results/nlist40.csv",
        },
    },
    {
        "name": "S-HNSW",
        "result_path": {
            "YFCC100M": "output/overall_results/shared_hnsw/yfcc100m-10m_test0.001/results/ef128_m32.csv",
            "arXiv": "output/overall_results_v2/shared_hnsw/arxiv-large-10_test0.005/results/ef128_m16.csv",
        },
    },
    {
        "name": "S-IVF",
        "result_path": {
            "YFCC100M": "output/overall_results/shared_ivf/yfcc100m-10m_test0.001/results/nlist32768_nprobe32.csv",
            "arXiv": "output/overall_results_v2/shared_ivf/arxiv-large-10_test0.005/results/nlist1600_nprobe16.csv",
        },
    },
    {
        "name": "Curator",
        "result_path": {
            "YFCC100M": "output/overall_results/curator_opt/yfcc100m-10m_test0.001/results/nlist32_sl256.csv",
            "arXiv": "output/overall_results_v2/curator_opt/arxiv-large-10_test0.005/results/nlist32_sl256.csv",
        },
    },
]


def plot_update_results(
    output_path: str = "output/update_results/figs/revision_update_perf.pdf",
):
    datasets = ["YFCC100M", "arXiv"]
    index_keys = [info["name"] for info in baseline_info]

    results = []
    for info in baseline_info:
        for dataset in datasets:
            if info["result_path"][dataset] is None:
                continue
            df = pd.read_csv(info["result_path"][dataset])
            results.append(
                {
                    "index_key": info["name"],
                    "dataset": dataset,
                    "insert_lat_avg": df["insert_lat_avg"][0] * 1000,
                    "access_grant_lat_avg": df["access_grant_lat_avg"][0] * 1000,
                    "delete_lat_avg": (
                        df["delete_lat_avg"][0] * 1000
                        if "delete_lat_avg" in df
                        else None
                    ),
                    "revoke_access_lat_avg": (
                        df["revoke_access_lat_avg"][0] * 1000
                        if "revoke_access_lat_avg" in df
                        else None
                    ),
                }
            )

    df = pd.DataFrame(results)

    print(df)

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(7, 5), constrained_layout=True)
    gs = fig.add_gridspec(2, 5)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :3])
    ax3 = fig.add_subplot(gs[1, 3:])

    sns.barplot(
        data=df,
        x="index_key",
        y="access_grant_lat_avg",
        hue="dataset",
        order=index_keys,
        hue_order=datasets,
        ax=ax1,
    )
    ax1.set_title("Label Insertion")
    ax1.set_yscale("log")
    ax1.set_xlabel("")
    ax1.set_ylabel("Latency (ms)")
    ax1.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    sns.barplot(
        data=df[~df["index_key"].str.startswith("P-")],
        x="index_key",
        y="insert_lat_avg",
        hue="dataset",
        order=[i for i in index_keys if not i.startswith("P-")],
        hue_order=datasets,
        ax=ax2,
    )
    ax2.set_title("Vector Insertion")
    ax2.set_yscale("log")
    ax2.set_xlabel("")
    ax2.set_ylabel("Latency (ms)")
    ax2.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    delete_df = pd.melt(
        df[df["index_key"] == "Curator"],  # type: ignore
        id_vars=["dataset", "index_key"],
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
        hue="dataset",
        hue_order=datasets,
        ax=ax3,
    )
    ax3.set_title("Deletion (Curator)")
    ax3.set_yscale("log")
    ax3.set_xlabel("")
    ax3.set_ylabel("")
    ax3.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    ylim_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0], ax3.get_ylim()[0])
    ylim_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1], ax3.get_ylim()[1])
    for ax in [ax1, ax2, ax3]:
        ax.set_ylim(ylim_min, ylim_max)

    ax1.get_legend().set_title("")
    ax2.get_legend().remove()
    ax3.get_legend().remove()

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


if __name__ == "__main__":
    fire.Fire()
