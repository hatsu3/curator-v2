from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

profile_results = [
    {
        "name": "P-HNSW",
        "build_time_sec": {
            "YFCC100M": None,
            "arXiv": 15882,
        },
    },
    {
        "name": "P-IVF",
        "build_time_sec": {
            "YFCC100M": None,
            "arXiv": 297,
        },
    },
    {
        "name": "Parlay",
        "build_time_sec": {
            "YFCC100M": 130615,
            "arXiv": 22122,
        },
    },
    {
        "name": "DiskANN",
        "build_time_sec": {
            "YFCC100M": 989280,
            "arXiv": 50871,
        },
    },
    {
        "name": r"ACORN-$\gamma$",
        "build_time_sec": {
            "YFCC100M": 340328,
            "arXiv": 24188,
        },
    },
    {
        "name": "ACORN-1",
        "build_time_sec": {
            "YFCC100M": 8496,
            "arXiv": 3003,
        },
    },
    {
        "name": "S-HNSW",
        "build_time_sec": {
            "YFCC100M": 7436,
            "arXiv": 1601,
        },
    },
    {
        "name": "S-IVF",
        "build_time_sec": {
            "YFCC100M": 44437,
            "arXiv": 387,
        },
    },
    {
        "name": "Curator",
        "build_time_sec": {
            "YFCC100M": 721,
            "arXiv": 107,
        },
    },
]

dataset_info = [
    {
        "name": "YFCC100M",
        "num_vecs": 9990000,
        "num_mds": 55293233,
    },
    {
        "name": "arXiv",
        "num_vecs": 1990000,
        "num_mds": 19755960,
    },
]


def plot_construction_time(
    output_path: str = "output/overall_results/figs/revision_construction_time.pdf",
):
    datasets = profile_results[0]["build_time_sec"].keys()

    df = pd.DataFrame(
        [
            {
                "index_key": res["name"],
                "construction_time": res["build_time_sec"][dataset],
                "dataset": dataset,
            }
            for res in profile_results
            for dataset in datasets
            if res["build_time_sec"][dataset] is not None
        ]
    )

    index_keys = [res["name"] for res in profile_results]

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(
        data=df,
        x="index_key",
        y="construction_time",
        hue="dataset",
        hue_order=["YFCC100M", "arXiv"],
        order=index_keys,
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Construction Time (s)")
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    ax.legend(title="", fontsize="small", ncol=len(datasets))

    for tick in ax.get_xticklabels():
        tick.set_fontsize(12)

    for tick in ax.get_xticklabels():
        tick.set_rotation(20)

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)


if __name__ == "__main__":
    fire.Fire()
