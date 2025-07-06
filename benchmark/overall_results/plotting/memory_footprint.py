from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

profile_results = [
    {
        "name": "P-HNSW",
        "index_size_kb": {
            "YFCC100M": None,
            "arXiv": 32537764,
        },
    },
    {
        "name": "P-IVF",
        "index_size_kb": {
            "YFCC100M": None,
            "arXiv": 41045088,
        },
    },
    {
        "name": "Parlay",
        "index_size_kb": {
            "YFCC100M": 17447588,
            "arXiv": 5812440,
        },
    },
    {
        "name": "DiskANN",
        "index_size_kb": {
            "YFCC100M": 28227948,
            "arXiv": 3860676,
        },
    },
    {
        "name": r"ACORN-$\gamma$",
        "index_size_kb": {
            "YFCC100M": 17244532,
            "arXiv": 4066796,
        },
    },
    {
        "name": "ACORN-1",
        "index_size_kb": {
            "YFCC100M": 16501704,
            "arXiv": 4784472,
        },
    },
    {
        "name": "S-HNSW",
        "index_size_kb": {
            "YFCC100M": 14838828,
            "arXiv": 3260800,
        },
    },
    {
        "name": "S-IVF",
        "index_size_kb": {
            "YFCC100M": 14461348,
            "arXiv": 2918480,
        },
    },
    {
        "name": "Curator",
        "index_size_kb": {
            "YFCC100M": 11382532,
            "arXiv": 1810980,
        },
    },
]


def plot_memory_footprint(
    output_path: str = "output/overall_results/figs/revision_memory_footprint.pdf",
):
    datasets = profile_results[0]["index_size_kb"].keys()

    df = pd.DataFrame(
        [
            {
                "index_key": res["name"],
                "index_size_gb": res["index_size_kb"][dataset] / 1024 / 1024,
                "dataset": dataset,
            }
            for res in profile_results
            for dataset in datasets
            if res["index_size_kb"][dataset] is not None
        ]
    )

    index_keys = [res["name"] for res in profile_results]

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(
        data=df,
        x="index_key",
        y="index_size_gb",
        hue="dataset",
        order=index_keys,
        hue_order=["YFCC100M", "arXiv"],
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Memory Footprint (GB)")
    ax.set_ylim(0.7, None)
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
