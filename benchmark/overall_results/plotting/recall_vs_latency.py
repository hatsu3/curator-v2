import json
import pickle as pkl
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark.profiler import Dataset

baseline_info = [
    {
        "name": "Per-Label HNSW",
        "result_path": {
            "yfcc100m": None,  # too large
            "arxiv": "output/overall_results_v2/per_label_hnsw/arxiv-large-10_test0.005/results/ef128_m16_preproc.csv",
        },
    },
    {
        "name": "Per-Label IVF",
        "result_path": {
            "yfcc100m": None,  # too large
            "arxiv": "output/overall_results_v2/per_label_ivf/arxiv-large-10_test0.005/results/nlist40_preproc.csv",
        },
    },
    {
        "name": "Parlay IVF",
        "result_path": {
            "yfcc100m": "output/overall_results_v2/parlay_ivf/yfcc100m-10m_test0.001/results/cutoff1000c40r16i10_preproc.csv",
            "arxiv": "output/overall_results_v2/parlay_ivf/arxiv-large-10_test0.005/results/cutoff10000c1000r16i10_preproc.csv",
        },
    },
    {
        "name": "Filtered DiskANN",
        "result_path": {
            "yfcc100m": "output/overall_results_v2/filtered_diskann/yfcc100m-10m_test0.001/results/R512_L600_a1.2_preproc.csv",
            "arxiv": "output/overall_results_v2/filtered_diskann/arxiv-large-10_test0.005/results/r256ef256a1.2_preproc.csv",
        },
    },
    {
        "name": "Shared HNSW",
        "result_path": {
            "yfcc100m": "output/overall_results_v2/shared_hnsw/yfcc100m-10m_test0.001/results/ef128_m32_preproc.csv",
            "arxiv": "output/overall_results_v2/shared_hnsw/arxiv-large-10_test0.005/results/ef128_m16_preproc.csv",
        },
    },
    {
        "name": "Shared IVF",
        "result_path": {
            "yfcc100m": "output/overall_results_v2/shared_ivf/yfcc100m-10m_test0.001/results/nlist32768_preproc.csv",
            "arxiv": "output/overall_results_v2/shared_ivf/arxiv-large-10_test0.005/results/nlist1600_preproc.csv",
        },
    },
    {
        "name": "ACORN-1",
        "result_path": {
            "yfcc100m": "output/overall_results_v2/acorn/yfcc100m-10m_test0.001/results/m64_g1_b128_preproc.csv",
            "arxiv": "output/overall_results_v2/acorn/arxiv-large-10_test0.005/results/m64_g1_b128_preproc.csv",
        },
    },
    {
        "name": r"ACORN-$\gamma$",
        "result_path": {
            "yfcc100m": "output/overall_results_v2/acorn/yfcc100m-10m_test0.001/results/m64_g20_b128_preproc.csv",
            "arxiv": "output/overall_results_v2/acorn/arxiv-large-10_test0.005/results/m32_gamma20_m_beta64_preproc.csv",
        },
    },
    {
        "name": "Curator",
        "result_path": {
            "yfcc100m": "output/overall_results_v2/curator/yfcc100m-10m_test0.001/results/nlist32_sl256_preproc.csv",
            "arxiv": "output/overall_results_v2/curator/arxiv-large-10_test0.005/results/nlist32_sl256_preproc.csv",
        },
    },
]


def group_labels_by_selectivity(
    labels_per_group: int = 100,
    percentiles: list[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset: Dataset | None = None,
):
    """Group labels by selectivity

    Each group contains top-labels_per_group labels with selectivity closest to the percentile
    of the group.

    Args:
        labels_per_group (int): number of labels per group
        percentiles (list[float]): corresponding percentiles of selectivity of each label group
        dataset (Dataset | None): dataset object. if None, will be loaded by dataset_key and test_size

    Returns:
        list[dict]: list of dictionaries, each containing:
            percentile (float): percentile of selectivity of the group
            selectivity (float): selectivity of the group
            labels (list[int]): labels in the group
    """
    if dataset is None:
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    label2sel: dict[int, float] = dataset.label_selectivities
    sorted_sels = sorted(label2sel.values())

    label_groups = []

    for percentile in percentiles:
        percentile_index = int(np.ceil(percentile * len(sorted_sels)) - 1)
        selectivity = sorted_sels[percentile_index]
        sorted_labels = sorted(
            label2sel.keys(), key=lambda label: abs(label2sel[label] - selectivity)
        )
        label_groups.append(
            {
                "percentile": percentile,
                "selectivity": selectivity,
                "labels": sorted_labels[:labels_per_group],
            }
        )

    return label_groups


def preprocess_per_config_result(
    per_config_results: dict, dataset: Dataset, label_groups: list[dict]
):
    """Aggregate profiling results of a specific search configuration

    For each label group, compute the average recall and latency across queries of
    all labels in the group.

    Args:
        per_config_results (dict): profiling results of a specific search configuration
        dataset (Dataset): dataset object
        label_groups (list[dict]): label groups by selectivity.
            Each entry is a dictionary with keys: percentile (0-1), selectivity (0-1), labels (list[int])
    Returns:
        pd.DataFrame: aggregated dataframe with 4 columns:
            percentile, selectivity, recall, latency
    """

    recall_gen = iter(per_config_results["query_recalls"])
    latency_gen = iter(per_config_results["query_latencies"])

    per_sel_results = [
        {
            "percentile": label_group["percentile"],
            "selectivity": label_group["selectivity"],
            "recalls": list(),
            "latencies": list(),
        }
        for label_group in label_groups
    ]

    label_to_group = {
        label: idx
        for idx, label_group in enumerate(label_groups)
        for label in label_group["labels"]
    }

    for __, label_list in zip(dataset.test_vecs, dataset.test_mds):
        for label in label_list:
            recall, latency = next(recall_gen), next(latency_gen)

            # some labels may not be in any label group
            if label not in label_to_group:
                continue

            group_idx = label_to_group[label]
            per_sel_results[group_idx]["recalls"].append(recall)
            per_sel_results[group_idx]["latencies"].append(latency)

    per_sel_results_df = pd.DataFrame(
        [
            {
                "percentile": res["percentile"],
                "selectivity": res["selectivity"],
                "recall": np.mean(res["recalls"]).item(),
                "latency": np.mean(res["latencies"]).item(),
            }
            for res in per_sel_results
        ]
    )

    return per_sel_results_df


def aggregate_per_selectivity_results(
    results_path: Path | str,
    labels_per_group: int = 100,
    percentiles: list[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    dataset_key: str = "yfcc100m-10m",
    test_size: float = 0.001,
    output_path: Path | str | None = None,
):
    """Aggregate profiling results of a specific index across all search configurations

    Group labels by selectivity and aggregate profiling results of a specific index across
    all search configurations: each csv file contains profiling results of multiple search
    configurations.

    Args:
        results_path (Path | str): path to the results csv file, which should contain two
            list-valued columns: query_latencies, query_recalls
        output_path (Path | str | None): path to the output csv file. if None, will be set to
            results_path.with_name(f"{results_path.stem}_preproc.csv")
        labels_per_group (int): number of labels per group
        percentiles (list[float]): corresponding percentiles of selectivity of each label group
    Returns:
        pd.DataFrame: aggregated dataframe with 4 columns:
            percentile, selectivity, recall, latency
            rows with the same (percentile, selectivity) values correspond to different configurations
    """
    results_path = Path(results_path)

    if output_path is None:
        output_path = results_path.with_name(f"{results_path.stem}_preproc.csv")
    else:
        output_path = Path(output_path)

    print(f"Loading dataset {dataset_key} ...")
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    label_groups = group_labels_by_selectivity(
        labels_per_group, percentiles, dataset_key, test_size, dataset=dataset
    )

    print(f"Loading results from {results_path} ...")
    # convert results to DataFrame with two list-valued columns: query_latencies, query_recalls
    if results_path.suffix == ".pkl":
        results = pkl.load(results_path.open("rb"))
    elif results_path.suffix == ".csv":
        results_df = pd.read_csv(results_path)
        for col in ["query_latencies", "query_recalls"]:
            results_df[col] = results_df[col].apply(json.loads)
        results = results_df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file extension {results_path.suffix}")

    print("Preprocessing results ...")
    preproc_res = pd.concat(
        [preprocess_per_config_result(res, dataset, label_groups) for res in results]
    )

    print(f"Saving preprocessed results to {output_path} ...")
    preproc_res.to_csv(output_path, index=False)


def plot_per_selectivity_results_recall_vs_latency(
    percentiles: list[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    dataset_name: str = "yfcc100m",
    output_path: str = "output/overall_results/figs/revision_recall_vs_latency_per_sel_yfcc100m.pdf",
):
    """
    Plot recall vs latency across different selectivity levels on a specific dataset
    Args:
        labels_per_group (int): number of labels per group
        percentiles (list[float]): corresponding percentiles of selectivity of each label group
        recompute (bool): whether to re-aggregate preprocessed results. should be set if impl or config changes
    """

    # aggregate profiling results across all selectivity levels for each index
    all_results = {
        info["name"]: pd.read_csv(info["result_path"][dataset_name]).assign(
            index_key=info["name"]
        )
        for info in baseline_info
        if info["result_path"][dataset_name] is not None
    }

    per_pct_results = dict()
    for percentile in percentiles:
        res = pd.concat(
            [
                index_res[index_res["percentile"] == percentile]
                for index_res in all_results.values()
            ]
        )
        res["latency"] *= 1000  # convert to ms
        per_pct_results[percentile] = res

    pct_to_sel = {
        res["percentile"].iloc[0]: res["selectivity"].iloc[0]
        for res in per_pct_results.values()
    }

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, len(percentiles), figsize=(3 * len(percentiles), 3))
    index_keys = [
        info["name"]
        for info in baseline_info
        if info["result_path"][dataset_name] is not None
    ]

    for i, (ax, percentile) in enumerate(zip(axes, percentiles)):
        per_pct_df = per_pct_results[percentile]
        sns.lineplot(
            data=per_pct_df,
            x="latency",
            y="recall",
            hue="index_key",
            hue_order=index_keys,
            style="index_key",
            style_order=index_keys,
            ax=ax,
            markers={
                "Per-Label HNSW": "o",
                "Per-Label IVF": "X",
                "Parlay IVF": "d",
                "Filtered DiskANN": "P",
                "Shared HNSW": "h",
                "Shared IVF": "s",
                "ACORN-1": "*",
                r"ACORN-$\gamma$": "p",  # pentagon
                "Curator": "^",
            },
            palette={
                "Per-Label HNSW": "tab:blue",
                "Per-Label IVF": "tab:orange",
                "Parlay IVF": "tab:green",
                "Filtered DiskANN": "tab:red",
                "Shared HNSW": "tab:purple",
                "Shared IVF": "tab:brown",
                "ACORN-1": "tab:pink",
                r"ACORN-$\gamma$": "tab:gray",
                "Curator": "tab:olive",
            },
            dashes=False,
        )

        ax.set_xscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("Recall@10" if i == 0 else "")

        selectivity = pct_to_sel[percentile]
        percentile = 0.99 if percentile == 1.0 else percentile
        ax.set_title(f"{int(percentile * 100)}p Sel ({selectivity:.4f})")
        ax.grid(axis="x", which="major", linestyle="-", alpha=0.6)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    if len(index_keys) <= 8:
        legend = fig.legend(
            axes[0].get_legend().legend_handles,
            index_keys,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncols=len(index_keys),
            columnspacing=1.0,
        )
    else:
        legend = fig.legend(
            axes[0].get_legend().legend_handles,
            index_keys,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=(len(index_keys) + 1) // 2,
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


if __name__ == "__main__":
    fire.Fire()
