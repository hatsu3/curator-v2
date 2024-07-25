import json
from functools import cache
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from benchmark.selectivity.dataset import SelectivityDataset


@cache
def load_dataset(dataset_cache_dir: str = "data/selectivity/random_yfcc100m"):
    return SelectivityDataset.load(dataset_cache_dir)


def select_best_result(
    output_dir: str = "output/selectivity/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    max_selectivity: float = 1.0,
    min_recall: float = 0.8,
    max_recall: float = 0.9,
    max_latency: float = 1e-2,
):
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    dataset = load_dataset(dataset_cache_dir)

    agg_results = list()
    json_files = list(results_dir.glob("*.json"))
    for json_file in tqdm(
        json_files, total=len(json_files), desc=f"Processing results in {results_dir}"
    ):
        preproc_res_path = json_file.with_suffix(".csv")
        if not preproc_res_path.exists():
            preproc_res = pd.concat(
                [
                    preprocess_per_config_result(result, dataset, max_selectivity)
                    for result in json.load(open(json_file))
                ]
            )
            print(f"Saving preprocessed results to {preproc_res_path} ...")
            preproc_res.to_csv(preproc_res_path, index=False)
        else:
            print(f"Loading preprocessed results from {preproc_res_path} ...")
            preproc_res = pd.read_csv(preproc_res_path)

        agg_results.append(preproc_res)

    agg_results_df = pd.concat(agg_results)
    if "filtered_diskann" in str(output_dir):
        print(agg_results_df[agg_results_df["selectivity"] < 0.06])
    filtered_df = agg_results_df[
        (agg_results_df["recall"] >= min_recall)
        & (agg_results_df["recall"] <= max_recall)
        & (agg_results_df["latency"] <= max_latency)
    ]
    best_results_df = (
        filtered_df.groupby("selectivity")
        .apply(lambda group: group.loc[group["latency"].idxmin()])  # type: ignore
        .reset_index(drop=True)
    )

    return best_results_df


def preprocess_per_config_result(
    results: dict, dataset: SelectivityDataset, max_selectivity: float = 1.0
):
    stats_gen = iter(
        results.get("query_stats", [{"n_dists": 1} for _ in results["query_recalls"]])
    )
    recalls_gen = iter(results["query_recalls"])
    lats_gen = iter(results["query_latencies"])

    per_selectivity_results = {
        sel: {
            "n_dists": list(),
            "recalls": list(),
            "latencies": list(),
        }
        for sel in dataset.all_selectivities
    }
    for __, access_list in zip(dataset.test_vecs, dataset.test_mds):
        for label in access_list:
            sel = dataset.label_to_selectivity[label]
            if sel > max_selectivity:
                continue
            stats, recall, latency = next(stats_gen), next(recalls_gen), next(lats_gen)
            per_selectivity_results[sel]["n_dists"].append(
                stats.get("n_dists", None) or stats["n_ndists"]
            )
            per_selectivity_results[sel]["recalls"].append(recall)
            per_selectivity_results[sel]["latencies"].append(latency)

    per_selectivity_results_agg = pd.DataFrame(
        {
            sel: {
                "n_dists": np.mean(res["n_dists"]).item(),
                "recall": np.mean(res["recalls"]).item(),
                "latency": np.mean(res["latencies"]).item(),
                "memory_usage_kb": results["index_size_kb"],
            }
            for sel, res in per_selectivity_results.items()
        }
    ).T.reset_index(names=["selectivity"])

    return per_selectivity_results_agg


def plot_overall_results(
    index_keys: list[str] = [
        "curator_opt",
        "shared_hnsw",
        "pre_filtering",
        "filtered_diskann",
    ],
    index_keys_readable: list[str] = [
        "Curator",
        "Shared HNSW",
        "Pre-Filtering",
        "Filtered DiskANN",
    ],
    output_dir: str = "output/selectivity",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    pre_filtering_max_sel: float = 0.1,
    min_recall: float = 0.8,
    max_recall: float = 1.0,
    max_latency: float = 0.02,
    output_path: str = "output/selectivity/figs/overall_results.pdf",
):
    best_results: dict[str, pd.DataFrame] = dict()
    for index_key in index_keys:
        max_sel = pre_filtering_max_sel if index_key == "pre_filtering" else 1.0
        best_results[index_key] = select_best_result(
            output_dir=str(Path(output_dir) / index_key),
            dataset_key=dataset_key,
            test_size=test_size,
            dataset_cache_dir=dataset_cache_dir,
            max_selectivity=max_sel,
            min_recall=min_recall,
            max_recall=max_recall,
            max_latency=max_latency,
        )

    df = pd.concat(
        [
            results.assign(index_type=index_type)
            for index_type, results in best_results.items()
        ]
    ).reset_index(drop=True)
    df["index_type"] = df["index_type"].map(dict(zip(index_keys, index_keys_readable)))
    df["latency_ms"] = df["latency"] * 1000

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        data=df,
        x="selectivity",
        y="latency_ms",
        hue="index_type",
        style="index_type",
        markers=True,
        dashes=False,
        hue_order=index_keys_readable,
        errorbar=None,
        markersize=8,
        linewidth=2,
        ax=ax,
    )
    ax.set_xlabel("Filter Selectivity")
    ax.set_ylabel("Query Latency (ms)")
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    legend = plt.legend(
        ax.get_legend().legend_handles,
        index_keys_readable,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.2),
        ncols=len(index_keys_readable),
        fontsize="small",
        columnspacing=0.5,
    )

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire()
