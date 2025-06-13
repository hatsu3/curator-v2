import json
from pathlib import Path

import faiss
import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.selectivity.dataset import SelectivityDataset
from benchmark.selectivity.plotting import preprocess_per_config_result
from indexes.hybrid_curator import HybridCurator


def exp_hybrid_curator_selectivity(
    output_path: str,
    M: int = 32,
    gamma: int = 10,
    M_beta: int = 64,
    n_branches: int = 16,
    leaf_size: int = 128,
    use_local_sel: bool = False,
    sel_threshold: float = 0.2,
    search_ef_space: list[int] = [16, 32, 64, 128, 256],
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
):
    profiler = IndexProfiler()

    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)
    profiler.set_dataset(dataset)

    print(
        f"Building index with M = {M}, gamma = {gamma}, M_beta = {M_beta}, "
        f"n_branches = {n_branches}, leaf_size = {leaf_size} ... "
    )
    index_config = IndexConfig(
        index_cls=HybridCurator,
        index_params={
            "dim": dataset.dim,
            "M": M,
            "gamma": gamma,
            "M_beta": M_beta,
            "n_branches": n_branches,
            "leaf_size": leaf_size,
            "n_uniq_labels": dataset.num_labels,
            "use_local_sel": use_local_sel,
            "sel_threshold": sel_threshold,
        },
        search_params={
            "curator_search_ef": search_ef_space[0] * 8,
            "acorn_search_ef": search_ef_space[0],
        },
    )

    n_threads = faiss.omp_get_max_threads()
    print(f"Setting # threads to {n_threads} ...")

    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=True,
    )

    print(f"Setting # threads to 1 ...")
    faiss.omp_set_num_threads(1)

    results = list()
    for search_ef in search_ef_space:
        print(f"Querying index with search_ef = {search_ef} ... ")
        profiler.set_index_search_params(
            {
                "curator_search_ef": search_ef * 8,
                "acorn_search_ef": search_ef,
            }
        )
        query_res = profiler.do_query(return_verbose=True, return_stats=False)
        query_res.pop("query_results")
        results.append(
            {
                "M": M,
                "gamma": gamma,
                "M_beta": M_beta,
                "n_branches": n_branches,
                "leaf_size": leaf_size,
                "use_local_sel": use_local_sel,
                "sel_threshold": sel_threshold,
                "search_ef": search_ef,
                **query_res,
                **build_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def plot_profiling_results(
    hybrid_curator_results_json_path: str,
    acorn_results_json_path: str,
    output_path: str,
    min_recall: float = 0.88,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    max_selectivity: float = 1.0,
    log_scale: bool = False,
):
    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)

    def load_results(results_json_path: str):
        print(f"Loading results from {results_json_path} ...")
        with open(results_json_path, "r") as f:
            results = json.load(f)

        preproc_res_l = list()
        for result in results:
            d = preprocess_per_config_result(result, dataset, max_selectivity)
            d["search_ef"] = result["search_ef"]
            preproc_res_l.append(d)

        # See benchmark/selectivity/plotting.py:select_best_result
        preproc_res = pd.concat(preproc_res_l)
        filtered_df = preproc_res[(preproc_res["recall"] >= min_recall)]
        best_results_df = (
            filtered_df.groupby("selectivity")
            .apply(lambda group: group.iloc[group["latency"].argmin()])  # type: ignore
            .reset_index(drop=True)
        )
        best_results_df["latency_ms"] = best_results_df["latency"] * 1e3
        print(best_results_df)
        return best_results_df

    hybrid_curator_df = load_results(hybrid_curator_results_json_path)
    acorn_df = load_results(acorn_results_json_path)
    hybrid_curator_df["index"] = "Hybrid Curator"
    acorn_df["index"] = "Acorn"
    best_results_df = pd.concat([hybrid_curator_df, acorn_df])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    sns.lineplot(
        data=best_results_df,
        x="selectivity",
        y="latency_ms",
        hue="index",
        markers=True,
        dashes=False,
        errorbar=None,
        markersize=8,
        linewidth=2,
        ax=ax1,
    )
    ax1.set_xlabel("Filter Selectivity")
    ax1.set_ylabel("Query Latency (ms)")
    ax1.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    if log_scale:
        ax1.set_xscale("log")
    ax1.legend_.set_title(None)

    sns.lineplot(
        data=best_results_df,
        x="selectivity",
        y="recall",
        hue="index",
        markers=True,
        dashes=False,
        errorbar=None,
        markersize=8,
        linewidth=2,
        ax=ax2,
    )
    ax2.set_xlabel("Filter Selectivity")
    ax2.set_ylabel("Recall")
    ax2.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    ax2.set_ylim(0.0, 1.0)
    if log_scale:
        ax2.set_xscale("log")
    ax2.legend_.set_title(None)

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


def plot_profiling_results2(
    hybrid_curator_global_results_json_path: str,
    hybrid_curator_local_results_json_path: str,
    acorn_gamma1_results_json_path: str,
    # acorn_gamma10_results_json_path: str,
    output_path: str,
    min_recall: float = 0.88,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    max_selectivity: float = 1.0,
    log_scale: bool = False,
):
    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)

    def load_results(results_json_path: str):
        print(f"Loading results from {results_json_path} ...")
        with open(results_json_path, "r") as f:
            results = json.load(f)

        preproc_res_l = list()
        for result in results:
            d = preprocess_per_config_result(result, dataset, max_selectivity)
            d["search_ef"] = result["search_ef"]
            preproc_res_l.append(d)

        # See benchmark/selectivity/plotting.py:select_best_result
        preproc_res = pd.concat(preproc_res_l)
        filtered_df = preproc_res[(preproc_res["recall"] >= min_recall)]
        best_results_df = (
            filtered_df.groupby("selectivity")
            .apply(lambda group: group.iloc[group["latency"].argmin()])  # type: ignore
            .reset_index(drop=True)
        )
        best_results_df["latency_ms"] = best_results_df["latency"] * 1e3
        print(best_results_df)
        return best_results_df

    hybrid_curator_global_df = load_results(hybrid_curator_global_results_json_path)
    hybrid_curator_local_df = load_results(hybrid_curator_local_results_json_path)
    acorn_gamma1_df = load_results(acorn_gamma1_results_json_path)
    # acorn_gamma10_df = load_results(acorn_gamma10_results_json_path)
    hybrid_curator_global_df["index"] = "Curator (Global)"
    hybrid_curator_local_df["index"] = "Curator (Local)"
    acorn_gamma1_df["index"] = "Acorn-1"
    # acorn_gamma10_df["index"] = "Acorn-10"
    best_results_df = pd.concat([
        hybrid_curator_global_df,
        hybrid_curator_local_df,
        acorn_gamma1_df,
        # acorn_gamma10_df,
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    sns.lineplot(
        data=best_results_df,
        x="selectivity",
        y="latency_ms",
        hue="index",
        markers=True,
        dashes=False,
        errorbar=None,
        markersize=8,
        linewidth=2,
        ax=ax1,
    )
    ax1.set_xlabel("Filter Selectivity")
    ax1.set_ylabel("Query Latency (ms)")
    ax1.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    if log_scale:
        ax1.set_xscale("log")
    ax1.legend_.set_title(None)

    sns.lineplot(
        data=best_results_df,
        x="selectivity",
        y="recall",
        hue="index",
        markers=True,
        dashes=False,
        errorbar=None,
        markersize=8,
        linewidth=2,
        ax=ax2,
    )
    ax2.set_xlabel("Filter Selectivity")
    ax2.set_ylabel("Recall")
    ax2.grid(axis="y", which="major", linestyle="-", alpha=0.6)
    ax2.set_ylim(0.0, 1.0)
    if log_scale:
        ax2.set_xscale("log")
    ax2.legend_.set_title(None)

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


if __name__ == "__main__":
    """
    python -m benchmark.selectivity.hybrid_curator \
        exp_hybrid_curator_selectivity \
            --output_path test_hybrid_curator_selectivity.json \
            --M 32 \
            --gamma 10 \
            --M_beta 64 \
            --n_branches 16 \
            --leaf_size 128 \
            --sel_threshold 0.2 \
            --search_ef_space "[16, 32, 64, 128, 256]" \
            --dataset_cache_dir data/selectivity/random_yfcc100m

    python -m benchmark.selectivity.hybrid_curator \
        plot_profiling_results \
            --hybrid_curator_results_json_path test_hybrid_curator_selectivity.json \
            --acorn_results_json_path test_acorn_selectivity.json \
            --output_path test_hybrid_curator_selectivity.pdf \
            --dataset_cache_dir data/selectivity/random_yfcc100m \
            --min_recall 0.90 \
            --max_selectivity 1.0

    python -m benchmark.selectivity.hybrid_curator \
        plot_profiling_results2 \
            --hybrid_curator_global_results_json_path test_hybrid_curator_selectivity_th01_log_global.json \
            --hybrid_curator_local_results_json_path test_hybrid_curator_selectivity_th01_log_local.json \
            --acorn_gamma1_results_json_path test_acorn_selectivity_log_global.json \
            --output_path test_hybrid_curator_selectivity_th01_combined.pdf \
            --dataset_cache_dir data/selectivity/log_scale_yfcc100m \
            --min_recall 0.90 \
            --max_selectivity 1.0 \
            --log_scale
    """
    fire.Fire()
