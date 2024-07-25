import json
import logging
from functools import partial
from itertools import product
from pathlib import Path
from typing import IO

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import parse
import seaborn as sns
from tqdm import tqdm

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config, get_memory_usage, load_dataset
from indexes.filtered_diskann import FilteredDiskANN, FilteredDiskANNPy


def get_filtered_diskann_index(
    dim: int,
    ef_construct: int = 64,
    graph_degree: int = 32,
    alpha: float = 1.5,
    filter_ef_construct: int = 64,
    ef_search: int = 64,
    index_dir: str | Path = "index",
    track_stats: bool = False,
    cache_index: bool = False,
) -> FilteredDiskANN:
    index = FilteredDiskANN(
        index_dir=str(index_dir),
        d=dim,
        ef_construct=ef_construct,
        graph_degree=graph_degree,
        alpha=alpha,
        filter_ef_construct=filter_ef_construct,
        ef_search=ef_search,
        construct_threads=16,
        search_threads=1,
        cache_index=cache_index,
    )

    if track_stats:
        print("Enabling stats tracking...")
        index.enable_stats_tracking()

    return index


def train_index(
    index: FilteredDiskANN,
    train_vecs: np.ndarray,
    train_mds: list[list[int]],
):
    index.batch_create(train_vecs, [], train_mds)


def query_index(
    index: FilteredDiskANN,
    test_vecs: np.ndarray,
    test_mds: list[list[int]],
    ground_truth: list[list[int]],
) -> tuple[float, list[dict[str, int]]]:
    results = list()
    stats = list()

    for vec, access_list in tqdm(
        zip(test_vecs, test_mds),
        total=len(test_vecs),
        desc="Querying index",
    ):
        if not access_list:
            continue

        for tenant in access_list:
            pred = index.query(vec, 10, tenant)
            results.append(pred)
            stats.append(index.get_search_stats())

    recalls = list()
    for pred, truth in zip(results, ground_truth):
        truth = [t for t in truth if t != -1]
        recalls.append(len(set(pred) & set(truth)) / len(truth))

    return np.mean(recalls).item(), stats


def exp_filtered_diskann_stats(
    ef_construct: int = 64,
    graph_degree: int = 32,
    alpha: float = 1.5,
    filter_ef_construct: int = 64,
    ef_search: int = 64,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    index_path: str | None = None,
    output_path: str = "filtered_diskann_stats.csv",
):
    print("Loading dataset...", flush=True)
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config,
    )

    print("Initializing index...", flush=True)
    index = get_filtered_diskann_index(
        dim=dim,
        ef_construct=ef_construct,
        graph_degree=graph_degree,
        alpha=alpha,
        filter_ef_construct=filter_ef_construct,
        ef_search=ef_search,
        track_stats=True,
        index_dir=index_path or "index",
        cache_index=(index_path is not None),
    )

    print("Training or loading index...", flush=True)
    train_index(index, train_vecs, train_mds)

    print("Querying index...", flush=True)
    test_vecs = test_vecs[:sample_test_size]
    test_mds = test_mds[:sample_test_size]
    recall, stats = query_index(index, test_vecs, test_mds, ground_truth.tolist())
    print(f"Recall@10: {recall:.4f}", flush=True)

    print(f"Saving results to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats).to_csv(output_path, index=False)


# this will be used to replace the profile function in the IndexProfiler class
def profile_filtered_diskann(
    self,
    index_config: IndexConfig,
    dataset_config: DatasetConfig,
    k: int = 10,
    verbose: bool = True,
    seed: int = 42,
    log_file: IO[str] | None = None,
    timeout: int | None = None,
):
    logging.info(
        "\n\n"
        "=========================================\n"
        "Index config: %s\n"
        "=========================================\n"
        "Dataset config: %s\n"
        "=========================================\n",
        index_config,
        dataset_config,
    )

    np.random.seed(seed)

    logging.info("Loading dataset...")
    train_vecs, train_mds, test_vecs, test_mds = self.loaded_dataset[1][:4]

    logging.info("Initializing index...")
    mem_before_init = get_memory_usage()
    index: FilteredDiskANN = self._initialize_index(index_config)

    logging.info("Inserting vectors...")
    index.batch_create(train_vecs, [], train_mds)
    index_size = get_memory_usage() - mem_before_init

    logging.info("Querying index...")
    query_results, query_latencies = self.run_query(
        index, k, test_vecs, test_mds, verbose, log_file, timeout
    )

    return {
        "train_latency": 0.0,
        "index_size_kb": index_size,
        "query_results": query_results,
        "insert_latencies": [0],
        "access_grant_latencies": [0],
        "query_latencies": query_latencies,
        "delete_latencies": [0],
        "revoke_access_latencies": [0],
    }


def exp_filtered_diskann(
    ef_construct_space=[32, 64, 128],
    graph_degree_space=[16, 32, 64],
    alpha_space=[1.0, 1.5, 2.0],
    filter_ef_construct_space=[32, 64, 128],
    ef_search_space=[32, 64, 128],
    construct_threads: int = 16,
    search_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    num_runs: int = 1,
    timeout: int = 600,
    output_dir: str = "output/filtered_diskann/yfcc100m",
    cache_index: bool = False,
):
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_configs = []
    for ef_construct, graph_degree, alpha, filter_ef_construct, ef_search in product(
        ef_construct_space,
        graph_degree_space,
        alpha_space,
        filter_ef_construct_space,
        ef_search_space,
    ):
        index_dir = (
            Path(output_dir)
            / f"ef{ef_construct}_m{graph_degree}_a{alpha}_fef{filter_ef_construct}.index"
        )

        index_configs.append(
            IndexConfig(
                index_cls=FilteredDiskANN,
                index_params={
                    "index_dir": str(index_dir),
                    "d": dim,
                    "ef_construct": ef_construct,
                    "graph_degree": graph_degree,
                    "alpha": alpha,
                    "filter_ef_construct": filter_ef_construct,
                    "construct_threads": construct_threads,
                    "search_threads": search_threads,
                    "cache_index": cache_index,
                },
                search_params={
                    "ef_search": ef_search,
                },
                train_params={
                    "train_ratio": 1,
                    "min_train": 50,
                    "random_seed": 42,
                },
            )
        )

    profiler = IndexProfiler()
    profiler.profile = partial(profile_filtered_diskann, profiler)
    results = profiler.batch_profile(
        index_configs, [dataset_config], num_runs=num_runs, timeout=timeout
    )

    print(f"Saving results to {output_dir} ...", flush=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_path = Path(output_dir) / "results.csv"
    df = pd.DataFrame(
        [
            {
                **config.index_params,
                **config.search_params,
                **res,
            }
            for res, config in zip(results, index_configs)
        ]
    )
    df.to_csv(results_path, index=False)

    config_path = Path(output_dir) / "config.json"
    with config_path.open("w") as f:
        json.dump(
            {
                "ef_construct_space": ef_construct_space,
                "graph_degree_space": graph_degree_space,
                "alpha_space": alpha_space,
                "filter_ef_construct_space": filter_ef_construct_space,
                "ef_search_space": ef_search_space,
                "construct_threads": construct_threads,
                "search_threads": search_threads,
                "dataset_key": dataset_key,
                "test_size": test_size,
            },
            f,
            indent=4,
        )


def exp_filtered_diskann_py(
    index_dir: str,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    search_ef: int = 128,
    verify_index: bool = False,
):
    print("Loading dataset...", flush=True)
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config,
    )

    print(f"Loading index from {index_dir} ...", flush=True)
    index = FilteredDiskANNPy.load(index_dir)

    if verify_index:
        print(f"Verifying index ...", flush=True)
        assert np.allclose(train_vecs, index.data, rtol=1e-3), "Data mismatch"
        expected_stored_mds = index.get_converted_access_lists(train_mds)
        assert len(expected_stored_mds) == len(
            index.access_lists
        ), "Access lists mismatch"
        for expected_mds, stored_mds in zip(expected_stored_mds, index.access_lists):
            if not expected_mds:
                continue
            assert set(expected_mds) == set(
                stored_mds
            ), f"Access lists mismatch: {expected_mds} != {stored_mds}"

        for label, start_point in index.label_start_points.items():
            assert label in index.access_lists[start_point], f"Label mismatch: {label}"

    print("Querying index...", flush=True)
    results = list()

    for vec, access_list in tqdm(
        zip(test_vecs[:sample_test_size], test_mds[:sample_test_size]),
        total=min(sample_test_size, len(test_vecs)),
        desc="Querying index",
    ):
        if not access_list:
            continue

        for tenant in access_list:
            pred = index.query(vec, tenant, search_ef, k=10)
            results.append(pred)

    recalls = list()
    for pred, truth in zip(results, ground_truth):
        truth = [t for t in truth if t != -1]
        recalls.append(len(set(pred) & set(truth)) / len(truth))

    print(f"Recall@10: {np.mean(recalls):.4f}", flush=True)


def exp_filter_diskann_py_search_ef(
    index_dir: str,
    search_efs: list[int] = [64, 128, 256, 512],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    output_path: str = "filtered_diskann_py_search_ef.csv",
):
    print("Loading dataset...", flush=True)
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    __, test_vecs, __, test_mds, ground_truth, __ = load_dataset(
        dataset_config,
    )

    print(f"Loading index from {index_dir} ...", flush=True)
    index = FilteredDiskANNPy.load(index_dir)

    print("Querying index...", flush=True)
    results = list()

    for search_ef in search_efs:
        preds = list()
        stats = list()

        for vec, access_list in tqdm(
            zip(test_vecs[:sample_test_size], test_mds[:sample_test_size]),
            total=min(sample_test_size, len(test_vecs)),
            desc="Querying index",
        ):
            if not access_list:
                continue

            for tenant in access_list:
                pred = index.query(vec, tenant, search_ef, k=10)
                stats.append(index.search_stats)
                preds.append(pred)

        recalls = list()
        for pred, truth in zip(preds, ground_truth):
            truth = [t for t in truth if t != -1]
            recalls.append(len(set(pred) & set(truth)) / len(truth))

        n_hops = [len(s["hops"]) for s in stats]
        n_dists = [s["n_dists"] for s in stats]

        results.append(
            {
                "search_ef": search_ef,
                "recall": np.mean(recalls).item(),
                "n_hops": np.mean(n_hops).item(),
                "n_dists": np.mean(n_dists).item(),
            }
        )

    print(f"Saving results to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def plot_filter_diskann_py_search_ef(
    results_path: str = "filtered_diskann_py_search_ef.csv",
    output_path: str = "filtered_diskann_py_search_ef.png",
):
    print("Loading results from {results_path} ...", flush=True)
    results = pd.read_csv(results_path)

    print("Plotting results ...", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    sns.lineplot(x="search_ef", y="recall", data=results, marker="o", ax=ax[0])
    ax[0].set_xlabel("search_ef")
    ax[0].set_ylabel("Recall@10")
    ax[0].set_xscale("log")

    ax2 = ax[1].twinx()
    c1, c2 = sns.color_palette("tab10")[:2]
    sns.lineplot(
        x="search_ef",
        y="n_hops",
        data=results,
        marker="o",
        label="n_hops",
        ax=ax[1],
        legend=False,
        c=c1,
    )
    sns.lineplot(
        x="search_ef",
        y="n_dists",
        data=results,
        marker="o",
        label="n_dists",
        ax=ax2,
        legend=False,
        c=c2,
    )
    ax[1].set_xlabel("search_ef")
    ax[1].set_ylabel("Number of hops")
    ax2.set_ylabel("Number of distance computations")
    ax[1].set_xscale("log")

    handles1, labels1 = ax[1].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles = handles1 + handles2
    labels = labels1 + labels2
    ax[1].legend(handles, labels, loc="upper left")

    plt.tight_layout()

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def build_filtered_diskann_index(
    index_dir: str | Path,
    ef_construct: int = 128,
    graph_degree: int = 128,
    alpha: float = 1.2,
    ef_search: int = 256,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
):
    print("Loading dataset...", flush=True)
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, __, train_mds, __, __, __ = load_dataset(dataset_config)

    print("Initializing index...", flush=True)
    index_dir = Path(index_dir) / f"ef{ef_construct}_m{graph_degree}_a{alpha}"
    index = get_filtered_diskann_index(
        dim=dim,
        ef_construct=ef_construct,
        graph_degree=graph_degree,
        alpha=alpha,
        filter_ef_construct=ef_construct,
        ef_search=ef_search,
        track_stats=True,
        index_dir=index_dir,
        cache_index=True,
    )

    print("Training or loading index...", flush=True)
    index.batch_create(train_vecs, [], train_mds)


def exp_filtered_diskann_py_alpha(
    index_dir: str,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    search_efs: list[int] = [32, 64, 128, 256, 512],
    output_path: str = "filtered_diskann_py_alpha.csv",
):
    print("Loading dataset...", flush=True)
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    __, test_vecs, __, test_mds, ground_truth, __ = load_dataset(
        dataset_config,
    )

    print(f"Loading index from {index_dir} ...", flush=True)
    index_dirs = list(Path(index_dir).iterdir())
    index_configs = []

    for idir in index_dirs:
        parsed = parse.parse("ef{ef_construct}_m{graph_degree}_a{alpha}", idir.name)
        assert isinstance(parsed, parse.Result), f"Invalid index directory: {idir}"

        index_configs.append(
            {
                "ef_construct": int(parsed["ef_construct"]),
                "graph_degree": int(parsed["graph_degree"]),
                "alpha": float(parsed["alpha"]),
            }
        )

    assert (
        len(set(c["ef_construct"] for c in index_configs)) == 1
    ), "ef_construct should be the same"
    assert (
        len(set(c["graph_degree"] for c in index_configs)) == 1
    ), "graph_degree should be the same"

    alphas = sorted([c["alpha"] for c in index_configs])

    results = list()

    for alpha in alphas:
        print(f"Evaluating index for alpha={alpha} ...", flush=True)
        idir = next(idir for idir in index_dirs if f"a{alpha}" in idir.name)
        index = FilteredDiskANNPy.load(idir)

        for search_ef in search_efs:

            preds, stats = [], []

            for vec, access_list in tqdm(
                zip(test_vecs[:sample_test_size], test_mds[:sample_test_size]),
                total=min(sample_test_size, len(test_vecs)),
                desc=f"Querying index with search_ef={search_ef}",
            ):
                if not access_list:
                    continue

                for tenant in access_list:
                    pred = index.query(vec, tenant, search_ef, k=10)
                    preds.append(pred)
                    stats.append(index.search_stats)

            recalls = list()
            for pred, truth in zip(preds, ground_truth):
                truth = [t for t in truth if t != -1]
                recalls.append(len(set(pred) & set(truth)) / len(truth))

            results.append(
                {
                    "alpha": alpha,
                    "n_edges": index.num_edges,
                    "search_ef": search_ef,
                    "recall": np.mean(recalls).item(),
                    "n_hops": np.mean([len(s["hops"]) for s in stats]).item(),
                    "n_dists": np.mean([s["n_dists"] for s in stats]).item(),
                }
            )

    print(f"Saving results to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def plot_filtered_diskann_py_alpha(
    results_path: str = "filtered_diskann_py_alpha.csv",
    output_path: str = "filtered_diskann_py_alpha.png",
):
    print(f"Loading results from {results_path} ...", flush=True)
    results = pd.read_csv(results_path)
    results["avg_degree"] = results["n_dists"] / results["n_hops"]

    print("Plotting results ...", flush=True)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    sns.lineplot(
        x="search_ef", y="recall", hue="alpha", data=results, marker="o", ax=ax[0, 0]
    )
    ax[0, 0].set_xlabel("search_ef")
    ax[0, 0].set_ylabel("Recall@10")
    ax[0, 0].set_xscale("log")

    sns.lineplot(x="alpha", y="n_edges", data=results, marker="o", ax=ax[0, 1])
    ax[0, 1].set_xlabel("alpha")
    ax[0, 1].set_ylabel("Number of edges")

    sns.lineplot(
        x="search_ef", y="n_hops", hue="alpha", data=results, marker="o", ax=ax[1, 0]
    )
    ax[1, 0].set_xlabel("search_ef")
    ax[1, 0].set_ylabel("Number of hops")
    ax[1, 0].set_xscale("log")

    sns.lineplot(
        x="search_ef",
        y="avg_degree",
        hue="alpha",
        data=results,
        marker="o",
        ax=ax[1, 1],
    )
    ax[1, 1].set_xlabel("search_ef")
    ax[1, 1].set_ylabel("Average effective degree")
    ax[1, 1].set_xscale("log")

    plt.tight_layout()

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def exp_filtered_diskann_py_graph_degree(
    index_dir: str,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    search_efs: list[int] = [32, 64, 128, 256, 512],
    output_path: str = "filtered_diskann_py_graph_degree.csv",
):
    print("Loading dataset...", flush=True)
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    __, test_vecs, __, test_mds, ground_truth, __ = load_dataset(
        dataset_config,
    )

    print(f"Loading index from {index_dir} ...", flush=True)
    index_dirs = list(Path(index_dir).iterdir())
    index_configs = []

    for idir in index_dirs:
        parsed = parse.parse("ef{ef_construct}_m{graph_degree}_a{alpha}", idir.name)
        assert isinstance(parsed, parse.Result), f"Invalid index directory: {idir}"

        index_configs.append(
            {
                "ef_construct": int(parsed["ef_construct"]),
                "graph_degree": int(parsed["graph_degree"]),
                "alpha": float(parsed["alpha"]),
            }
        )

    assert (
        len(set(c["ef_construct"] for c in index_configs)) == 1
    ), "ef_construct should be the same"
    assert len(set(c["alpha"] for c in index_configs)) == 1, "alpha should be the same"

    graph_degrees = sorted([c["graph_degree"] for c in index_configs])

    results = list()

    for graph_degree in graph_degrees:
        print(f"Evaluating index for graph_degree={graph_degree} ...", flush=True)
        idir = next(idir for idir in index_dirs if f"m{graph_degree}" in idir.name)
        index = FilteredDiskANNPy.load(idir)

        for search_ef in search_efs:

            preds, stats = [], []

            for vec, access_list in tqdm(
                zip(test_vecs[:sample_test_size], test_mds[:sample_test_size]),
                total=min(sample_test_size, len(test_vecs)),
                desc=f"Querying index with search_ef={search_ef}",
            ):
                if not access_list:
                    continue

                for tenant in access_list:
                    pred = index.query(vec, tenant, search_ef, k=10)
                    preds.append(pred)
                    stats.append(index.search_stats)

            recalls = list()
            for pred, truth in zip(preds, ground_truth):
                truth = [t for t in truth if t != -1]
                recalls.append(len(set(pred) & set(truth)) / len(truth))

            results.append(
                {
                    "graph_degree": graph_degree,
                    "n_edges": index.num_edges,
                    "search_ef": search_ef,
                    "recall": np.mean(recalls).item(),
                    "n_hops": np.mean([len(s["hops"]) for s in stats]).item(),
                    "n_dists": np.mean([s["n_dists"] for s in stats]).item(),
                }
            )

    print(f"Saving results to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def plot_filtered_diskann_py_graph_degree(
    results_path: str = "filtered_diskann_py_graph_degree.csv",
    output_path: str = "filtered_diskann_py_graph_degree.png",
):
    print(f"Loading results from {results_path} ...", flush=True)
    results = pd.read_csv(results_path)
    results["avg_degree"] = results["n_dists"] / results["n_hops"]

    print("Plotting results ...", flush=True)
    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    sns.lineplot(
        x="search_ef",
        y="recall",
        hue="graph_degree",
        data=results,
        marker="o",
        ax=ax[0, 0],
    )
    ax[0, 0].set_xlabel("search_ef")
    ax[0, 0].set_ylabel("Recall@10")
    ax[0, 0].set_xscale("log")

    sns.lineplot(x="graph_degree", y="n_edges", data=results, marker="o", ax=ax[0, 1])
    ax[0, 1].set_xlabel("graph_degree")
    ax[0, 1].set_ylabel("Number of edges")

    sns.lineplot(
        x="search_ef",
        y="n_hops",
        hue="graph_degree",
        data=results,
        marker="o",
        ax=ax[1, 0],
    )
    ax[1, 0].set_xlabel("search_ef")
    ax[1, 0].set_ylabel("Number of hops")
    ax[1, 0].set_xscale("log")

    sns.lineplot(
        x="search_ef",
        y="avg_degree",
        hue="graph_degree",
        data=results,
        marker="o",
        ax=ax[1, 1],
    )
    ax[1, 1].set_xlabel("search_ef")
    ax[1, 1].set_ylabel("Average effective degree")
    ax[1, 1].set_xscale("log")

    plt.tight_layout()

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)

    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca()
    sns.lineplot(
        x="recall", y="n_dists", data=results, marker="o", hue="graph_degree", ax=ax
    )
    ax.set_xlabel("Recall@10")
    ax.set_ylabel("#Distance computations")
    fig.tight_layout()
    fig.savefig(output_path.replace(".png", "_scatter.png"), dpi=200)


if __name__ == "__main__":
    fire.Fire()
