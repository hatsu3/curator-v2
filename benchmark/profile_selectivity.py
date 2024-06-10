import json
import pickle as pkl
import time
from dataclasses import dataclass
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from benchmark.config import IndexConfig
from benchmark.utils import get_dataset_config, load_dataset
from dataset.utils import compute_ground_truth_cuda
from indexes.hnsw_mt_hnswlib import HNSWMultiTenantHnswlib
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss


@dataclass
class Dataset:
    train_vecs: np.ndarray
    test_vecs: np.ndarray
    train_mds: list[list[int]]
    test_mds: list[list[int]]
    ground_truth: np.ndarray


def generate_synthesized_dataset(
    n_selectivities: int = 20,
    n_labels_per_selectivity: int = 10,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    seed: int = 42,
    output_dir: str = "data/selectivity/random_yfcc100m",
):
    print("Loading base dataset...", flush=True)
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, __, __, __, __ = load_dataset(dataset_config)

    print("Generating synthesized dataset...", flush=True)
    np.random.seed(seed)

    train_mds = [[] for _ in train_vecs]
    test_mds = [[] for _ in test_vecs]
    label_to_selectivity = {}
    cur_label = 0

    for selectivity in tqdm(
        np.linspace(0.01, 1.00, n_selectivities), total=n_selectivities
    ):
        for _ in range(n_labels_per_selectivity):
            train_mask = np.random.rand(train_vecs.shape[0]) < selectivity
            test_mask = np.random.rand(test_vecs.shape[0]) < selectivity

            for id in np.where(train_mask)[0]:
                train_mds[id].append(cur_label)

            for id in np.where(test_mask)[0]:
                test_mds[id].append(cur_label)

            label_to_selectivity[cur_label] = selectivity
            cur_label += 1

    print("Computing ground truth...", flush=True)
    ground_truth = compute_ground_truth_cuda(
        train_vecs,
        train_mds,
        test_vecs,
        test_mds,
        set(range(cur_label)),
    )

    print("Total number of labels:", cur_label, flush=True)
    print(
        "Total number of (vector, label) pairs:",
        sum(len(md) for md in train_mds),
        flush=True,
    )

    print(f"Saving synthesized dataset to {output_dir} ...", flush=True)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(output_dir) / "ground_truth.npy", ground_truth)
    pkl.dump(
        {
            "train_mds": train_mds,
            "test_mds": test_mds,
        },
        open(Path(output_dir) / "access_lists.pkl", "wb"),
    )
    pd.DataFrame(label_to_selectivity, index=[0]).to_csv(
        Path(output_dir) / "label_to_selectivity.csv", index=False
    )
    json.dump(
        {
            "n_selectivities": n_selectivities,
            "n_labels_per_selectivity": n_labels_per_selectivity,
            "dataset_key": dataset_key,
            "test_size": test_size,
            "seed": seed,
        },
        open(Path(output_dir) / "meta.json", "w"),
        indent=4,
    )


def load_synthesized_dataset(
    dataset_dir: str = "data/selectivity/random_yfcc100m",
    verbose: bool = True,
) -> Dataset:
    if verbose:
        print("Loading base dataset...", flush=True)

    meta = json.load(open(Path(dataset_dir) / "meta.json"))
    dataset_config, __ = get_dataset_config(
        meta["dataset_key"], test_size=meta["test_size"]
    )
    train_vecs, test_vecs, *__ = load_dataset(dataset_config)

    if verbose:
        print("Loading access lists...", flush=True)

    access_lists = pkl.load(open(Path(dataset_dir) / "access_lists.pkl", "rb"))
    train_mds, test_mds = access_lists["train_mds"], access_lists["test_mds"]

    if verbose:
        print("Loading ground truth...", flush=True)

    ground_truth = np.load(Path(dataset_dir) / "ground_truth.npy")

    return Dataset(
        train_vecs=train_vecs,
        test_vecs=test_vecs,
        train_mds=train_mds,
        test_mds=test_mds,
        ground_truth=ground_truth,
    )


def get_curator_index(
    dim: int,
    nlist: int = 16,
    max_sl_size: int = 128,
    nprobe: int = 4000,
    prune_thres: float = 1.6,
    variance_boost: float = 0.4,
) -> IVFFlatMultiTenantBFHierFaiss:
    index_config = IndexConfig(
        index_cls=IVFFlatMultiTenantBFHierFaiss,
        index_params={
            "d": dim,
            "nlist": nlist,
            "max_sl_size": max_sl_size,
            "max_leaf_size": max_sl_size,
            "bf_capacity": 1000,
            "bf_error_rate": 0.01,
            "update_bf_interval": 100,
            "clus_niter": 20,
        },
        search_params={
            "nprobe": nprobe,
            "prune_thres": prune_thres,
            "variance_boost": variance_boost,
        },
        train_params={
            "train_ratio": 1,
            "min_train": 50,
            "random_seed": 42,
        },
    )

    index = index_config.index_cls(
        **index_config.index_params, **index_config.search_params
    )
    assert isinstance(index, IVFFlatMultiTenantBFHierFaiss)

    return index


def get_shared_hnsw_index(
    construct_ef: int = 32,
    m: int = 32,
    search_ef: int = 32,
    max_elements: int = 1000000,
) -> HNSWMultiTenantHnswlib:
    index_config = IndexConfig(
        index_cls=HNSWMultiTenantHnswlib,
        index_params={
            "construction_ef": construct_ef,
            "m": m,
            "search_ef": search_ef,
            "num_threads": 1,
            "max_elements": max_elements,
        },
        search_params={},
        train_params={},
    )

    index = index_config.index_cls(
        **index_config.index_params, **index_config.search_params
    )
    assert isinstance(index, HNSWMultiTenantHnswlib)

    return index


def eval_curator(
    dataset: Dataset,
    nlist: int = 16,
    max_sl_size: int = 128,
    prune_threses: list[float] = [1.2, 1.4, 1.6, 1.8, 2.0],
    verbose: bool = False,
) -> list[dict]:
    if verbose:
        print(
            f"Training curator index with nlist={nlist}, max_sl_size={max_sl_size}...",
            flush=True,
        )

    index = get_curator_index(dataset.train_vecs.shape[1], nlist, max_sl_size)
    index.enable_stats_tracking()

    index.train(dataset.train_vecs)

    for i, (vec, access_list) in tqdm(
        enumerate(zip(dataset.train_vecs, dataset.train_mds)),
        total=len(dataset.train_vecs),
        desc="Building index",
        disable=not verbose,
    ):
        if not access_list:
            continue

        index.create(vec, i, access_list[0])
        for tenant in access_list[1:]:
            index.grant_access(i, tenant)

    if verbose:
        print("Evaluating curator index...", flush=True)

    results = []
    for prune_thres in prune_threses:
        index.search_params = {"prune_thres": prune_thres}

        preds, latencies, n_dists = [], [], []
        for vec, access_list in tqdm(
            zip(dataset.test_vecs, dataset.test_mds),
            total=len(dataset.test_vecs),
            desc=f"Querying index with prune_thres={prune_thres:.2f}",
            disable=not verbose,
        ):
            for tenant in access_list:
                start = time.time()
                pred = index.query(vec, k=10, tenant_id=tenant)
                latencies.append(time.time() - start)
                n_dists.append(index.get_search_stats()["n_dists"])
                preds.append(pred)

        recalls = []
        for pred, truth in zip(preds, dataset.ground_truth):
            truth = [t for t in truth if t != -1]
            recalls.append(len(set(pred) & set(truth)) / len(truth))

        results.append(
            {
                "nlist": nlist,
                "max_sl_size": max_sl_size,
                "prune_thres": prune_thres,
                "recall": np.mean(recalls).item(),
                "latency": np.mean(latencies).item(),
                "n_dists": np.mean(n_dists).item(),
            }
        )

    return results


def eval_curator_worker(args):
    dataset_dir, *args = args
    dataset = load_synthesized_dataset(dataset_dir, verbose=False)
    return eval_curator(dataset, *args, verbose=False)


def eval_curator_sweep(
    nlists: list[int] = [8, 16, 32],
    max_sl_sizes: list[int] = [64, 128, 256],
    prune_threses: list[float] = [1.2, 1.4, 1.6, 1.8, 2.0],
    dataset_dir: str = "data/selectivity/random_yfcc100m",
    output_path: str = "profile_selectivity_curator_sweep.csv",
    num_workers: int = 8,
):
    tasks = [
        (dataset_dir, nlist, max_sl_size, prune_threses)
        for nlist, max_sl_size in product(nlists, max_sl_sizes)
    ]

    with Pool(num_workers) as p:
        results = list(
            tqdm(
                p.imap_unordered(eval_curator_worker, tasks),
                total=len(tasks),
                desc="Evaluating curator index",
            )
        )

    print(f"Saving results to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def eval_shared_hnsw(
    dataset: Dataset,
    construct_ef: int = 32,
    m: int = 32,
    search_efs: list[int] = [32, 64, 128, 256],
    verbose: bool = False,
) -> list[dict]:
    if verbose:
        print(
            f"Training shared HNSW index with construct_ef={construct_ef}, m={m}...",
            flush=True,
        )

    index = get_shared_hnsw_index(construct_ef, m)

    for i, (vec, access_list) in tqdm(
        enumerate(zip(dataset.train_vecs, dataset.train_mds)),
        total=len(dataset.train_vecs),
        desc="Building index",
        disable=not verbose,
    ):
        if not access_list:
            continue

        index.create(vec, i, access_list[0])
        for tenant in access_list[1:]:
            index.grant_access(i, tenant)

    if verbose:
        print("Evaluating shared HNSW index...", flush=True)

    results = []
    for search_ef in search_efs:
        index.search_params = {"search_ef": search_ef}

        preds, latencies, n_dists = [], [], []
        for vec, access_list in tqdm(
            zip(dataset.test_vecs, dataset.test_mds),
            total=len(dataset.test_vecs),
            desc=f"Querying index with search_ef={search_ef}",
            disable=not verbose,
        ):
            for tenant_id in access_list:
                start = time.time()
                pred = index.query(vec, k=10, tenant_id=tenant_id)
                latencies.append(time.time() - start)
                n_dists.append(index.get_search_stats()["n_dists"])
                preds.append(pred)

        recalls = []
        for pred, truth in zip(preds, dataset.ground_truth):
            truth = [t for t in truth if t != -1]
            recalls.append(len(set(pred) & set(truth)) / len(truth))

        results.append(
            {
                "construct_ef": construct_ef,
                "m": m,
                "search_ef": search_ef,
                "recall": np.mean(recalls).item(),
                "latency": np.mean(latencies).item(),
                "n_dists": np.mean(n_dists).item(),
            }
        )

    return results


def eval_shared_hnsw_worker(args):
    dataset_dir, *args = args
    dataset = load_synthesized_dataset(dataset_dir, verbose=False)
    return eval_shared_hnsw(dataset, *args, verbose=False)


def eval_shared_hnsw_sweep(
    construct_efs: list[int] = [16, 32, 64, 128],
    ms: list[int] = [16, 32, 64, 128],
    search_efs: list[int] = [16, 32, 64, 128, 256],
    dataset_dir: str = "data/selectivity/random_yfcc100m",
    output_path: str = "profile_selectivity_shared_hnsw_sweep.csv",
    num_workers: int = 8,
):
    tasks = [
        (dataset_dir, construct_ef, m, search_efs)
        for construct_ef, m in product(construct_efs, ms)
    ]

    with Pool(num_workers) as p:
        results = list(
            tqdm(
                p.imap_unordered(eval_shared_hnsw_worker, tasks),
                total=len(tasks),
                desc="Evaluating shared HNSW index",
            )
        )

    print(f"Saving results to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def plot_sweep_results(
    results_path: str = "profile_selectivity_curator_sweep.csv",
    output_path: str = "profile_selectivity_curator_sweep.png",
):
    print(f"Loading results from {results_path} ...", flush=True)
    results = pd.read_csv(results_path)

    print("Plotting results...", flush=True)
    sns.lineplot(
        data=results,
        x="recall",
        y="n_dists",
        hue="nlist",
        style="max_sl_size",
        markers=True,
        dashes=False,
    )
    plt.xlabel("Recall@10")
    plt.ylabel("# Distances Computed")

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def select_curator_best_config(
    recall_thres: float = 0.9,
    results_path: str = "profile_selectivity_curator_sweep.csv",
    output_path: str = "profile_selectivity_curator_best_config.json",
):
    print(f"Loading results from {results_path} ...", flush=True)
    results = pd.read_csv(results_path)

    print(f"Selecting best config with recall > {recall_thres} ...", flush=True)
    results = results[results["recall"] > recall_thres]
    results = results.groupby(["nlist", "max_sl_size"]).min().reset_index()
    results = results.drop(columns=["prune_thres", "recall", "latency"])
    best_config = results.loc[results["n_dists"].idxmin()].to_dict()
    print(f"Best config: {best_config}", flush=True)

    print(f"Saving best config to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(best_config, open(output_path, "w"), indent=4)


def select_shared_hnsw_best_config(
    recall_thres: float = 0.9,
    results_path: str = "profile_selectivity_shared_hnsw_sweep.csv",
    output_path: str = "profile_selectivity_shared_hnsw_best_config.json",
):
    print(f"Loading results from {results_path} ...", flush=True)
    results = pd.read_csv(results_path)

    print(f"Selecting best config with recall > {recall_thres} ...", flush=True)
    results = results[results["recall"] > recall_thres]
    results = results.groupby(["construct_ef", "m"]).min().reset_index()
    results = results.drop(columns=["search_ef", "recall", "latency"])
    best_config = results.loc[results["n_dists"].idxmin()].to_dict()
    print(f"Best config: {best_config}", flush=True)

    print(f"Saving best config to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(best_config, open(output_path, "w"), indent=4)


# profile curator and hnsw using the best parameters
# measure per-query ndists and latency and group by selectivity of query filter
def profile_curator_ndists_vs_selectivity(
    prune_threses: list[float] = [1.2, 1.4, 1.6, 1.8, 2.0],
    best_config_path: str = "profile_selectivity_curator_best_config.json",
    dataset_dir: str = "data/selectivity/random_yfcc100m",
    output_path: str = "profile_selectivity_curator_ndists_vs_selectivity.csv",
):
    print(f"Loading dataset from {dataset_dir} ...", flush=True)
    label_to_sel = pd.read_csv(Path(dataset_dir) / "label_to_selectivity.csv")
    label_to_sel = label_to_sel.to_dict(orient="records")[0]
    dataset = load_synthesized_dataset(dataset_dir, verbose=True)
    dim = dataset.train_vecs.shape[1]

    print(
        f"Creating curator index with best config from {best_config_path} ...",
        flush=True,
    )
    best_config = json.load(open(best_config_path))
    index = get_curator_index(dim, best_config["nlist"], best_config["max_sl_size"])

    print("Training curator index...", flush=True)
    index.train(dataset.train_vecs)

    for i, (vec, access_list) in tqdm(
        enumerate(zip(dataset.train_vecs, dataset.train_mds)),
        total=len(dataset.train_vecs),
        desc="Building index",
    ):
        if not access_list:
            continue

        index.create(vec, i, access_list[0])
        for tenant in access_list[1:]:
            index.grant_access(i, tenant)

    print("Evaluating curator index...", flush=True)
    results = []

    for prune_thres in prune_threses:
        ground_truth = iter(dataset.ground_truth)

        for vec, access_list in tqdm(
            zip(dataset.test_vecs, dataset.test_mds),
            total=len(dataset.test_vecs),
            desc=f"Querying index with prune_thres={prune_thres:.2f}",
        ):
            for tenant in access_list:
                start = time.time()
                pred = index.query(vec, k=10, tenant_id=tenant)
                latency = time.time() - start

                n_dists = index.get_search_stats()["n_dists"]
                gt = next(ground_truth)
                recall = len(set(pred) & set(gt)) / len(gt)

                results.append(
                    {
                        "selectivity": label_to_sel[tenant],
                        "prune_thres": prune_thres,
                        "recall": recall,
                        "latency": latency,
                        "n_dists": n_dists,
                    }
                )

    print(f"Saving results to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def profile_shared_hnsw_ndists_vs_selectivity(
    search_efs: list[int] = [16, 32, 64, 128, 256],
    best_config_path: str = "profile_selectivity_shared_hnsw_best_config.json",
    dataset_dir: str = "data/selectivity/random_yfcc100m",
    output_path: str = "profile_selectivity_shared_hnsw_ndists_vs_selectivity.csv",
):
    print(f"Loading dataset from {dataset_dir} ...", flush=True)
    label_to_sel = pd.read_csv(Path(dataset_dir) / "label_to_selectivity.csv")
    label_to_sel = label_to_sel.to_dict(orient="records")[0]
    dataset = load_synthesized_dataset(dataset_dir, verbose=True)

    print(
        f"Creating shared HNSW index with best config from {best_config_path} ...",
        flush=True,
    )
    best_config = json.load(open(best_config_path))
    index = get_shared_hnsw_index(best_config["construct_ef"], best_config["m"])

    for i, (vec, access_list) in tqdm(
        enumerate(zip(dataset.train_vecs, dataset.train_mds)),
        total=len(dataset.train_vecs),
        desc="Building index",
    ):
        if not access_list:
            continue

        index.create(vec, i, access_list[0])
        for tenant in access_list[1:]:
            index.grant_access(i, tenant)

    print("Evaluating shared HNSW index...", flush=True)
    results = []

    for search_ef in search_efs:
        ground_truth = iter(dataset.ground_truth)

        for vec, access_list in tqdm(
            zip(dataset.test_vecs, dataset.test_mds),
            total=len(dataset.test_vecs),
            desc=f"Querying index with search_ef={search_ef}",
        ):
            for tenant in access_list:
                start = time.time()
                pred = index.query(vec, k=10, tenant_id=tenant)
                latency = time.time() - start

                n_dists = index.get_search_stats()["n_dists"]
                gt = next(ground_truth)
                recall = len(set(pred) & set(gt)) / len(gt)

                results.append(
                    {
                        "selectivity": label_to_sel[tenant],
                        "search_ef": search_ef,
                        "recall": recall,
                        "latency": latency,
                        "n_dist": n_dists,
                    }
                )

    print(f"Saving results to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def plot_ndists_vs_selectivity(
    curator_results_path: str = "profile_selectivity_curator_ndists_vs_selectivity.csv",
    shared_hnsw_results_path: str = "profile_selectivity_shared_hnsw_ndists_vs_selectivity.csv",
    output_path: str = "profile_selectivity_ndists_vs_selectivity.png",
):
    print(f"Loading results from {curator_results_path} ...", flush=True)
    curator_results = pd.read_csv(curator_results_path)

    print(f"Loading results from {shared_hnsw_results_path} ...", flush=True)
    shared_hnsw_results = pd.read_csv(shared_hnsw_results_path)

    print("Plotting results...", flush=True)
    ...


if __name__ == "__main__":
    fire.Fire()
