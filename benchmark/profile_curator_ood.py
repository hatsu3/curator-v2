import pickle as pkl
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.profile_curator_py import get_curator_index as get_curator_index_py
from benchmark.profile_curator_py import train_index
from benchmark.utils import get_dataset_config, load_dataset, recall
from dataset.utils import compute_ground_truth_cuda
from indexes.base import Index
from indexes.curator_py import CuratorIndexPy
from indexes.hnsw_mt_hnswlib import HNSWMultiTenantHnswlib
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss


def run_query(
    index: Index,
    train_vecs: np.ndarray,
    test_vecs: np.ndarray,
    train_mds: list[list[int]],
    test_mds: list[list[int]],
    ground_truth: np.ndarray,
):
    # train index
    print("Training index...")
    try:
        index.train(train_vecs, train_mds)
    except NotImplementedError:
        print("Training not necessary, skipping...")

    # insert vectors into index
    for i, (vec, mds) in tqdm(
        enumerate(zip(train_vecs, train_mds)),
        total=len(train_vecs),
        desc="Building index",
    ):
        if not mds:
            continue

        index.create(vec, i, mds[0])
        for md in mds[1:]:
            index.grant_access(i, md)

    # query index
    query_results = list()
    query_latencies = list()
    for vec, tenant_ids in tqdm(
        zip(test_vecs, test_mds),
        total=len(test_vecs),
        desc="Querying index",
    ):
        for tenant_id in tenant_ids:
            start = time.time()
            ids = index.query(vec, k=10, tenant_id=int(tenant_id))
            query_latencies.append(time.time() - start)
            query_results.append(ids)

    recalls = list()
    for res, gt in zip(query_results, ground_truth):
        recalls.append(recall([res], gt[None]))

    return query_latencies, recalls


def generate_ood_dataset(
    train_vecs: np.ndarray,
    train_mds: list[list[int]],
    test_vecs: np.ndarray,
    test_mds: list[list[int]],
    sampled_cates: set[int],
    seed: int = 42,
    batch_size: int = 8,
):
    np.random.seed(seed)

    sampled_indices = np.random.permutation(test_vecs.shape[0])
    test_vecs = test_vecs[sampled_indices]

    ground_truth = compute_ground_truth_cuda(
        train_vecs,
        train_mds,
        test_vecs,
        test_mds,
        sampled_cates,
        batch_size=batch_size,
    )

    return train_vecs, train_mds, test_vecs, test_mds, sampled_indices, ground_truth


def generate_ood_dataset_v2(
    train_vecs: np.ndarray,
    train_mds: list[list[int]],
    test_vecs: np.ndarray,
    test_mds: list[list[int]],
    sampled_cates: set[int],
    seed: int = 42,
    batch_size: int = 8,
):
    from sklearn.cluster import KMeans

    np.random.seed(seed)

    all_vecs = np.concatenate([train_vecs, test_vecs], axis=0)
    all_mds = train_mds + test_mds

    kmeans = KMeans(n_clusters=2, random_state=seed)
    cluster_labels = kmeans.fit_predict(all_vecs)

    train_vecs_new = all_vecs[cluster_labels == 0]
    train_mds_new = [mds for i, mds in enumerate(all_mds) if cluster_labels[i] == 0]
    train_vecs_indices = np.arange(all_vecs.shape[0])[cluster_labels == 0]

    test_vecs_new = all_vecs[cluster_labels == 1]
    test_mds_new = [mds for i, mds in enumerate(all_mds) if cluster_labels[i] == 1]
    test_vecs_indices = np.arange(all_vecs.shape[0])[cluster_labels == 1]

    sampled_indices = np.random.permutation(test_vecs_new.shape[0])[: len(test_vecs)]
    test_vecs_new = test_vecs_new[sampled_indices]
    test_mds_new = [mds for i, mds in enumerate(test_mds_new) if i in sampled_indices]
    test_vecs_indices = test_vecs_indices[sampled_indices]

    ground_truth = compute_ground_truth_cuda(
        train_vecs_new,
        train_mds_new,
        test_vecs_new,
        test_mds_new,
        sampled_cates,
        batch_size=batch_size,
    )

    return (
        train_vecs_new,
        train_mds_new,
        test_vecs_new,
        test_mds_new,
        (train_vecs_indices, test_vecs_indices),
        ground_truth,
    )


def load_or_generate_ood_dataset(
    dataset_config: DatasetConfig,
    dataset_cache_path: str,
    ood_version: str = "v1",
):
    train_vecs, test_vecs, train_mds, test_mds, __, train_cates = load_dataset(
        dataset_config
    )

    if Path(dataset_cache_path).exists():
        print(f"Loading cached dataset from {dataset_cache_path}...", flush=True)
        if ood_version == "v1":
            dataset = np.load(dataset_cache_path)
            sampled_indices = dataset["sampled_indices"]
            test_vecs = test_vecs[sampled_indices]
            ground_truth = dataset["ground_truth"]
        else:
            # sampled_indices_train = dataset["sampled_indices_train"]
            # sampled_indices_test = dataset["sampled_indices_test"]
            # all_vecs = np.concatenate([train_vecs, test_vecs], axis=0)
            # all_mds = train_mds + test_mds
            # train_vecs = all_vecs[sampled_indices_train]
            # train_mds = [all_mds[i] for i in sampled_indices_train]
            # test_vecs = all_vecs[sampled_indices_test]
            # test_mds = [all_mds[i] for i in sampled_indices_test]
            # ground_truth = dataset["ground_truth"]
            dataset = pkl.load(open(dataset_cache_path, "rb"))
            train_vecs = dataset["train_vecs"]
            train_mds = dataset["train_mds"]
            test_vecs = dataset["test_vecs"]
            test_mds = dataset["test_mds"]
            ground_truth = dataset["ground_truth"]
    elif ood_version == "v1":
        train_vecs, train_mds, test_vecs, test_mds, sampled_indices, ground_truth = (
            generate_ood_dataset(
                train_vecs,
                train_mds,
                test_vecs,
                test_mds,
                train_cates,
            )
        )
        test_vecs = test_vecs[sampled_indices]

        print(f"Saving dataset to {dataset_cache_path}...", flush=True)
        Path(dataset_cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            dataset_cache_path,
            sampled_indices=sampled_indices,
            ground_truth=ground_truth,
            version=1,
        )
    else:
        train_vecs, train_mds, test_vecs, test_mds, sampled_indices, ground_truth = (
            generate_ood_dataset_v2(
                train_vecs,
                train_mds,
                test_vecs,
                test_mds,
                train_cates,
            )
        )

        print(f"Saving dataset to {dataset_cache_path}...", flush=True)
        Path(dataset_cache_path).parent.mkdir(parents=True, exist_ok=True)
        pkl.dump(
            {
                "train_vecs": train_vecs,
                "train_mds": train_mds,
                "test_vecs": test_vecs,
                "test_mds": test_mds,
                "ground_truth": ground_truth,
                "version": 2,
            },
            open(dataset_cache_path, "wb"),
        )

    return train_vecs, test_vecs, train_mds, test_mds, ground_truth


def get_curator_index(
    dim: int,
    nlist: int = 16,
    prune_thres: float = 1.6,
    max_sl_size: int = 256,
    max_leaf_size: int = 128,
    nprobe: int = 4000,
    variance_boost: float = 0.4,
) -> IVFFlatMultiTenantBFHierFaiss:
    index_config = IndexConfig(
        index_cls=IVFFlatMultiTenantBFHierFaiss,
        index_params={
            "d": dim,
            "nlist": nlist,
            "bf_capacity": 1000,
            "bf_error_rate": 0.01,
            "max_sl_size": max_sl_size,
            "max_leaf_size": max_leaf_size,
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
    construction_ef: int = 32,
    search_ef: int = 16,
    m: int = 32,
) -> HNSWMultiTenantHnswlib:
    index_config = IndexConfig(
        index_cls=HNSWMultiTenantHnswlib,
        index_params={
            "construction_ef": construction_ef,
            "m": m,
            "max_elements": 2000000,
        },
        search_params={
            "search_ef": search_ef,
        },
        train_params=None,
    )

    index = index_config.index_cls(
        **index_config.index_params, **index_config.search_params
    )
    assert isinstance(index, HNSWMultiTenantHnswlib)

    return index


def exp_curator_ood(
    nlist: int = 16,
    prune_thres: float = 1.6,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ood_version: str = "v1",
    dataset_cache_path: str = "yfcc100m_ood.npz",
    output_path: str = "curator_ood.pkl",
):
    print(f"Running experiment for original dataset...", flush=True)
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config
    )
    index = get_curator_index(
        dim=train_vecs.shape[1], nlist=nlist, prune_thres=prune_thres
    )
    latencies, recalls = run_query(
        index, train_vecs, test_vecs, train_mds, test_mds, ground_truth
    )

    print(f"Running experiment for OOD dataset...", flush=True)
    print(f"Loading OOD dataset...", flush=True)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth = (
        load_or_generate_ood_dataset(dataset_config, dataset_cache_path, ood_version)
    )
    index = get_curator_index(
        dim=train_vecs.shape[1], nlist=nlist, prune_thres=prune_thres
    )
    ood_latencies, ood_recalls = run_query(
        index, train_vecs, test_vecs, train_mds, test_mds, ground_truth
    )

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump(
        {
            "latencies": latencies,
            "recalls": recalls,
            "ood_latencies": ood_latencies,
            "ood_recalls": ood_recalls,
            "nlist": nlist,
            "prune_thres": prune_thres,
            "dataset_key": dataset_key,
            "test_size": test_size,
        },
        open(output_path, "wb"),
    )


def exp_curator_ood_py(
    index_path: str = "curator_py_yfcc100m.index",
    ood_index_path: str = "curator_py_yfcc100m_ood.index",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    ood_version: str = "v2",
    dataset_cache_path: str = "yfcc100m_ood_v2.pkl",
    output_path: str = "curator_py_ood.pkl",
):
    def query_curator_py(index, test_vecs, test_mds, ground_truth):
        results = list()
        ndists = list()
        for vec, access_list in tqdm(
            zip(test_vecs, test_mds),
            total=len(test_vecs),
            desc="Querying index",
        ):
            if not access_list:
                continue

            for tenant in access_list:
                pred, stats = index.search(vec, 10, tenant, return_stats=True)
                results.append(pred)
                ndists.append(stats["ndists"])

        recalls = list()
        for pred, truth in zip(results, ground_truth):
            truth = [t for t in truth if t != -1]
            recalls.append(len(set(pred) & set(truth)) / len(truth))

        return ndists, recalls

    def load_or_train_index(index_path, train_vecs, train_mds):
        if Path(index_path).exists():
            print(f"Loading index from {index_path}...", flush=True)
            index = CuratorIndexPy.load(index_path)
        else:
            print(f"Initializing index...", flush=True)
            index = get_curator_index_py(dim=train_vecs.shape[1])

            print(f"Training index...", flush=True)
            train_index(index, train_vecs, train_mds)

            print(f"Saving index to {index_path}...", flush=True)
            index.save(index_path)

        return index

    print(f"Running experiment for original dataset...", flush=True)
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config
    )
    test_vecs = test_vecs[:sample_test_size]
    test_mds = test_mds[:sample_test_size]

    index = load_or_train_index(index_path, train_vecs, train_mds)
    ndists, recalls = query_curator_py(index, test_vecs, test_mds, ground_truth)

    print(f"Running experiment for OOD dataset...", flush=True)
    print(f"Loading OOD dataset...", flush=True)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth = (
        load_or_generate_ood_dataset(dataset_config, dataset_cache_path, ood_version)
    )
    test_vecs = test_vecs[:sample_test_size]
    test_mds = test_mds[:sample_test_size]

    index = load_or_train_index(ood_index_path, train_vecs, train_mds)
    ood_ndists, ood_recalls = query_curator_py(index, test_vecs, test_mds, ground_truth)

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump(
        {
            "ndists": ndists,
            "recalls": recalls,
            "ood_ndists": ood_ndists,
            "ood_recalls": ood_recalls,
            "dataset_key": dataset_key,
            "test_size": test_size,
        },
        open(output_path, "wb"),
    )


def exp_shared_hnsw_ood(
    construction_ef: int = 32,
    search_ef: int = 16,
    m: int = 32,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ood_version: str = "v1",
    dataset_cache_path: str = "yfcc100m_ood.npz",
    output_path: str = "shared_hnsw_ood.pkl",
):
    print(f"Running experiment for original dataset...", flush=True)
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config
    )
    index = get_shared_hnsw_index(
        construction_ef=construction_ef, search_ef=search_ef, m=m
    )
    latencies, recalls = run_query(
        index, train_vecs, test_vecs, train_mds, test_mds, ground_truth
    )

    print(f"Running experiment for OOD dataset...", flush=True)
    print(f"Loading OOD dataset...", flush=True)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth = (
        load_or_generate_ood_dataset(dataset_config, dataset_cache_path, ood_version)
    )
    index = get_shared_hnsw_index(
        construction_ef=construction_ef, search_ef=search_ef, m=m
    )
    ood_latencies, ood_recalls = run_query(
        index, train_vecs, test_vecs, train_mds, test_mds, ground_truth
    )

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump(
        {
            "latencies": latencies,
            "recalls": recalls,
            "ood_latencies": ood_latencies,
            "ood_recalls": ood_recalls,
            "construction_ef": construction_ef,
            "search_ef": search_ef,
            "m": m,
            "dataset_key": dataset_key,
            "test_size": test_size,
        },
        open(output_path, "wb"),
    )


def plot_ood_dataset_pca(
    tenant_id: int,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ood_version: str = "v1",
    dataset_cache_path: str = "yfcc100m_ood.npz",
    output_path: str = "ood_dataset.png",
):
    from sklearn.decomposition import PCA

    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, __ = load_or_generate_ood_dataset(
        dataset_config, dataset_cache_path, ood_version
    )

    print("Projecting vectors to 2D...", flush=True)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(train_vecs)
    train_vecs_2d = pca.transform(train_vecs)
    test_vecs_2d = pca.transform(test_vecs)

    print(f"Filtering vectors for tenant {tenant_id}...", flush=True)
    train_mask = np.array([tenant_id in mds for mds in train_mds])
    test_mask = np.array([tenant_id in mds for mds in test_mds])
    train_vecs_2d_masked = train_vecs_2d[train_mask]
    test_vecs_2d_masked = test_vecs_2d[test_mask]

    print("Generating scatterplot...", flush=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # ax.scatter(train_vecs_2d[:, 0], train_vecs_2d[:, 1], color="gray", alpha=0.2, s=1)
    ax.scatter(
        train_vecs_2d_masked[:, 0], train_vecs_2d_masked[:, 1], color="blue", s=1
    )
    ax.scatter(test_vecs_2d_masked[:, 0], test_vecs_2d_masked[:, 1], color="red")
    plt.tight_layout()

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_ood_dataset_mahalanobis_dist(
    tenant_id: int,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ood_version: str | None = None,
    dataset_cache_path: str = "yfcc100m_ood.npz",
    output_path: str = "ood_dataset_mahalanobis.png",
):
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)

    if ood_version is None:
        train_vecs, test_vecs, train_mds, test_mds, __, __ = load_dataset(
            dataset_config
        )
    else:
        train_vecs, test_vecs, train_mds, test_mds, __ = load_or_generate_ood_dataset(
            dataset_config, dataset_cache_path, ood_version
        )

    print(f"Filtering vectors for tenant {tenant_id}...", flush=True)
    train_mask = np.array([tenant_id in mds for mds in train_mds])
    test_mask = np.array([tenant_id in mds for mds in test_mds])
    train_vecs_masked = train_vecs[train_mask]
    test_vecs_masked = test_vecs[test_mask]

    print("Computing average Mahalanobis distances...", flush=True)

    def mahalanobis_dist(Q, X):
        X_mean = np.mean(X, axis=0)
        X_cov = np.cov(X, rowvar=False)
        X_cov_inv = np.linalg.inv(X_cov)
        Q_diff = Q - X_mean
        return np.sqrt(np.einsum("ij,jk,ik->i", Q_diff, X_cov_inv, Q_diff))

    test_train_dists = mahalanobis_dist(test_vecs_masked, train_vecs_masked)
    train_train_dists = mahalanobis_dist(train_vecs_masked, train_vecs_masked)

    def plot_cdf(data, ax, color, label):
        data_sorted = np.sort(data)
        cdf = np.arange(1, len(data) + 1) / len(data)
        sns.lineplot(x=data_sorted, y=cdf, ax=ax, color=color, label=label)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    c1, c2 = sns.color_palette()[:2]
    plot_cdf(test_train_dists, ax, c1, "Test-Train")
    plot_cdf(train_train_dists, ax, c2, "Train-Train")
    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("CDF")
    plt.tight_layout()

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_ood_results(
    latency_ms_cap: float = 4.0,
    results_path: str = "curator_ood.pkl",
    output_path: str = "curator_ood.png",
):
    results = pkl.load(open(results_path, "rb"))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    latencies_df = pd.DataFrame(
        {
            "latency": results["latencies"] + results["ood_latencies"],
            "type": ["Original"] * len(results["latencies"])
            + ["OOD"] * len(results["ood_latencies"]),
        }
    )
    latencies_df["latency"] *= 1000
    # latencies_df["latency_bin"] = pd.cut(
    #     latencies_df["latency"], bins=40, labels=[f"{i/10}" for i in range(40)]
    # )
    # latencies_df = latencies_df.groupby(["type", "latency_bin"]).size().reset_index(name="count")  # type: ignore
    # latencies_df["cumulative_count"] = latencies_df.groupby("type")["count"].cumsum()
    # latencies_df["cdf"] = latencies_df["cumulative_count"] / latencies_df.groupby(
    #     "type"
    # )["count"].transform("sum")

    recalls_df = pd.DataFrame(
        {
            "recall": results["recalls"] + results["ood_recalls"],
            "type": ["Original"] * len(results["recalls"])
            + ["OOD"] * len(results["ood_recalls"]),
        }
    )
    recalls_df["recall_bin"] = pd.cut(
        recalls_df["recall"], bins=10, labels=[f"{i/10}" for i in range(10)]
    )
    recalls_df = recalls_df.groupby(["type", "recall_bin"]).size().reset_index(name="count")  # type: ignore
    recalls_df["cumulative_count"] = recalls_df.groupby("type")["count"].cumsum()
    recalls_df["cdf"] = recalls_df["cumulative_count"] / recalls_df.groupby("type")[
        "count"
    ].transform("sum")

    sns.kdeplot(
        data=latencies_df,
        x="latency",
        hue="type",
        ax=ax[0],
        hue_order=["Original", "OOD"],
    )
    # sns.lineplot(
    #     data=latencies_df,
    #     x="latency_bin",
    #     y="cdf",
    #     hue="type",
    #     ax=ax[0],
    #     hue_order=["Original", "OOD"],
    # )
    # ax[0].set_xticks([0, 10, 20, 30, 40])

    # sns.barplot(data=recalls_df, x="recall_bin", y="count", hue="type", ax=ax[1])
    sns.lineplot(
        data=recalls_df,
        x="recall_bin",
        y="cdf",
        hue="type",
        ax=ax[1],
        hue_order=["Original", "OOD"],
    )

    ax[0].set_xlim(0, latency_ms_cap)
    ax[0].set_xlabel("Latency (ms)")
    ax[0].set_ylabel("Density")
    ax[1].set_xlabel("Recall@10")
    # ax[1].set_ylabel("Count")
    ax[1].set_ylabel("Cumulative Distribution")

    plt.tight_layout()

    print(f"Saving plot to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_ood_results_py(
    results_path: str = "curator_py_ood_v2.pkl",
    output_path: str = "curator_py_ood_v2.png",
):
    results = pkl.load(open(results_path, "rb"))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ndists_df = pd.DataFrame(
        {
            "ndists": results["ndists"] + results["ood_ndists"],
            "type": ["Original"] * len(results["ndists"])
            + ["OOD"] * len(results["ood_ndists"]),
        }
    )

    recalls_df = pd.DataFrame(
        {
            "recall": results["recalls"] + results["ood_recalls"],
            "type": ["Original"] * len(results["recalls"])
            + ["OOD"] * len(results["ood_recalls"]),
        }
    )
    recalls_df["recall_bin"] = pd.cut(
        recalls_df["recall"], bins=10, labels=[f"{i/10}" for i in range(10)]
    )
    recalls_df = recalls_df.groupby(["type", "recall_bin"]).size().reset_index(name="count")  # type: ignore
    recalls_df["cumulative_count"] = recalls_df.groupby("type")["count"].cumsum()
    recalls_df["cdf"] = recalls_df["cumulative_count"] / recalls_df.groupby("type")[
        "count"
    ].transform("sum")

    sns.kdeplot(
        data=ndists_df,
        x="ndists",
        hue="type",
        ax=ax[0],
        hue_order=["Original", "OOD"],
        log_scale=(True, False),
    )
    sns.lineplot(
        data=recalls_df,
        x="recall_bin",
        y="cdf",
        hue="type",
        ax=ax[1],
        hue_order=["Original", "OOD"],
    )

    ax[0].set_xlabel("#Distance computation")
    ax[0].set_ylabel("Density")
    ax[1].set_xlabel("Recall@10")
    ax[1].set_ylabel("Cumulative Distribution")

    plt.tight_layout()

    print(f"Saving plot to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    fire.Fire()
