import pickle as pkl
import time
from collections import Counter
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.utils import get_dataset_config, recall
from dataset import get_dataset, get_metadata
from dataset.utils import compute_ground_truth, load_sampled_metadata
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss


def load_dataset(dataset_config: DatasetConfig):
    train_vecs, test_vecs, metadata = get_dataset(
        dataset_name=dataset_config.dataset_name, **dataset_config.dataset_params
    )

    train_mds, test_mds = get_metadata(
        synthesized=dataset_config.synthesize_metadata,
        train_vecs=train_vecs,
        test_vecs=test_vecs,
        dataset_name=dataset_config.dataset_name,
        **dataset_config.metadata_params,
    )

    ground_truth, train_cates = compute_ground_truth(
        train_vecs,
        train_mds,
        test_vecs,
        test_mds,
        k=10,
        metric=metadata["metric"],
        multi_tenant=True,
    )

    train_mds, test_mds = load_sampled_metadata(train_mds, test_mds, train_cates)

    return train_vecs, test_vecs, train_mds, test_mds, ground_truth


def neighborhood_density(
    X: np.ndarray, mds: list[list[int]], q: np.ndarray, tid: int, rs: list[float]
):
    dists = np.linalg.norm(X - q, axis=1)

    densities = list()
    for r in rs:
        neigh_mask = dists <= r
        num_neigh = np.sum(neigh_mask).item()

        if num_neigh == 0:
            densities.append(-1)
            continue

        num_neigh_filtered = 0
        for i in neigh_mask.nonzero()[0]:
            if tid in mds[i]:
                num_neigh_filtered += 1

        densities.append(num_neigh_filtered / num_neigh)

    return densities


def neighborhood_density_v2(
    X: np.ndarray, mds: list[list[int]], q: np.ndarray, tid: int, rs: list[int]
):
    dists = np.linalg.norm(X - q, axis=1)
    ranks = np.argsort(dists)

    densities = list()
    for r in rs:
        num_neigh_filtered = 0
        for i in ranks[:r]:
            if tid in mds[i]:
                num_neigh_filtered += 1
        densities.append(num_neigh_filtered / r)

    return densities


def neighborhood_density_cuda(
    X: torch.Tensor, mds: list[list[int]], q: torch.Tensor, tid: int, rs: list[float]
):
    dists = torch.norm(X - q, dim=-1)

    densities = list()
    for r in rs:
        neigh_mask = dists <= r
        num_neigh = torch.sum(neigh_mask).item()

        if num_neigh == 0:
            densities.append(-1)
            continue

        num_neigh_filtered = 0
        for i in neigh_mask.nonzero(as_tuple=True)[0]:
            if tid in mds[i]:
                num_neigh_filtered += 1

        densities.append(num_neigh_filtered / num_neigh)

    return densities


def neighborhood_density_v2_cuda(
    X: torch.Tensor, mds: list[list[int]], q: torch.Tensor, tid: int, rs: list[int]
):
    assert all(rs[i] < rs[i + 1] for i in range(len(rs) - 1)), "rs must be sorted"

    dists = torch.norm(X - q, dim=-1)
    ranks = torch.argsort(dists)

    densities = list()
    num_neigh_filtered = 0

    for r, i in enumerate(ranks[: max(rs)]):
        if tid in mds[i]:
            num_neigh_filtered += 1

        if r + 1 in rs:
            densities.append(num_neigh_filtered / (r + 1))

    return densities


def neighborhood_density_cuda_batch(
    train_vecs: np.ndarray,
    train_mds: list[list[int]],
    test_vecs: np.ndarray,
    test_mds: list[list[int]],
    r: float,
    batch_size: int = 8,
) -> list[float]:
    train_vecs_pth = torch.tensor(train_vecs, device="cuda").unsqueeze(0)
    test_vecs_pth = torch.tensor(test_vecs, device="cuda").unsqueeze(1)

    densities = list()
    for i in tqdm(range(0, len(test_vecs_pth), batch_size), desc="Computing density"):
        test_vecs_pth_batch = test_vecs_pth[i : i + batch_size]
        neigh_mask = torch.norm(train_vecs_pth - test_vecs_pth_batch, dim=-1) <= r
        neigh_mask0, neigh_mask1 = torch.nonzero(neigh_mask, as_tuple=True)

        for j, mds in enumerate(test_mds[i : i + batch_size]):
            num_neigh = torch.sum(neigh_mask0 == j)

            if num_neigh == 0:
                densities.extend([-1] * len(mds))
                continue

            for tid in mds:
                num_neigh_filtered = 0
                for k in neigh_mask1[neigh_mask0 == j]:
                    if tid in train_mds[k]:
                        num_neigh_filtered += 1
                densities.append(num_neigh_filtered / num_neigh)

    return densities


def neighborhood_density_v2_cuda_batch(
    train_vecs: np.ndarray,
    train_mds: list[list[int]],
    test_vecs: np.ndarray,
    test_mds: list[list[int]],
    r: int,
    sample_prob: float = 0.01,
    seed: int = 42,
    batch_size: int = 8,
) -> list[float]:
    np.random.seed(seed)

    train_vecs_pth = torch.tensor(train_vecs, device="cuda").unsqueeze(0)
    test_vecs_pth = torch.tensor(test_vecs, device="cuda").unsqueeze(1)

    densities = list()
    for i in tqdm(range(0, len(test_vecs_pth), batch_size), desc="Computing density"):
        test_vecs_pth_batch = test_vecs_pth[i : i + batch_size]
        dists = torch.norm(train_vecs_pth - test_vecs_pth_batch, dim=-1)
        ranks = torch.argsort(dists, dim=1)

        for j, mds in enumerate(test_mds[i : i + batch_size]):
            for tid in mds:
                if np.random.rand() >= sample_prob:
                    continue

                num_neigh_filtered = 0
                for k in ranks[j, :r]:
                    if tid in train_mds[k]:
                        num_neigh_filtered += 1

                densities.append(num_neigh_filtered / r)

    return densities


def plot_per_label_selectivity(
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ood: bool = False,
    ood_dataset_cache_path: str | None = None,
    ood_version: str = "v1",
    cache_csv_path: str = "per_label_selectivity.csv",
    output_path: str = "per_label_selectivity.png",
):
    if not Path(cache_csv_path).exists():
        print("Loading dataset...", flush=True)
        dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)

        if not ood:
            __, __, train_mds, __, __ = load_dataset(dataset_config)
        else:
            from benchmark.profile_curator_ood import load_or_generate_ood_dataset

            print("Loading/generating OOD dataset...", flush=True)
            assert ood_dataset_cache_path is not None
            __, __, train_mds, __, __ = (
                load_or_generate_ood_dataset(
                    dataset_config, ood_dataset_cache_path, ood_version
                )
            )

        print("Computing selectivity...", flush=True)
        counter = Counter()
        for mds in tqdm(train_mds, total=len(train_mds), desc="Calculating"):
            counter.update(mds)

        print(f"Saving results to {cache_csv_path}...", flush=True)
        results = pd.DataFrame(
            [
                {"Label ID": label_id, "Selectivity": count / len(train_mds)}
                for label_id, count in counter.items()
            ]
        )
        results.to_csv(cache_csv_path, index=False)
    else:
        print(f"Loading cached results from {cache_csv_path}...", flush=True)
        results = pd.read_csv(cache_csv_path)

    print("Plotting...", flush=True)
    sns.displot(x="Selectivity", data=results, kind="kde", log_scale=(True, False))
    plt.xlabel("Selectivity")
    plt.ylabel("Density")

    print(f"Saving plot to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_per_label_density(
    r: int = 3000,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ood: bool = False,
    ood_dataset_cache_path: str | None = None,
    ood_version: str = "v1",
    sample_prob: float = 0.01,
    seed: int = 42,
    density_cache_path: str = "neigh_density.pkl",
    cache_csv_path: str = "per_label_density.csv",
    output_path: str = "per_label_density.png",
):
    if not Path(cache_csv_path).exists():
        print("Loading dataset...", flush=True)
        dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)

        if not ood:
            train_vecs, test_vecs, train_mds, test_mds, __ = load_dataset(
                dataset_config
            )
        else:
            from benchmark.profile_curator_ood import load_or_generate_ood_dataset

            print("Loading/generating OOD dataset...", flush=True)
            assert ood_dataset_cache_path is not None
            train_vecs, test_vecs, train_mds, test_mds, __ = (
                load_or_generate_ood_dataset(
                    dataset_config, ood_dataset_cache_path, ood_version
                )
            )

        print("Computing neighborhood density...", flush=True)
        if Path(density_cache_path).exists():
            print(f"Loading cached results from {density_cache_path}...", flush=True)
            cache = pkl.load(open(density_cache_path, "rb"))
            assert cache["dataset_key"] == dataset_key
            assert cache["test_size"] == test_size
            assert cache["neigh_radius"] == r
            densities = cache["densities"]
        else:
            Path(density_cache_path).parent.mkdir(parents=True, exist_ok=True)
            densities = neighborhood_density_v2_cuda_batch(
                train_vecs,
                train_mds,
                test_vecs,
                test_mds,
                r=r,
                sample_prob=sample_prob,
                seed=seed,
            )

            print(f"Saving results to {density_cache_path}...", flush=True)
            pkl.dump(
                {
                    "dataset_key": dataset_key,
                    "neigh_radius": r,
                    "test_size": test_size,
                    "densities": densities,
                },
                open(density_cache_path, "wb"),
            )

        np.random.seed(seed)

        results = list()
        densities_iter = iter(densities)

        for mds in tqdm(test_mds, total=len(test_mds), desc="Calculating"):
            for tid in mds:
                if np.random.rand() >= sample_prob:
                    continue

                density = next(densities_iter)
                results.append(
                    {
                        "Label ID": tid,
                        "Density": density,
                    }
                )

        print(f"Saving results to {cache_csv_path}...", flush=True)
        results = pd.DataFrame(results)
        results.to_csv(cache_csv_path, index=False)
    else:
        print(f"Loading cached results from {cache_csv_path}...", flush=True)
        results = pd.read_csv(cache_csv_path)

    print("Plotting...", flush=True)
    results = results.groupby("Label ID").agg({"Density": "mean"}).reset_index()
    results = results[results["Density"] > 0]
    sns.displot(x="Density", data=results, kde=True, log_scale=(True, False))

    print(f"Saving plot to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_per_label_relative_density(
    selectivity_csv_path: str = "per_label_selectivity.csv",
    density_csv_path: str = "per_label_density.csv",
    output_path: str = "per_label_relative_density.png",
):
    if not Path(selectivity_csv_path).exists() or not Path(density_csv_path).exists():
        raise ValueError("Cache files not found")

    print(f"Loading cached results from {selectivity_csv_path}...", flush=True)
    selectivity_df = pd.read_csv(selectivity_csv_path)

    print(f"Loading cached results from {density_csv_path}...", flush=True)
    density_df = pd.read_csv(density_csv_path)
    density_df = density_df.groupby("Label ID").agg({"Density": "mean"}).reset_index()

    print("Plotting...", flush=True)
    results = pd.merge(selectivity_df, density_df, on="Label ID")
    results["Relative Density"] = results["Density"] / results["Selectivity"]
    results = results[results["Relative Density"] > 0]
    sns.displot(x="Relative Density", data=results, kde=True, log_scale=(True, False))

    print(f"Saving plot to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_density_vs_radius(
    rmin: int = 1,
    rstep: int = 5,
    rnum: int = 20,
    sample_prob: float = 0.01,
    seed: int = 42,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    cache_csv_path: str = "density_vs_radius.csv",
    output_path: str = "density_vs_radius.png",
):
    if not Path(cache_csv_path).exists():
        print("Loading dataset...", flush=True)
        dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
        train_vecs, test_vecs, train_mds, test_mds, __ = load_dataset(dataset_config)
        train_vecs_pth = torch.tensor(train_vecs, device="cuda")
        test_vecs_pth = torch.tensor(test_vecs, device="cuda")

        np.random.seed(seed)

        results = []
        for i, (q, mds) in tqdm(
            enumerate(zip(test_vecs_pth, test_mds)),
            total=len(test_vecs_pth),
            desc="Generating",
        ):
            for tid in mds:
                if np.random.rand() >= sample_prob:
                    continue

                rs = list(range(rmin, rmin + rnum * rstep, rstep))
                densities = neighborhood_density_v2_cuda(
                    train_vecs_pth, train_mds, q, tid, rs
                )
                results.extend(
                    [
                        {
                            "Neighborhood radius": r,
                            "Density": density,
                            "Query ID": i,
                            "Tenant ID": tid,
                        }
                        for r, density in zip(rs, densities)
                    ]
                )

        print(f"Saving results to {cache_csv_path}...", flush=True)
        results = pd.DataFrame(results)
        results.to_csv(cache_csv_path, index=False)
    else:
        print(f"Loading cached results from {cache_csv_path}...", flush=True)
        results = pd.read_csv(cache_csv_path)

    print("Plotting...", flush=True)
    sns.lineplot(
        x="Neighborhood radius",
        y="Density",
        data=results,
        errorbar=("pi", 50),
        estimator="median",
    )

    print(f"Saving plot to {output_path}...", flush=True)
    plt.savefig(output_path, dpi=200)


def plot_density_vs_radius_per_queries(
    cache_csv_path: str = "density_vs_radius.csv",
    output_path: str = "density_vs_radius_per_queries.png",
):
    print(f"Loading cached results from {cache_csv_path}...", flush=True)
    results = pd.read_csv(cache_csv_path)

    print("Plotting...", flush=True)

    results_agg = (
        results.groupby(["Query ID", "Tenant ID"])
        .agg({"Density": ["mean"]})
        .reset_index()
    )
    results_agg.columns = ["Query ID", "Tenant ID", "Density"]
    results_agg = results_agg.sort_values("Density").reset_index(drop=True)
    density_qs = results_agg["Density"].quantile(np.linspace(0.0, 1.0, 11))

    sampled_rows = list()
    for q in density_qs:
        row = results_agg[results_agg["Density"] >= q].iloc[0]
        sampled_rows.append(row)

    sampled_rows = (
        pd.DataFrame(sampled_rows)
        .drop_duplicates(subset=["Tenant ID", "Query ID"])
        .reset_index(drop=True)
    )

    for __, row in sampled_rows.iterrows():
        query_id = row["Query ID"]
        tenant_id = row["Tenant ID"]
        res = results[
            (results["Query ID"] == query_id) & (results["Tenant ID"] == tenant_id)
        ]
        sns.lineplot(x="Neighborhood radius", y="Density", data=res)

    print(f"Saving plot to {output_path}...", flush=True)
    plt.savefig(output_path, dpi=200)


def exp_curator_skewness(
    nlist: int = 16,
    prune_thres: float = 1.6,
    neigh_radius: int = 3000,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_prob: float = 0.01,
    density_cache_path: str = "neigh_density.pkl",
    output_path: str = "curator_skewness.pkl",
):
    # load dataset
    print("Loading dataset...", flush=True)
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth = load_dataset(
        dataset_config
    )

    # compute neighborhood density
    print("Computing neighborhood density...", flush=True)
    if Path(density_cache_path).exists():
        cache = pkl.load(open(density_cache_path, "rb"))
        assert cache["dataset_key"] == dataset_key
        assert cache["test_size"] == test_size
        assert cache["neigh_radius"] == neigh_radius
        densities = cache["densities"]
    else:
        Path(density_cache_path).parent.mkdir(parents=True, exist_ok=True)
        densities = neighborhood_density_v2_cuda_batch(
            train_vecs,
            train_mds,
            test_vecs,
            test_mds,
            r=neigh_radius,
            sample_prob=sample_prob,
        )
        pkl.dump(
            {
                "dataset_key": dataset_key,
                "neigh_radius": neigh_radius,
                "test_size": test_size,
                "densities": densities,
            },
            open(density_cache_path, "wb"),
        )

    index_config = IndexConfig(
        index_cls=IVFFlatMultiTenantBFHierFaiss,
        index_params={
            "d": dim,
            "nlist": nlist,
            "bf_capacity": 1000,
            "bf_error_rate": 0.01,
            "max_sl_size": 256,
            "max_leaf_size": 128,
            "update_bf_interval": 100,
            "clus_niter": 20,
        },
        search_params={
            "nprobe": 4000,
            "prune_thres": prune_thres,
            "variance_boost": 0.4,
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

    # train index
    print("Training index...", flush=True)
    index.train(train_vecs, train_mds)

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

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pkl.dump(
            {
                "densities": densities,
                "recalls": recalls,
                "latencies": query_latencies,
                "nlist": nlist,
                "prune_thres": prune_thres,
                "neigh_radius": neigh_radius,
                "dataset_key": dataset_key,
                "test_size": test_size,
            },
            open(output_path, "wb"),
        )


def plot_curator_skewness(
    sample_prob: float = 0.01,
    seed: int = 42,
    result_path="curator_skewness.pkl",
    output_path="curator_skewness.png",
):
    print(f"Loading cached results from {result_path}...", flush=True)
    results = pkl.load(open(result_path, "rb"))

    np.random.seed(seed)
    mask = [np.random.rand() < sample_prob for _ in results["recalls"]]

    df = pd.DataFrame(
        {
            "Recall": np.array(results["recalls"])[mask],
            "Density": results["densities"],
            "Latency": np.array(results["latencies"])[mask] * 1000,
        }
    )
    df["Density Bin"] = pd.cut(df["Density"], bins=8, labels=False)

    print("Plotting...", flush=True)
    __, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.violinplot(
        x="Density Bin",
        y="Recall",
        data=df,
        ax=ax1,
        inner="box",
        density_norm="width",
    )
    sns.violinplot(
        x="Density Bin",
        y="Latency",
        data=df,
        ax=ax2,
        inner="box",
        density_norm="width",
    )
    ax1.set_xlabel("Neighborhood density (binned)")
    ax2.set_xlabel("Neighborhood density (binned)")
    ax1.set_ylabel("Recall@10")
    ax2.set_ylabel("Latency (ms)")
    plt.tight_layout()

    output_path1 = output_path.replace(".png", "_violin.png")
    print(f"Saving plot to {output_path1} ...", flush=True)
    plt.savefig(output_path1, dpi=200)

    plt.figure(figsize=(5, 5))
    sns.jointplot(x="Density", y="Recall", data=df, s=5, alpha=0.3)
    plt.xlabel("Neighborhood density")
    plt.ylabel("Recall@10")

    output_path1 = output_path.replace(".png", "_density_vs_recall_joint.png")
    print(f"Saving plot to {output_path1} ...", flush=True)
    plt.savefig(output_path1, dpi=200)

    plt.figure(figsize=(5, 5))
    sns.jointplot(x="Density", y="Latency", data=df, s=5, alpha=0.3)
    plt.xlabel("Neighborhood density")
    plt.ylabel("Latency (ms)")

    output_path1 = output_path.replace(".png", "_density_vs_latency_joint.png")
    print(f"Saving plot to {output_path1} ...", flush=True)
    plt.savefig(output_path1, dpi=200)


if __name__ == "__main__":
    fire.Fire()
