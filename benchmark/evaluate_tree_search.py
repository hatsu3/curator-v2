import pickle as pkl
from dataclasses import dataclass
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from tqdm import tqdm

from benchmark.evaluate_exit_condition import per_tenant_flat_index
from benchmark.profile_curator_ood import load_or_generate_ood_dataset
from benchmark.utils import get_dataset_config, load_dataset
from indexes.curator_py import CuratorIndexPy


@dataclass
class FlatIVF:
    k: int
    centroids: np.ndarray
    buckets: list[list[int]]
    boost: np.ndarray | None = None

    def dists_to_query(self, q: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(self.centroids - q, axis=1)
        if self.boost is not None:
            dists -= self.boost
        return dists


def train_flat_ivf(
    X: np.ndarray, labels: np.ndarray, k: int, seed: int = 42
) -> FlatIVF:
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(X)
    buckets = [labels[np.where(kmeans.labels_ == i)[0]].tolist() for i in range(k)]
    return FlatIVF(k, kmeans.cluster_centers_, buckets)


def curator_to_flat_ivf(index: CuratorIndexPy, tenant: int) -> FlatIVF:
    flat_index = per_tenant_flat_index(index, tenant)
    centroids = np.array([node.centroid for node in flat_index])
    buckets = [node.shortlists[tenant] for node in flat_index]
    ivf = FlatIVF(len(flat_index), centroids, buckets)
    return ivf


def calc_curator_var_boosts(
    index: CuratorIndexPy, tenant: int, var_boost: float
) -> np.ndarray:
    flat_index = per_tenant_flat_index(index, tenant)
    boost = np.array([node.variance.get() for node in flat_index]) * var_boost
    return boost


def topk_radius(ivf: FlatIVF, q: np.ndarray, ground_truth: list[int]) -> list[int]:
    dists = ivf.dists_to_query(q)
    sorted_bucket_idxs = np.argsort(dists)

    radii = list()
    ndists = 0
    gt_set = set(ground_truth)

    for bucket_idx in sorted_bucket_idxs:
        ndists += len(ivf.buckets[bucket_idx])
        gt_found = gt_set & set(ivf.buckets[bucket_idx])
        radii.extend([ndists] * len(gt_found))

        gt_set -= gt_found
        if not gt_set:
            break

    assert not gt_set, "Some ground truth vectors are not found"
    assert len(radii) == len(ground_truth)

    return radii


def exp_topk_radius_curator_vs_flat_ivf(
    index_path: str = "curator_py_yfcc100m.index",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_ntenants: int = 100,
    seed: int = 42,
    var_boosts: list[float] = [0.0],
    pt_flat_idxs_cache_path: str = "pt_flat_idxs.pkl",
    output_path: str = "topk_radius_curator_vs_flat_ivf.pkl",
) -> None:
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, all_tenants = (
        load_dataset(dataset_config)
    )

    np.random.seed(seed)
    sampled_tenants = np.random.choice(
        sorted(all_tenants), sample_ntenants, replace=False
    ).tolist()

    print(f"Loading curator index from {index_path}...", flush=True)
    index = CuratorIndexPy.load(index_path)

    pt_curator_flat_idxs = [
        curator_to_flat_ivf(index, tenant) for tenant in sampled_tenants
    ]

    pt_curator_var_boosts = {
        (tenant, var_boost): calc_curator_var_boosts(index, tenant, var_boost)
        for tenant in sampled_tenants
        for var_boost in var_boosts
    }

    if Path(pt_flat_idxs_cache_path).exists():
        print(f"Loading flat indexes from {pt_flat_idxs_cache_path}...", flush=True)
        pt_flat_idxs = pkl.load(open(pt_flat_idxs_cache_path, "rb"))
    else:
        pt_flat_idxs = list()
        for tenant, curator_flat_idx in tqdm(
            zip(sampled_tenants, pt_curator_flat_idxs),
            total=len(sampled_tenants),
            desc="Training flat IVFs",
        ):
            mask = np.array(
                [i for i, access_list in enumerate(train_mds) if tenant in access_list]
            )
            flat_idx = train_flat_ivf(
                train_vecs[mask], mask, curator_flat_idx.k, seed=seed
            )
            pt_flat_idxs.append(flat_idx)

        print(f"Saving flat indexes to {pt_flat_idxs_cache_path}...", flush=True)
        pkl.dump(pt_flat_idxs, open(pt_flat_idxs_cache_path, "wb"))

    results = list()
    gt_gen = iter(ground_truth)

    for vec, access_list in tqdm(
        zip(test_vecs, test_mds), total=len(test_vecs), desc="Calculating topk radius"
    ):
        for tenant in access_list:
            gt = next(gt_gen)
            if tenant not in sampled_tenants:
                continue

            tenant_idx = sampled_tenants.index(tenant)
            result = {
                "tenant": tenant,
                "flat": topk_radius(pt_flat_idxs[tenant_idx], vec, gt),
            }

            for var_boost in var_boosts:
                curator_ivf = pt_curator_flat_idxs[tenant_idx]
                curator_ivf.boost = pt_curator_var_boosts[(tenant, var_boost)]
                result[f"curator_{var_boost}"] = topk_radius(curator_ivf, vec, gt)

            results.append(result)

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump(results, open(output_path, "wb"))


def plot_topk_radius_curator_vs_flat_ivf(
    results_path: str = "topk_radius_curator_vs_flat_ivf.pkl",
    output_path: str = "topk_radius_curator_vs_flat_ivf.png",
) -> None:
    results = pkl.load(open(results_path, "rb"))

    curator_radii = np.array([res["curator_0.0"][-1] for res in results])
    flat_radii = np.array([res["flat"][-1] for res in results])
    max_radii = max(curator_radii.max(), flat_radii.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.scatterplot(x=curator_radii, y=flat_radii, ax=ax1, s=10, alpha=0.2)
    sns.lineplot(x=[0, max_radii], y=[0, max_radii], color="red", ax=ax1)
    ax1.set_xlabel("Curator top-k radius")
    ax1.set_ylabel("Flat IVF top-k radius")
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    curator_labels = list(results[0].keys() - {"tenant", "flat"})
    curator_labels.sort(key=lambda x: float(x.split("_")[1]))
    curator_colors = sns.color_palette("Greys", len(curator_labels))

    for label, color in zip(curator_labels, curator_colors):
        radii = np.array([res[label][-1] for res in results])
        sns.kdeplot(
            radii, label=label, ax=ax2, log_scale=True, cumulative=True, color=color
        )

    sns.kdeplot(flat_radii, label="Flat IVF", ax=ax2, log_scale=True, cumulative=True)
    ax2.legend()
    ax2.set_xlabel("Top-k radius")
    ax2.set_ylabel("Cumulative density")

    # ax = sns.jointplot(
    #     x=curator_radii, y=flat_radii, kind="scatter", marker=".", s=10, alpha=0.3
    # )
    # ax.ax_joint.plot([10, max_radii], [10, max_radii])
    # ax.ax_joint.set_xscale("log")
    # ax.ax_joint.set_yscale("log")
    # ax.ax_joint.set_xlabel("Curator top-k radius")
    # ax.ax_joint.set_ylabel("Flat IVF top-k radius")
    # plt.tight_layout()

    print(f"Saving plot to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def exp_topk_radius_id_vs_ood(
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ood_version: str = "v2",
    dataset_cache_path: str = "yfcc100m_ood_v2.pkl",
    sample_ntenants: int = 100,
    seed: int = 42,
    output_path: str = "topk_radius_id_vs_ood.pkl",
):
    def exp_on_dataset(
        train_vecs, test_vecs, train_mds, test_mds, ground_truth, sampled_tenants
    ):
        pt_flat_idxs = list()
        for tenant in tqdm(sampled_tenants, desc="Training flat IVFs"):
            mask = np.array(
                [i for i, access_list in enumerate(train_mds) if tenant in access_list]
            )
            k = np.sqrt(len(mask)).astype(int)
            flat_idx = train_flat_ivf(train_vecs[mask], mask, k, seed=seed)
            pt_flat_idxs.append(flat_idx)

        results = list()
        gt_gen = iter(ground_truth)

        for vec, access_list in tqdm(
            zip(test_vecs, test_mds),
            total=len(test_vecs),
            desc="Calculating topk radius",
        ):
            for tenant in access_list:
                gt = next(gt_gen)
                if tenant not in sampled_tenants:
                    continue

                tenant_idx = sampled_tenants.index(tenant)
                results.append(
                    {
                        "tenant": tenant,
                        "radii": topk_radius(pt_flat_idxs[tenant_idx], vec, gt),
                    }
                )

        return results

    print(f"Running experiment on the original dataset...", flush=True)
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, all_tenants = (
        load_dataset(dataset_config)
    )

    np.random.seed(seed)
    sampled_tenants = np.random.choice(
        sorted(all_tenants), sample_ntenants, replace=False
    ).tolist()

    id_results = exp_on_dataset(
        train_vecs, test_vecs, train_mds, test_mds, ground_truth, sampled_tenants
    )

    print(f"Loading OOD dataset...", flush=True)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth = (
        load_or_generate_ood_dataset(dataset_config, dataset_cache_path, ood_version)
    )

    ood_results = exp_on_dataset(
        train_vecs, test_vecs, train_mds, test_mds, ground_truth, sampled_tenants
    )

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump({"id": id_results, "ood": ood_results}, open(output_path, "wb"))


def plot_topk_radius_id_vs_ood(
    results_path: str = "topk_radius_id_vs_ood.pkl",
    output_path: str = "topk_radius_id_vs_ood.png",
):
    results = pkl.load(open(results_path, "rb"))

    id_radii = np.array([res["radii"][-1] for res in results["id"]])
    ood_radii = np.array([res["radii"][-1] for res in results["ood"]])

    sns.kdeplot(id_radii, label="Original", log_scale=True, cumulative=True)
    sns.kdeplot(ood_radii, label="OOD", log_scale=True, cumulative=True)
    plt.legend()
    plt.xlabel("Top-k radius")
    plt.ylabel("Cumulative density")

    print(f"Saving plot to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    fire.Fire()
