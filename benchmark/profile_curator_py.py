import os
from pathlib import Path

import fire
import numpy as np
from tqdm import tqdm

from benchmark.utils import get_dataset_config, load_dataset
from indexes.curator_py import CuratorIndexPy, CuratorParam


def get_curator_index(
    dim: int,
    nlist: int = 16,
    prune_thres: float = 1.6,
    max_sl_size: int = 256,
    max_leaf_size: int = 128,
    nprobe: int = 4000,
    var_boost: float = 0.4,
) -> CuratorIndexPy:
    param = CuratorParam(
        n_clusters=nlist,
        prune_thres=prune_thres,
        max_sl_size=max_sl_size,
        max_leaf_size=max_leaf_size,
        nprobe=nprobe,
        var_boost=var_boost,
    )
    index = CuratorIndexPy(dim, param)
    return index


def train_index(
    index: CuratorIndexPy,
    train_vecs: np.ndarray,
    train_mds: list[list[int]],
):
    index.train(train_vecs)

    for i, (vec, access_list) in tqdm(
        enumerate(zip(train_vecs, train_mds)),
        total=len(train_vecs),
        desc="Building index",
    ):
        if not access_list:
            continue

        index.insert(vec, i)
        for tenant in access_list:
            index.grant_access(i, tenant)


def query_index(
    index: CuratorIndexPy,
    test_vecs: np.ndarray,
    test_mds: list[list[int]],
    ground_truth: list[list[int]],
):
    results = list()
    for vec, access_list in tqdm(
        zip(test_vecs, test_mds),
        total=len(test_vecs),
        desc="Querying index",
    ):
        if not access_list:
            continue

        for tenant in access_list:
            pred = index.search(vec, 10, tenant)
            results.append(pred)

    recalls = list()
    for pred, truth in zip(results, ground_truth):
        truth = [t for t in truth if t != -1]
        recalls.append(len(set(pred) & set(truth)) / len(truth))

    return np.mean(recalls)


def exp_curator_py(
    nlist: int = 16,
    prune_thres: float = 1.6,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    index_path: str = "curator_py.index",
    index_path_del: str = "curator_py_del.index",
    sanity_check: bool = False,
    sanity_check_del: bool = False,
):
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config
    )

    if Path(index_path).exists():
        print(f"Loading index from {index_path}...", flush=True)
        index = CuratorIndexPy.load(index_path)
    else:
        print(f"Initializing index...", flush=True)
        index = get_curator_index(
            dim=train_vecs.shape[1], nlist=nlist, prune_thres=prune_thres
        )

        print(f"Training index...", flush=True)
        train_index(index, train_vecs, train_mds)

        print(f"Saving index to {index_path}...", flush=True)
        index.save(index_path)

    if sanity_check:
        print(f"Sanity check...", flush=True)
        assert index.sanity_check({i: mds for i, mds in enumerate(train_mds) if mds})

    print(f"Querying index...", flush=True)
    test_vecs = test_vecs[:sample_test_size]
    test_mds = test_mds[:sample_test_size]
    recall = query_index(index, test_vecs, test_mds, ground_truth.tolist())
    print(f"Recall@10: {recall:.4f}", flush=True)

    if Path(index_path_del).exists():
        print(f"Loading index from {index_path_del}...", flush=True)
        index = CuratorIndexPy.load(index_path_del)
    else:
        print("Deleting vectors from index...", flush=True)
        train_mds_after_del = train_mds.copy()

        np.random.seed(42)
        for i, __ in tqdm(
            enumerate(train_vecs), total=len(train_vecs), desc="Deleting"
        ):
            for j, tenant in enumerate(train_mds[i]):
                if np.random.rand() < 0.3:
                    index.revoke_access(i, tenant)
                    train_mds_after_del[i].pop(j)

                if not train_mds_after_del[i]:
                    index.delete(i)

        print(f"Saving index to {index_path_del}...", flush=True)
        index.save(index_path_del)

    if sanity_check_del:
        print(f"Sanity check after deletion...", flush=True)
        assert index.sanity_check(
            {i: mds for i, mds in enumerate(train_mds_after_del) if mds}
        )


def exp_curator_py_batch(
    nlist: int = 16,
    prune_thres: float = 1.6,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    index_path: str = "curator_py_yfcc100m_batch.index",
    sanity_check: bool = False,
):
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config
    )

    if Path(index_path).exists():
        print(f"Loading index from {index_path}...", flush=True)
        index = CuratorIndexPy.load(index_path)
    else:
        print(f"Initializing index...", flush=True)
        index = get_curator_index(
            dim=train_vecs.shape[1], nlist=nlist, prune_thres=prune_thres
        )

        print(f"Training index...", flush=True)
        index.train(train_vecs)
        index.batch_insert(train_vecs, train_mds)

        print(f"Saving index to {index_path}...", flush=True)
        index.save(index_path)

    if sanity_check:
        print(f"Sanity check...", flush=True)
        assert index.sanity_check({i: mds for i, mds in enumerate(train_mds) if mds})

    print(f"Querying index...", flush=True)
    test_vecs = test_vecs[:sample_test_size]
    test_mds = test_mds[:sample_test_size]
    results, stats = index.batch_search(test_vecs, test_mds, k=10)
    recall = np.mean(
        [
            len(set(pred) & set(truth)) / len(truth)
            for pred, truth in zip(results, ground_truth)
        ]
    )
    print(f"Recall@10: {recall:.4f}", flush=True)


def test_curator_py():
    # generate synthesized data
    np.random.seed(42)

    n_samples = 100
    n_features = 100
    X = np.random.randn(n_samples, n_features)
    owners = np.array([i % 10 for i in range(n_samples)], dtype=int)

    # train index
    param = CuratorParam(n_clusters=4, max_sl_size=8, max_leaf_size=8, nprobe=10)
    index = CuratorIndexPy(n_features, param)
    index.train(X)

    # insert vectors
    for i in range(n_samples):
        index.insert(X[i], i)

    # grant access
    for i, owner in enumerate(owners):
        index.grant_access(i, owner)

    assert index.sanity_check({i: [owners[i]] for i in range(n_samples)})

    # delete vectors
    deleted_idxs = np.random.choice(n_samples, 30, replace=False)
    for i in deleted_idxs:
        index.revoke_access(i, int(owners[i]))
        index.delete(i)

    assert index.sanity_check(
        {i: [owners[i]] for i in range(n_samples) if i not in deleted_idxs}
    )

    Q = np.random.randn(n_samples, n_features)
    Q_owners = np.array([i % 10 for i in range(n_samples)], dtype=int)

    # compute ground truth
    def ground_truth(q, tenant, k):
        filtered_indexes = np.nonzero(owners == tenant)[0]
        filtered_indexes = np.array(
            list(set(filtered_indexes) - set(deleted_idxs)), dtype=int
        )
        distances = np.linalg.norm(X[filtered_indexes] - q, axis=1)
        return filtered_indexes[np.argsort(distances)[:k]]

    def recall(pred, truth):
        return len(set(pred) & set(truth)) / len(truth)

    index.save("curator_py.index")
    index = CuratorIndexPy.load("curator_py.index")
    os.remove("curator_py.index")

    k = 5
    recalls = list()
    for q, tenant in zip(Q, Q_owners):
        pred = index.search(q, k, tenant)

        if set(pred) & set(deleted_idxs):
            raise ValueError("Deleted vectors are returned")

        truth = ground_truth(q, tenant, k)
        recalls.append(recall(pred, truth))

    print(f"Recall@{k}: {np.mean(recalls):.2f}")


if __name__ == "__main__":
    fire.Fire()
