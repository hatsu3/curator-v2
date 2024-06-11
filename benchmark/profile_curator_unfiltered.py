import hashlib
import time
from pathlib import Path

import fire
import numpy as np
from tqdm import tqdm

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.utils import get_dataset_config, recall
from dataset import get_dataset, get_metadata
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

    return train_vecs, test_vecs, train_mds, test_mds, metadata


def compute_cache_key(train_vecs: np.ndarray, test_vecs: np.ndarray, k: int = 10):
    hasher = hashlib.md5()
    hasher.update(train_vecs.tobytes())
    hasher.update(test_vecs.tobytes())
    hasher.update(str(k).encode())
    return hasher.hexdigest()


def compute_ground_truth_cuda(
    test_vecs: np.ndarray,
    train_vecs: np.ndarray,
    k: int = 10,
    batch_size: int = 256,
):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    train_vecs_pth = torch.from_numpy(train_vecs).unsqueeze(0).cuda()
    test_vecs_pth = torch.from_numpy(test_vecs).unsqueeze(1).cuda()

    ground_truth = list()
    for i in tqdm(
        range(0, len(test_vecs_pth), batch_size), desc="Computing ground truth"
    ):
        test_vecs_pth_batch = test_vecs_pth[i : i + batch_size]
        dists_batch = torch.norm(train_vecs_pth - test_vecs_pth_batch, dim=2)
        topk_batch = torch.argsort(dists_batch, dim=1)[:, :k]
        ground_truth.append(topk_batch.cpu().numpy())

    ground_truth = np.concatenate(ground_truth, axis=0)
    return ground_truth


def compute_ground_truth_cpu(
    test_vecs: np.ndarray,
    train_vecs: np.ndarray,
    k: int = 10,
):
    ground_truth = []
    for vec in tqdm(test_vecs, desc="Computing ground truth"):
        dists = np.linalg.norm(train_vecs - vec, axis=1)
        labels = np.argsort(dists)[:k]
        ground_truth.append(labels)

    ground_truth = np.array(ground_truth)
    return ground_truth


def compute_ground_truth(
    test_vecs: np.ndarray,
    train_vecs: np.ndarray,
    k: int = 10,
    cache_dir: str | None = None,
    batch_size: int = 8,
) -> np.ndarray:
    cache_key = compute_cache_key(train_vecs, test_vecs, k)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"ground_truth_{cache_key}.npy"
        if cache_path.exists():
            print(f"Loading ground truth from cache {cache_path}...")
            return np.load(cache_path)
    else:
        cache_path = None

    try:
        ground_truth = compute_ground_truth_cuda(
            test_vecs, train_vecs, k=k, batch_size=batch_size
        )
    except RuntimeError:
        print("CUDA is not available / OOM. Using CPU...")
        ground_truth = compute_ground_truth_cpu(test_vecs, train_vecs, k=k)

    if cache_path is not None:
        np.save(cache_path, ground_truth)

    return ground_truth


def exp_curator_unfiltered(
    nlist: int = 16,
    max_sl_size: int = 256,
    max_leaf_size: int = 128,
    clus_niter: int = 20,
    update_bf_interval: int = 100,
    nprobe: int = 1200,
    prune_thres: float = 1.6,
    variance_boost: float = 0.4,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    gt_cache_dir: str | None = None,
    seed: int = 42,
):
    np.random.seed(seed)

    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, metadata = load_dataset(dataset_config)

    ground_truth = compute_ground_truth(
        test_vecs, train_vecs, k=10, cache_dir=gt_cache_dir
    )

    index_config = IndexConfig(
        index_cls=IVFFlatMultiTenantBFHierFaiss,
        index_params={
            "d": dim,
            "nlist": nlist,
            "bf_capacity": 1000,
            "bf_error_rate": 0.01,
            "max_sl_size": max_sl_size,
            "update_bf_interval": update_bf_interval,
            "clus_niter": clus_niter,
            "max_leaf_size": max_leaf_size,
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

    print("Index info:")
    index.index.print_tree_info()

    # query index
    query_results = []
    query_latencies = []

    for vec in tqdm(test_vecs, desc="Querying index"):
        query_start = time.time()
        labels = index.query_unfiltered(vec, k=10)
        query_latencies.append(time.time() - query_start)
        query_results.append(labels)

    avg_recall = recall(query_results, ground_truth)
    avg_search_lat = np.mean(query_latencies)
    std_search_lat = np.std(query_latencies)

    print(f"Average recall: {avg_recall:.4f}")
    print(f"Average search latency: {avg_search_lat:.4f}")
    print(f"Std search latency: {std_search_lat:.4f}")

    return avg_recall, avg_search_lat


if __name__ == "__main__":
    fire.Fire()
