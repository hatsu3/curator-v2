import hashlib
import pickle as pkl
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.utils import get_dataset_config, recall
from dataset import get_dataset, get_metadata
from indexes.hnsw_mt_hnswlib import HNSWMultiTenantHnswlib
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


def evaluate_predicate(formula: str, mds: list[int]):
    """
    Evaluate a boolean formula in Polish notation.
    Example: 'AND 1 OR NOT 0 2'
    A variable (like '1') is considered True if it is in mds.
    """
    tokens = formula.split()
    stack = []

    def eval_token(token):
        try:
            return int(token) in mds
        except ValueError:
            raise ValueError(f"Malformed formula: {formula}")

    for token in reversed(tokens):
        if token in ("AND", "OR", "NOT"):
            if token == "AND":
                a = stack.pop()
                b = stack.pop()
                stack.append(a and b)
            elif token == "OR":
                a = stack.pop()
                b = stack.pop()
                stack.append(a or b)
            elif token == "NOT":
                a = stack.pop()
                stack.append(not a)
        else:
            stack.append(eval_token(token))

    return stack.pop()


def generate_random_filters(
    num_filters_per_template: int, num_tenants: int, seed: int = 42
):
    def sample_parameters(n, d):
        np.random.seed(seed)
        return [
            np.random.choice(num_tenants, d, replace=False).tolist() for _ in range(n)
        ]

    templates = [
        "NOT {0}",
        "AND {0} {1}",
        "OR {0} {1}",
        # "AND {0} NOT {1}",
        # "OR {0} NOT {1}",
        # "OR {0} OR {1} {2}",
        # "OR {0} AND {1} {2}",
        # "OR {0} AND {1} NOT {2}",
        # "OR {0} OR {1} NOT {2}",
        # "AND {0} OR {1} {2}",
        # "AND {0} AND {1} {2}",
        # "AND {0} AND {1} NOT {2}",
        # "AND {0} OR {1} NOT {2}",
    ]

    filters = {}
    for template in templates:
        n_param = template.count("{")
        filters[template] = []
        for param in sample_parameters(num_filters_per_template, n_param):
            filters[template].append(template.format(*param))

    return filters


def compute_ground_truth(
    queries: np.ndarray,
    filters: list[str],
    train_vecs: np.ndarray,
    train_mds: list[list[int]],
    k: int = 10,
    cache_dir: str | None = None,
    batch_size: int = 8,
):
    try:
        import torch

        print("Using PyTorch for computing distances")
        use_cuda = True
    except:
        print("Using numpy for computing distances")
        use_cuda = False

    def compute_cache_key(queries, filters, k) -> str:
        arr_bytes = queries.tobytes()
        hasher = hashlib.md5()
        hasher.update(arr_bytes)
        for filter in sorted(filters):
            hasher.update(filter.encode())
        hasher.update(str(k).encode())
        return hasher.hexdigest()

    cache_key = compute_cache_key(queries, filters, k)

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"{cache_key}.pkl"
        if cache_path.exists():
            data = pkl.load(cache_path.open("rb"))
            return data["ground_truths"], data["selectivities"]
    else:
        cache_path = None

    ground_truths = {}
    selectivities = {}

    for filter in tqdm(filters, desc="Computing ground truth"):
        filtered_ids = np.array(
            [
                i
                for i, md in enumerate(train_mds)
                if md and evaluate_predicate(filter, md)
            ]
        )

        selectivities[filter] = filtered_ids.size / train_vecs.shape[0]

        if filtered_ids.size == 0:
            ground_truths[filter] = [[-1] * k for _ in range(queries.shape[0])]
            continue

        filtered_vecs = train_vecs[filtered_ids]

        if use_cuda:
            filtered_vecs_pth = torch.from_numpy(filtered_vecs).unsqueeze(0).cuda()
            queries_pth = torch.from_numpy(queries).unsqueeze(1).cuda()

            topk_ids = []
            for i in range(0, len(queries), batch_size):
                queries_pth_batch = queries_pth[i : i + batch_size]
                dists_pth_batch = torch.norm(
                    filtered_vecs_pth - queries_pth_batch, dim=2
                )
                topk_pth_batch = torch.argsort(dists_pth_batch, dim=1)[:, :k]
                topk_ids_batch = filtered_ids[topk_pth_batch.cpu().numpy()]
                topk_ids.extend(topk_ids_batch.tolist())

            topk_ids = np.array(topk_ids)
        else:
            dists = np.linalg.norm(filtered_vecs - queries[:, None], axis=2)
            topk_ids = filtered_ids[np.argsort(dists, axis=1)[:, :k]]

        topk_ids = np.pad(
            topk_ids, ((0, 0), (0, k - topk_ids.shape[1])), constant_values=-1
        )
        ground_truths[filter] = topk_ids.tolist()

    if cache_path is not None:
        pkl.dump(
            {"ground_truths": ground_truths, "selectivities": selectivities},
            cache_path.open("wb"),
        )

    return ground_truths, selectivities


def exp_curator_complex_predicate(
    num_filters_per_template: int,
    num_queries: int,
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
    output_path: str | None = None,
    seed: int = 42,
):
    filters = generate_random_filters(num_filters_per_template, 1000, seed)
    print(f"Generated {sum(len(fs) for fs in filters.values())} filters")

    np.random.seed(seed)

    # load dataset
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, metadata = load_dataset(dataset_config)

    # randomly sample query vectors from the test set
    if num_queries > test_vecs.shape[0]:
        raise ValueError(
            f"Number of queries ({num_queries}) exceeds the size of the test set ({test_vecs.shape[0]})"
        )
    query_vecs = test_vecs[
        np.random.choice(test_vecs.shape[0], num_queries, replace=False)
    ]

    # compute ground truth
    all_filters = [filter for fs in filters.values() for filter in fs]
    ground_truths, selectivities = compute_ground_truth(
        query_vecs,
        all_filters,
        train_vecs,
        train_mds,
        k=10,
        cache_dir=gt_cache_dir,
    )

    # initialize index
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

    # query index with filters
    results = []
    n_filters = sum(len(fs) for fs in filters.values())
    pbar = tqdm(total=n_filters * num_queries, desc="Querying index")
    for template, fs in filters.items():
        for filter in fs:
            query_results = []
            query_latencies = []

            for qv in query_vecs:
                pbar.update(1)
                query_start = time.time()
                labels = index.query_with_filter(qv, 10, filter)
                query_latencies.append(time.time() - query_start)
                query_results.append(labels)

            avg_recall = recall(query_results, ground_truths[filter]).item()
            avg_search_lat = np.array(query_latencies).mean()

            results.append(
                {
                    "template": template,
                    "filter": filter,
                    "selectivity": selectivities[filter],
                    "nlist": nlist,
                    "max_sl_size": max_sl_size,
                    "max_leaf_size": max_leaf_size,
                    "clus_niter": clus_niter,
                    "update_bf_interval": update_bf_interval,
                    "nprobe": nprobe,
                    "prune_thres": prune_thres,
                    "variance_boost": variance_boost,
                    "avg_recall": avg_recall,
                    "avg_search_lat": avg_search_lat,
                }
            )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False)

    return results


def exp_shared_hnsw_complex_predicate(
    num_filters_per_template: int,
    num_queries: int,
    construction_ef: int = 32,
    search_ef: int = 16,
    m: int = 32,
    max_elements: int = 1000000,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    gt_cache_dir: str | None = None,
    output_path: str | None = None,
    seed: int = 42,
):
    filters = generate_random_filters(num_filters_per_template, 1000, seed)
    print(f"Generated {sum(len(fs) for fs in filters.values())} filters")

    np.random.seed(seed)

    # load dataset
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, metadata = load_dataset(dataset_config)

    # randomly sample query vectors from the test set
    if num_queries > test_vecs.shape[0]:
        raise ValueError(
            f"Number of queries ({num_queries}) exceeds the size of the test set ({test_vecs.shape[0]})"
        )
    query_vecs = test_vecs[
        np.random.choice(test_vecs.shape[0], num_queries, replace=False)
    ]

    # compute ground truth
    all_filters = [filter for fs in filters.values() for filter in fs]
    ground_truths, selectivities = compute_ground_truth(
        query_vecs,
        all_filters,
        train_vecs,
        train_mds,
        k=10,
        cache_dir=gt_cache_dir,
    )

    index_config = IndexConfig(
        index_cls=HNSWMultiTenantHnswlib,
        index_params={
            "construction_ef": construction_ef,
            "m": m,
            "max_elements": max_elements,
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

    # query index with filters
    results = []
    n_filters = sum(len(fs) for fs in filters.values())
    pbar = tqdm(total=n_filters * num_queries, desc="Querying index")
    for template, fs in filters.items():
        for filter in fs:
            query_results = []
            query_latencies = []

            for qv in query_vecs:
                pbar.update(1)
                query_start = time.time()
                labels = index.query_with_filter(qv, 10, filter)
                query_latencies.append(time.time() - query_start)
                query_results.append(labels)

            avg_recall = recall(query_results, ground_truths[filter]).item()
            avg_search_lat = np.array(query_latencies).mean()

            results.append(
                {
                    "template": template,
                    "filter": filter,
                    "selectivity": selectivities[filter],
                    "construction_ef": construction_ef,
                    "search_ef": search_ef,
                    "m": m,
                    "avg_recall": avg_recall,
                    "avg_search_lat": avg_search_lat,
                }
            )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False)

    return results


def plot_figure():
    df_curator = pd.read_csv("curator_complex_predicate.csv")
    df_hnsw = pd.read_csv("shared_hnsw_complex_predicate.csv")

    df_curator_agg = df_curator.groupby("template").agg(
        avg_selectivity=("avg_selectivity", "mean"),
        avg_recall=("avg_recall", "mean"),
        avg_search_lat=("avg_search_lat", "mean"),
        std_selectivity=("avg_selectivity", "std"),
        std_recall=("avg_recall", "std"),
        std_search_lat=("avg_search_lat", "std"),
    )
    df_curator_agg = df_curator_agg.reset_index()

    df_hnsw_agg = df_hnsw.groupby("template").agg(
        avg_selectivity=("avg_selectivity", "mean"),
        avg_recall=("avg_recall", "mean"),
        avg_search_lat=("avg_search_lat", "mean"),
        std_selectivity=("avg_selectivity", "std"),
        std_recall=("avg_recall", "std"),
        std_search_lat=("avg_search_lat", "std"),
    )
    df_hnsw_agg = df_hnsw_agg.reset_index()

    df_curator_agg["index_type"] = "Curator"
    df_hnsw_agg["index_type"] = "Shared HNSW"
    print("Aggregated results:")
    print(pd.concat([df_curator_agg, df_hnsw_agg]))

    df_curator["index_type"] = "Curator"
    df_hnsw["index_type"] = "Shared HNSW"
    df_merged = pd.concat([df_curator, df_hnsw])
    df_merged = df_merged.rename(columns={"index_type": "Index Type"})

    plt.rcParams.update({"font.size": 16})

    sns.catplot(
        data=df_merged,
        kind="bar",
        x="template",
        y="avg_recall",
        hue="Index Type",
        errorbar="sd",
        height=6,
    )
    plt.xlabel("Predicate Template")
    plt.ylabel("Average Recall@10")

    plt.savefig("complex_predicate_avg_recall.png", dpi=200)

    plt.clf()

    sns.catplot(
        data=df_merged,
        kind="bar",
        x="template",
        y="avg_search_lat",
        hue="Index Type",
        errorbar="sd",
        height=6,
    )
    plt.xlabel("Predicate Template")
    plt.ylabel("Average Search Latency (s)")
    plt.yscale("log")

    plt.savefig("complex_predicate_avg_search_lat.png", dpi=200)

    plt.clf()

    sns.barplot(
        data=df_merged,
        x="template",
        y="avg_selectivity",
        errorbar="sd",
    )
    plt.xlabel("Predicate Template")
    plt.ylabel("Average Selectivity")
    plt.yscale("log")

    plt.savefig("complex_predicate_avg_selectivity.png", dpi=200)


if __name__ == "__main__":
    fire.Fire()
