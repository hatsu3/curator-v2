import hashlib
import json
import pickle as pkl
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from benchmark.config import IndexConfig
from benchmark.utils import get_dataset_config, load_dataset, recall
from indexes.hnsw_mt_hnswlib import HNSWMultiTenantHnswlib
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss
from indexes.parlay_ivf import ParlayIVF


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


def generate_dataset(
    num_filters_per_template: int,
    num_queries: int,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    gt_cache_dir: str | None = None,
    seed: int = 42,
):
    filters = generate_random_filters(num_filters_per_template, 1000, seed)
    print(f"Generated {sum(len(fs) for fs in filters.values())} filters")

    np.random.seed(seed)

    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, __, __ = load_dataset(dataset_config)

    # randomly sample query vectors from the test set
    if num_queries > test_vecs.shape[0]:
        raise ValueError(
            f"Number of queries ({num_queries}) exceeds the size of the test set ({test_vecs.shape[0]})"
        )
    query_vecs = test_vecs[
        np.random.choice(test_vecs.shape[0], num_queries, replace=False)
    ]

    all_filters = [filter for fs in filters.values() for filter in fs]
    ground_truths, selectivities = compute_ground_truth(
        query_vecs,
        all_filters,
        train_vecs,
        train_mds,
        k=10,
        cache_dir=gt_cache_dir,
    )

    return train_vecs, query_vecs, train_mds, filters, ground_truths, selectivities, dim


def get_curator_index(
    dim: int,
    nlist: int = 16,
    max_sl_size: int = 128,
    prune_thres: float = 1.6,
    nprobe: int = 4000,
    seed: int = 42,
) -> IVFFlatMultiTenantBFHierFaiss:
    index_config = IndexConfig(
        index_cls=IVFFlatMultiTenantBFHierFaiss,
        index_params={
            "d": dim,
            "nlist": nlist,
            "bf_capacity": 1000,
            "bf_error_rate": 0.01,
            "max_sl_size": max_sl_size,
            "update_bf_interval": 100,
            "clus_niter": 20,
            "max_leaf_size": max_sl_size,
        },
        search_params={
            "nprobe": nprobe,
            "prune_thres": prune_thres,
            "variance_boost": 0.4,
        },
        train_params={
            "train_ratio": 1,
            "min_train": 50,
            "random_seed": seed,
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
    max_elements: int = 1000000,
) -> HNSWMultiTenantHnswlib:
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

    return index


def get_parlay_ivf_index(
    index_dir: str,
    ivf_cluster_size: int = 500,
    ivf_max_iter: int = 10,
    graph_degree: int = 16,
    ivf_search_radius: int = 1000,
    graph_search_L: int = 50,
    build_threads: int = 16,
) -> ParlayIVF:
    if not index_dir.endswith("/"):
        index_dir += "/"

    return ParlayIVF(
        index_dir=index_dir,
        cluster_size=ivf_cluster_size,
        max_iter=ivf_max_iter,
        weight_classes=[100000, 400000],
        max_degrees=[graph_degree] * 3,
        bitvector_cutoff=10000,
        target_points=ivf_search_radius,
        tiny_cutoff=1000,
        beam_widths=[graph_search_L] * 3,
        search_limits=[100000, 400000, 3000000],
        build_threads=build_threads,
    )


def exp_curator_complex_predicate(
    num_filters_per_template: int,
    num_queries: int,
    nlist: int = 16,
    max_sl_size: int = 256,
    prune_thres: float = 1.6,
    nprobe: int = 4000,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    gt_cache_dir: str | None = None,
    output_path: str | None = None,
    seed: int = 42,
):
    print("Generating dataset...", flush=True)
    train_vecs, query_vecs, train_mds, filters, ground_truths, selectivities, dim = (
        generate_dataset(
            num_filters_per_template,
            num_queries,
            dataset_key,
            test_size,
            gt_cache_dir,
            seed,
        )
    )

    print("Initializing index...", flush=True)
    index = get_curator_index(dim, nlist, max_sl_size, prune_thres, nprobe, seed)

    print("Training index...", flush=True)
    index.train(train_vecs, train_mds)

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
                    "nlist": index.nlist,
                    "max_sl_size": index.max_sl_size,
                    "max_leaf_size": index.max_leaf_size,
                    "clus_niter": index.clus_niter,
                    "update_bf_interval": index.update_bf_interval,
                    "nprobe": index.nprobe,
                    "prune_thres": index.prune_thres,
                    "variance_boost": index.variance_boost,
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
    print("Generating dataset...", flush=True)
    train_vecs, query_vecs, train_mds, filters, ground_truths, selectivities, dim = (
        generate_dataset(
            num_filters_per_template,
            num_queries,
            dataset_key,
            test_size,
            gt_cache_dir,
            seed,
        )
    )

    print("Initializing index...", flush=True)
    index = get_shared_hnsw_index(construction_ef, search_ef, m, max_elements)

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


def exp_parlay_ivf_complex_predicate(
    num_filters_per_template: int,
    num_queries: int,
    ivf_cluster_size: int = 500,
    ivf_max_iter: int = 10,
    graph_degree: int = 16,
    ivf_search_radius: int = 1000,
    graph_search_L: int = 50,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    gt_cache_dir: str | None = None,
    output_dir: str = "output/complex_predicate/parlay_ivf",
    seed: int = 42,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    index_dir = Path(output_dir) / "parlay_ivf.index"

    print("Generating dataset...", flush=True)
    train_vecs, query_vecs, train_mds, filters, ground_truths, selectivities, dim = (
        generate_dataset(
            num_filters_per_template,
            num_queries,
            dataset_key,
            test_size,
            gt_cache_dir,
            seed,
        )
    )

    print("Initializing index...", flush=True)
    index = get_parlay_ivf_index(
        str(index_dir),
        ivf_cluster_size,
        ivf_max_iter,
        graph_degree,
        ivf_search_radius,
        graph_search_L,
    )

    print("Training index...", flush=True)
    index.train(train_vecs, train_mds)

    results = []
    n_filters = len(filters["AND {0} {1}"])
    pbar = tqdm(total=n_filters * num_queries, desc="Querying index")

    for filter in filters["AND {0} {1}"]:
        query_results = []
        query_latencies = []

        for qv in query_vecs:
            pbar.update(1)
            query_start = time.time()
            labels = index.batch_and_query(qv[None], 10, [filter])[0]
            query_latencies.append(time.time() - query_start)
            query_results.append(labels)

        avg_recall = recall(query_results, ground_truths[filter]).item()
        avg_search_lat = np.array(query_latencies).mean()

        results.append(
            {
                "template": "AND {0} {1}",
                "filter": filter,
                "selectivity": selectivities[filter],
                "ivf_cluster_size": ivf_cluster_size,
                "ivf_max_iter": ivf_max_iter,
                "graph_degree": graph_degree,
                "ivf_search_radius": ivf_search_radius,
                "graph_search_L": graph_search_L,
                "avg_recall": avg_recall,
                "avg_search_lat": avg_search_lat,
            }
        )

    print(f"Saving results to {output_dir} ...")
    Path(output_dir).parent.mkdir(parents=True, exist_ok=True)
    results_path = Path(output_dir) / "results.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)

    config_path = Path(output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "num_filters_per_template": num_filters_per_template,
                "num_queries": num_queries,
                "ivf_cluster_size": ivf_cluster_size,
                "ivf_max_iter": ivf_max_iter,
                "graph_degree": graph_degree,
                "ivf_search_radius": ivf_search_radius,
                "graph_search_L": graph_search_L,
                "dataset_key": dataset_key,
                "test_size": test_size,
                "seed": seed,
            },
            f,
            indent=4,
        )


def plot_complex_predicate_results(
    results_path: str = "output/complex_predicate/parlay_ivf/results.csv",
    output_path: str = "output/complex_predicate/parlay_ivf/results.png",
):
    print(f"Loading results from {results_path} ...")
    results = pd.read_csv(results_path)
    results["avg_search_lat"] *= 1000

    print("Plotting results ...")
    sns.scatterplot(
        data=results,
        x="avg_recall",
        y="avg_search_lat",
        hue="template",
    )
    plt.xlabel("Recall@10")
    plt.ylabel("Search latency (ms)")

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def test_indexing_subexpression(
    num_filters_per_template: int,
    num_queries: int,
    nlist: int = 16,
    max_sl_size: int = 256,
    prune_thres: float = 1.6,
    nprobe: int = 4000,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    gt_cache_dir: str | None = None,
    seed: int = 42,
    output_dir: str = "output/complex_predicate/index_subexpr",
):
    print("Generating dataset...", flush=True)
    train_vecs, query_vecs, train_mds, filters, ground_truths, selectivities, dim = (
        generate_dataset(
            num_filters_per_template,
            num_queries,
            dataset_key,
            test_size,
            gt_cache_dir,
            seed,
        )
    )

    print("Initializing index...", flush=True)
    index = get_curator_index(dim, nlist, max_sl_size, prune_thres, nprobe, seed)

    print("Training index...", flush=True)
    index.train(train_vecs, train_mds)

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

    index.index.sanity_check()

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
                    "avg_recall": avg_recall,
                    "avg_search_lat": avg_search_lat,
                }
            )

    print("Before indexing subexpression:")
    results_df = pd.DataFrame(results)
    print(results_df)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/perf_before_index.csv"
    print(f"Saving results to {output_path} ...")
    results_df.to_csv(output_path, index=False)

    # index subexpression
    print("Indexing subexpression...", flush=True)
    for __, fs in filters.items():
        for filter in fs:
            index.index_filter(filter)

    index.index.sanity_check()

    results = []
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
                    "avg_recall": avg_recall,
                    "avg_search_lat": avg_search_lat,
                }
            )

    print("After indexing subexpression:")
    results_df = pd.DataFrame(results)
    print(results_df)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/perf_after_index.csv"
    print(f"Saving results to {output_path} ...")
    results_df.to_csv(output_path, index=False)

    output_path = f"{output_dir}/config.json"
    print(f"Saving experiment configuration to {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(
            {
                "num_filters_per_template": num_filters_per_template,
                "num_queries": num_queries,
                "nlist": nlist,
                "max_sl_size": max_sl_size,
                "prune_thres": prune_thres,
                "nprobe": nprobe,
                "dataset_key": dataset_key,
                "test_size": test_size,
                "gt_cache_dir": gt_cache_dir,
                "seed": seed,
            },
            f,
            indent=4,
        )


def plot_indexing_subexpression(
    results_dir: str = "output/complex_predicate/index_subexpr",
    output_path: str | None = None,
):
    if output_path is None:
        output_path = f"{results_dir}/index_subexpr_perf.png"

    print(f"Loading results from {results_dir} ...")
    results_before = pd.read_csv(f"{results_dir}/perf_before_index.csv")
    results_after = pd.read_csv(f"{results_dir}/perf_after_index.csv")

    results_before["Indexing"] = "Before"
    results_after["Indexing"] = "After"
    results = pd.concat([results_before, results_after])
    results["avg_search_lat"] *= 1000

    print("Plotting results ...")
    sns.scatterplot(
        data=results,
        x="avg_recall",
        y="avg_search_lat",
        hue="Indexing",
        style="template",
    )

    plt.xlabel("Recall@10")
    plt.ylabel("Search Latency (ms)")

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_filtered_search_latency_breakdown(
    log_path: str,
    output_path: str = "output/complex_predicate/perf_analysis/latency_breakdown.png",
):
    import parse

    with open(log_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(reversed(lines)):
        if "n_invocations" in line:
            lines = lines[-i:]

    results = dict()
    keys = [
        "update_var_map_time",
        "eval_filter_time",
        "infer_child_time",
        "heap_time",
        "dist_time",
        "total_time",
    ]

    for key, line in zip(keys, lines):
        res = parse.parse(f"{key}: {{val}}", line)
        assert isinstance(res, parse.Result), f"Failed to parse {key}: {line}"
        results[key] = float(res["val"])

    total = results.pop("total_time")
    results["misc"] = total - sum(results.values())

    df = pd.DataFrame(results.items(), columns=["Operation", "Percentage"])
    df.plot.pie(y="Percentage", labels=df["Operation"], autopct="%1.1f%%", legend=False)
    plt.ylabel("")

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


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
