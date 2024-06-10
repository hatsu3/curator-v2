from functools import cache
from itertools import product
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from benchmark.profile_curator_ood import load_dataset
from benchmark.profile_curator_py import get_curator_index, train_index
from benchmark.utils import get_dataset_config
from indexes.curator_py import CuratorIndexPy, CuratorNode

CuratorIndexPy.__hash__ = lambda self: id(self)


@cache
def per_tenant_flat_index(index: CuratorIndexPy, tenant: int) -> list[CuratorNode]:
    def _flatten(node: CuratorNode) -> list[CuratorNode]:
        if tenant not in node.bloom_filter:
            return []
        elif tenant in node.shortlists:
            return [node]
        else:
            return sum((_flatten(child) for child in node.children), [])

    assert index.root is not None, "Index is not trained yet"
    return _flatten(index.root)


def containing_shortlist(index: CuratorIndexPy, tenant: int, label: int) -> CuratorNode:
    node = index.vec2leaf[label]
    while node is not None:
        if tenant in node.shortlists:
            assert label in node.shortlists[tenant], f"Label {label} not in shortlist"
            return node
        node = node.parent

    raise RuntimeError("Cannot find shortlist")


def flat_search_recall(
    index: CuratorIndexPy,
    tenant: int,
    q: np.ndarray,
    nprobes: list[int],
    ground_truth: list[int],
) -> list[float]:
    def node_score(node: CuratorNode) -> float:
        dist = np.linalg.norm(q - node.centroid).item()
        var = node.variance.get()
        return dist - index.param.var_boost * var

    flat_index = per_tenant_flat_index(index, tenant)
    flat_index.sort(key=node_score)

    nprobes2 = sorted(nprobes)
    flat_recalls = list()
    scanned_vecs = set()

    for node in flat_index:
        if not nprobes2:
            break

        scanned_vecs.update(node.shortlists[tenant])

        while nprobes2 and len(scanned_vecs) >= nprobes2[0]:
            recall = len(set(ground_truth) & scanned_vecs) / len(ground_truth)
            flat_recalls.append(recall)
            nprobes2.pop(0)

    if nprobes2:
        flat_recalls.extend([1.0] * len(nprobes2))

    return flat_recalls


def exp_eval_tree_search(
    nprobes: list[int] = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
    index_path: str = "curator_py_yfcc100m.index",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    output_path: str = "eval_tree_search.csv",
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
        index = get_curator_index(dim=train_vecs.shape[1])

        print(f"Training index...", flush=True)
        train_index(index, train_vecs, train_mds)

        print(f"Saving index to {index_path}...", flush=True)
        index.save(index_path)

    test_vecs = test_vecs[:sample_test_size]
    test_mds = test_mds[:sample_test_size]

    ground_truth_gen = iter(ground_truth)

    flat_recalls = []
    for vec, access_list in tqdm(
        zip(test_vecs, test_mds),
        total=len(test_vecs),
        desc="Flat search",
    ):
        for tenant in access_list:
            recalls = flat_search_recall(
                index, tenant, vec, nprobes, next(ground_truth_gen)
            )
            flat_recalls.append(recalls)

    flat_recalls = pd.DataFrame(flat_recalls, columns=nprobes)
    flat_recalls = flat_recalls.melt(var_name="nprobes", value_name="recall")

    curator_recalls = []
    for nprobe in nprobes:
        index.param.nprobe = nprobe

        results = list()
        for vec, access_list in tqdm(
            zip(test_vecs, test_mds),
            total=len(test_vecs),
            desc=f"Curator search (nprobe={nprobe})",
        ):
            for tenant in access_list:
                pred = index.search(vec, 10, tenant)
                results.append(pred)

        recalls = list()
        for pred, truth in zip(results, ground_truth):
            truth = [t for t in truth if t != -1]
            recalls.append(len(set(pred) & set(truth)) / len(truth))

        curator_recalls.append(recalls)

    curator_recalls = pd.DataFrame(np.array(curator_recalls).T, columns=nprobes)
    curator_recalls = curator_recalls.melt(var_name="nprobes", value_name="recall")

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined_df = pd.concat([flat_recalls, curator_recalls], keys=["flat", "curator"])
    combined_df = combined_df.reset_index(level=0, names="index type")
    combined_df.to_csv(output_path, index=False)


def plot_eval_tree_search_results(
    result_path: str = "eval_tree_search.csv",
    output_path: str = "eval_tree_search.png",
):
    print(f"Loading results from {result_path} ...", flush=True)
    df = pd.read_csv(result_path)

    print(f"Plotting results ...", flush=True)
    sns.lineplot(data=df, x="nprobes", y="recall", hue="index type", errorbar="sd")

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def flat_search_min_ndists(
    index: CuratorIndexPy,
    tenant: int,
    q: np.ndarray,
    ground_truth: list[int],
) -> list[int]:
    def node_score(node: CuratorNode) -> float:
        dist = np.linalg.norm(q - node.centroid).item()
        var = node.variance.get()
        return dist - index.param.var_boost * var

    flat_index = per_tenant_flat_index(index, tenant)
    flat_index.sort(key=node_score)

    ground_truth_set = set(ground_truth)

    min_ndists = list()
    ndists = 0
    for node in flat_index:
        ndists += len(node.shortlists[tenant])
        gt_found = ground_truth_set & set(node.shortlists[tenant])
        min_ndists.extend([ndists] * len(gt_found))

        ground_truth_set -= gt_found
        if not ground_truth_set:
            break

    assert not ground_truth_set, "Some ground truth vectors are not found"
    assert len(min_ndists) == len(ground_truth)

    return min_ndists


def flat_search_ndists_pruned(
    index: CuratorIndexPy,
    tenant: int,
    q: np.ndarray,
    prune_thres: float,
    nprobe: int,
    ground_truth: list[int],
) -> tuple[int, float]:
    def node_score(node: CuratorNode) -> float:
        dist = np.linalg.norm(q - node.centroid).item()
        var = node.variance.get()
        return dist - index.param.var_boost * var

    flat_index = per_tenant_flat_index(index, tenant)
    flat_index.sort(key=node_score)

    best_score = node_score(flat_index[0])
    ndists = 0
    nfound = 0

    for node in flat_index:
        ndists += len(node.shortlists[tenant])
        if ndists > nprobe:
            break

        if node_score(node) > best_score * (1 + prune_thres):
            break

        gt_found = set(ground_truth) & set(node.shortlists[tenant])
        nfound += len(gt_found)

    recall = nfound / len(ground_truth)
    return ndists, recall


def exp_eval_exit_condition(
    prune_threses: list[float] = [0.2, 0.4, 0.6, 0.8],
    nprobes: list[int] = list(range(500, 4500, 500)),
    index_path: str = "curator_py_yfcc100m.index",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    output_path: str = "eval_exit_condition.csv",
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
        index = get_curator_index(dim=train_vecs.shape[1])

        print(f"Training index...", flush=True)
        train_index(index, train_vecs, train_mds)

        print(f"Saving index to {index_path}...", flush=True)
        index.save(index_path)

    test_vecs = test_vecs[:sample_test_size]
    test_mds = test_mds[:sample_test_size]

    ground_truth_gen = iter(ground_truth)

    optimal_ndists = []
    for vec, access_list in tqdm(
        zip(test_vecs, test_mds),
        total=len(test_vecs),
        desc="Calculating optimal ndists",
    ):
        for tenant in access_list:
            gt = next(ground_truth_gen)
            if len(gt) != 10:
                continue

            min_ndists = flat_search_min_ndists(index, tenant, vec, gt)
            optimal_ndists.append(min_ndists)

    optimal_ndists = pd.DataFrame(
        optimal_ndists, columns=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    optimal_ndists = optimal_ndists.melt(var_name="recall", value_name="ndists")

    output_path_optimal = output_path.replace(".csv", "_optimal.csv")
    print(f"Saving results to {output_path_optimal}...", flush=True)
    Path(output_path_optimal).parent.mkdir(parents=True, exist_ok=True)
    optimal_ndists.to_csv(output_path_optimal, index=False)

    results = []
    for prune_thres, nprobe in tqdm(
        product(prune_threses, nprobes),
        total=len(prune_threses) * len(nprobes),
        desc="Calculating actual ndists",
    ):
        ground_truth_gen = iter(ground_truth)

        for vec, access_list in zip(test_vecs, test_mds):
            for tenant in access_list:
                gt = next(ground_truth_gen)
                if len(gt) != 10:
                    continue

                ndists, recall = flat_search_ndists_pruned(
                    index, tenant, vec, prune_thres, nprobe, gt
                )
                results.append(
                    {
                        "prune_thres": prune_thres,
                        "nprobe": nprobe,
                        "ndists": ndists,
                        "recall": recall,
                    }
                )

    results = pd.DataFrame(results)

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)


def plot_eval_exit_condition_results(
    result_path: str = "eval_exit_condition.csv",
    output_path: str = "eval_exit_condition.png",
):
    print(f"Loading results from {result_path} ...", flush=True)
    optimal_df = pd.read_csv(result_path.replace(".csv", "_optimal.csv"))
    optimal_df = pd.concat(
        [optimal_df, pd.DataFrame({"recall": [0.0], "ndists": [0]})], ignore_index=True
    )
    results_df = pd.read_csv(result_path)
    results_df = results_df.groupby(["prune_thres", "nprobe"]).agg(
        {"recall": "mean", "ndists": "mean"}
    )
    results_df = results_df.reset_index()

    print(f"Plotting results ...", flush=True)
    sns.lineplot(
        data=optimal_df,
        x="recall",
        y="ndists",
        label="Optimal",
        errorbar=("pi", 50),
    )
    sns.lineplot(
        data=results_df,
        x="recall",
        y="ndists",
        hue="nprobe",
        errorbar=None,
    )

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def plot_topk_radius(
    cumulative: bool = False,
    result_path: str = "eval_exit_condition_optimal.csv",
    output_path: str = "topk_radius.png",
):
    print(f"Loading results from {result_path} ...", flush=True)
    df = pd.read_csv(result_path)

    print(f"Plotting results ...", flush=True)
    sns.kdeplot(
        data=df,
        x="ndists",
        hue="recall",
        log_scale=(True, False),
        cumulative=cumulative,
    )

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    fire.Fire()
