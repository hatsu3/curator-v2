import multiprocessing
import time
from itertools import product
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark.utils import get_dataset_config, load_dataset, get_memory_usage
from indexes.parlay_ivf import ParlayIVF


def get_parlay_ivf_index(
    index_dir: str,
    ivf_cluster_size: int = 500,
    ivf_max_iter: int = 10,
    graph_degree: int = 8,
    ivf_search_radius: int = 5000,
    graph_search_L: int = 100,
    build_threads: int = 8,
):
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


def exp_profile_parlay_ivf(
    index_dir: str = "parlay_ivf.index",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
):
    print("Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config,
    )

    print("Initializing index...", flush=True)
    mem_before = get_memory_usage()
    index = get_parlay_ivf_index(index_dir=index_dir)

    print("Training index...", flush=True)
    index.train(train_vecs, train_mds)
    mem_usage = get_memory_usage() - mem_before

    print("Querying index...", flush=True)
    begin = time.time()
    results = index.batch_query(test_vecs, 10, test_mds)
    latency = (time.time() - begin) / sum(len(mds) for mds in test_mds)

    recalls = list()
    for pred, truth in zip(results, ground_truth):
        truth = [t for t in truth if t != -1]
        recalls.append(len(set(pred) & set(truth)) / len(truth))

    recall = np.mean(recalls).item()
    print(f"Recall@10: {recall:.4f}", flush=True)
    print(f"Avg latency: {latency:.4f}", flush=True)
    print(f"Memory usage: {mem_usage:.4f} kB", flush=True)


class CSVLogger:
    def __init__(self, path: str, header: list[str], append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.header = header

        print(f"Logging to {path} ...", flush=True)
        if not append or not self.path.exists():
            self.file = self.path.open("w")
            self.file.write(",".join(header) + "\n")
        else:
            self.file = self.path.open("a")

    def log(self, row: dict[str, Any]):
        assert set(row.keys()) == set(self.header)
        self.file.write(",".join(str(row[key]) for key in self.header) + "\n")

    def __del__(self):
        self.file.close()


""" Command:
python -m benchmark.profile_parlay_ivf \
    exp_parlay_ivf_param_sweep \
        --dataset_key "yfcc100m" \
        --test_size 0.01 \
        --ivf_cluster_size_space "[100, 500, 1000]" \
        --ivf_max_iter_space "[10, 20]" \
        --graph_degree_space "[8, 12, 16]" \
        --ivf_search_radius_space "[1000, 3000, 5000]" \
        --graph_search_L_space "[50, 100, 200]" \
        --output_path "parlay_ivf_param_sweep.csv"
"""


def exp_parlay_ivf_param_sweep_worker(
    rank: int,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ivf_cluster_size: int = 500,
    ivf_max_iter: int = 10,
    graph_degree: int = 8,
    ivf_search_radius_space: list[int] = [1000, 3000, 5000],
    graph_search_L_space: list[int] = [50, 100, 200],
    output_path: str = "parlay_ivf_param_sweep.csv",
) -> list[dict]:
    print(f"[Worker {rank}] Loading dataset ...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
        dataset_config,
    )

    logger = CSVLogger(
        path=output_path,
        header=[
            "ivf_cluster_size",
            "ivf_max_iter",
            "graph_degree",
            "ivf_search_radius",
            "graph_search_L",
            "recall",
            "latency",
            "memory_usage_mb",
        ],
        append=True,
    )

    print(
        f"[Worker {rank}] Initializing index with params: {ivf_cluster_size}, {ivf_max_iter}, {graph_degree} ...",
        flush=True,
    )
    mem_before = get_memory_usage()
    index = get_parlay_ivf_index(
        index_dir="",
        ivf_cluster_size=ivf_cluster_size,
        ivf_max_iter=ivf_max_iter,
        graph_degree=graph_degree,
    )

    print(f"[Worker {rank}] Training index ...", flush=True)
    index.train(train_vecs, train_mds)
    mem_usage = (get_memory_usage() - mem_before) / 1024

    results = []
    for ivf_search_radius, graph_search_L in product(
        ivf_search_radius_space, graph_search_L_space
    ):
        index.search_params = {
            "target_points": ivf_search_radius,
            "beam_widths": [graph_search_L] * 3,
        }

        print(
            f"[Worker {rank}] Running query with params: {index.search_params} ...",
            flush=True,
        )

        begin = time.time()
        preds = index.batch_query(test_vecs, 10, test_mds)
        latency = (time.time() - begin) / sum(len(mds) for mds in test_mds)

        recalls = list()
        for pred, truth in zip(preds, ground_truth):
            truth = [t for t in truth if t != -1]
            recalls.append(len(set(pred) & set(truth)) / len(truth))

        recall = np.mean(recalls).item()

        result = {
            "ivf_cluster_size": ivf_cluster_size,
            "ivf_max_iter": ivf_max_iter,
            "graph_degree": graph_degree,
            "ivf_search_radius": ivf_search_radius,
            "graph_search_L": graph_search_L,
            "recall": recall,
            "latency": latency,
            "memory_usage_mb": mem_usage,
        }

        results.append(result)
        logger.log(result)

    del index
    return results


"""
python -m benchmark.profile_parlay_ivf \
    exp_parlay_ivf_param_sweep \
        --dataset_key "yfcc100m" \
        --test_size 0.01 \
        --ivf_cluster_size_space "[100, 500, 1000]" \
        --ivf_max_iter_space "[10, 20]" \
        --graph_degree_space "[8, 12, 16]" \
        --output_path "parlay_ivf_param_sweep.csv" \
        --num_workers 4
"""


"""
Or equivalently using shell scripts:
Command line parameters:
0. rank
1. cpu mask (used in taskset)
2. ivf_cluster_size
3. ivf_max_iter
4. graph_degree

```bash
rank=$1
cpu_mask=$2
ivf_cluster_size=$3
ivf_max_iter=$4
graph_degree=$5

taskset -c $cpu_mask \
python -m benchmark.profile_parlay_ivf \
    exp_parlay_ivf_param_sweep_worker \
        --rank $rank \
        --ivf_cluster_size $ivf_cluster_size \
        --ivf_max_iter $ivf_max_iter \
        --graph_degree $graph_degree \
        --output_path "parlay_ivf_param_sweep.csv"
```
"""


# def exp_parlay_ivf_param_sweep(
#     dataset_key: str = "yfcc100m",
#     test_size: float = 0.01,
#     ivf_cluster_size_space: list[int] = [100, 500, 1000],
#     ivf_max_iter_space: list[int] = [10, 20],
#     graph_degree_space: list[int] = [8, 12, 16],
#     output_path: str = "parlay_ivf_param_sweep.csv",
#     num_workers: int = 4,
# ):
#     with multiprocessing.Pool(num_workers) as pool:
#         results = [
#             pool.apply_async(
#                 exp_parlay_ivf_param_sweep_worker,
#                 (
#                     {
#                         "rank": rank,
#                         "dataset_key": dataset_key,
#                         "test_size": test_size,
#                         "ivf_cluster_size": ivf_cluster_size,
#                         "ivf_max_iter": ivf_max_iter,
#                         "graph_degree": graph_degree,
#                         "output_path": output_path,
#                     }
#                 ),
#             )
#             for rank, (ivf_cluster_size, ivf_max_iter, graph_degree) in enumerate(
#                 product(ivf_cluster_size_space, ivf_max_iter_space, graph_degree_space)
#             )
#         ]

#     results = sum([result.get() for result in results], [])
#     print("Results:", pd.DataFrame(results))


def plot_parlay_ivf_param_sweep(
    results_path: str = "parlay_ivf_param_sweep.csv",
    output_path: str = "parlay_ivf_param_sweep.png",
):
    print(f"Loading results from {results_path} ...", flush=True)
    results = pd.read_csv(results_path)
    results["latency"] = results["latency"] * 1000

    print("Plotting results ...", flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    sns.scatterplot(
        data=results,
        x="latency",
        y="recall",
        hue="ivf_cluster_size",
        style="ivf_max_iter",
        ax=axes[0, 0],
    )

    sns.scatterplot(
        data=results,
        x="latency",
        y="recall",
        hue="graph_degree",
        ax=axes[0, 1],
    )

    sns.scatterplot(
        data=results,
        x="latency",
        y="recall",
        hue="ivf_search_radius",
        ax=axes[1, 0],
    )

    sns.scatterplot(
        data=results,
        x="latency",
        y="recall",
        hue="graph_search_L",
        ax=axes[1, 1],
    )

    plt.tight_layout()

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    fire.Fire()
