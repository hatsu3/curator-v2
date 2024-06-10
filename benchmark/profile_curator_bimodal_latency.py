from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from benchmark.profile_curator_py import get_curator_index, train_index
from benchmark.utils import get_dataset_config, load_dataset
from indexes.curator_py import CuratorIndexPy, CuratorParam


def query_index(
    index: CuratorIndexPy,
    test_vecs: np.ndarray,
    test_mds: list[list[int]],
    ground_truth: list[list[int]],
):
    results = list()
    ndists = list()
    for vec, access_list in tqdm(
        zip(test_vecs, test_mds),
        total=len(test_vecs),
        desc="Querying index",
    ):
        for tenant in access_list:
            pred = index.search(vec, 10, tenant)
            results.append(pred)
            ndists.append(index.search_stats["ndists"])

    recalls = list()
    for pred, truth in zip(results, ground_truth):
        truth = [t for t in truth if t != -1]
        recalls.append(len(set(pred) & set(truth)) / len(truth))

    return recalls, ndists


def exp_curator_bimodal_latency(
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    sample_test_size: int = 100,
    index_path: str = "curator_py_yfcc100m.index",
    output_path: str = "curator_bimodal_latency.csv",
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

    print(f"Querying index...", flush=True)
    test_vecs = test_vecs[:sample_test_size]
    test_mds = test_mds[:sample_test_size]
    recalls, ndists = query_index(index, test_vecs, test_mds, ground_truth.tolist())
    print(f"Recall@10: {np.mean(recalls):.4f}")

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"recall": recalls, "ndists": ndists})
    df.to_csv(output_path, index=False)


def plot_results(
    results_path: str = "curator_bimodal_latency.csv",
    output_path: str = "curator_bimodal_latency.png",
):
    print(f"Loading results from {results_path}...", flush=True)
    df = pd.read_csv(results_path)

    print(f"Plotting results...", flush=True)
    sns.kdeplot(data=df, x="ndists")

    print(f"Saving plot to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


if __name__ == "__main__":
    fire.Fire()
