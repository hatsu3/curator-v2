import json
import pickle as pkl
from collections import Counter
from pathlib import Path

import fire
import numpy as np

from benchmark.profiler import Dataset
from dataset.utils import compute_ground_truth_cuda


class SkewnessDataset(Dataset):
    def __init__(
        self,
        train_vecs: np.ndarray,
        train_mds: list[list[int]],
        test_vecs: np.ndarray,
        test_mds: list[list[int]],
        all_labels: set[int],
        ground_truth: np.ndarray,
        dataset_key: str,
        test_size: float,
        seed: int,
    ):
        super().__init__(
            train_vecs, train_mds, test_vecs, test_mds, ground_truth, all_labels
        )

        self.dataset_key = dataset_key
        self.test_size = test_size
        self.seed = seed

    @classmethod
    def from_dataset_key(
        cls,
        dataset_key: str,
        test_size: float,
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ):
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

        # distribution of number of labels per vectors
        n_labels_per_vec = Counter(len(mds) for mds in dataset.train_mds)
        nlpv_vals = np.array(list(n_labels_per_vec.keys()))
        nlpv_probs = np.array(list(n_labels_per_vec.values())) / len(dataset.train_mds)
        assert np.isclose(nlpv_probs.sum(), 1)

        # distribution of labels
        label_freq = Counter()
        for access_list in dataset.train_mds:
            label_freq.update(access_list)

        label_vals = np.array(list(label_freq.keys()))
        label_probs = (
            np.array(list(label_freq.values())) / dataset.num_vector_label_pairs()
        )
        assert np.isclose(label_probs.sum(), 1)

        np.random.seed(seed)
        train_mds, test_mds = [], []

        for _ in dataset.train_vecs:
            n_labels = np.random.choice(nlpv_vals, p=nlpv_probs)
            labels = np.random.choice(
                label_vals, size=n_labels, replace=False, p=label_probs
            )
            train_mds.append(labels)

        for _ in dataset.test_vecs:
            n_labels = np.random.choice(nlpv_vals, p=nlpv_probs)
            labels = np.random.choice(
                label_vals, size=n_labels, replace=False, p=label_probs
            )
            test_mds.append(labels)

        print(
            "Average number of labels per vector in train set:",
            np.mean([len(mds) for mds in train_mds]),
        )

        ground_truth = compute_ground_truth_cuda(
            dataset.train_vecs,
            train_mds,
            dataset.test_vecs,
            test_mds,
            dataset.all_labels,
        )

        result = cls(
            train_vecs=dataset.train_vecs,
            train_mds=train_mds,
            test_vecs=dataset.test_vecs,
            test_mds=test_mds,
            all_labels=set(label_vals),
            ground_truth=ground_truth,
            dataset_key=dataset_key,
            test_size=test_size,
            seed=seed,
        )

        if cache_dir is not None:
            result.save(cache_dir)

        return result

    @classmethod
    def load(cls, cache_dir: str | Path):
        print(f"Loading synthesized dataset from {cache_dir} ...", flush=True)

        cache_dir = Path(cache_dir)
        meta = json.load(open(cache_dir / "meta.json"))
        dataset = Dataset.from_dataset_key(
            meta["dataset_key"], test_size=meta["test_size"]
        )

        access_lists = pkl.load(open(cache_dir / "access_lists.pkl", "rb"))
        train_mds, test_mds = access_lists["train_mds"], access_lists["test_mds"]
        ground_truth = np.load(cache_dir / "ground_truth.npy")

        all_labels = set()
        for mds in train_mds:
            all_labels.update(mds)

        return cls(
            train_vecs=dataset.train_vecs,
            train_mds=train_mds,
            test_vecs=dataset.test_vecs,
            test_mds=test_mds,
            all_labels=all_labels,
            ground_truth=ground_truth,
            dataset_key=meta["dataset_key"],
            test_size=meta["test_size"],
            seed=meta["seed"],
        )

    def save(self, cache_dir: str | Path):
        print(f"Saving synthesized dataset to {cache_dir} ...", flush=True)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        np.save(cache_dir / "ground_truth.npy", self.ground_truth)
        pkl.dump(
            {
                "train_mds": self.train_mds,
                "test_mds": self.test_mds,
            },
            open(cache_dir / "access_lists.pkl", "wb"),
        )

        json.dump(
            {
                "dataset_key": self.dataset_key,
                "test_size": self.test_size,
                "seed": self.seed,
            },
            open(cache_dir / "meta.json", "w"),
        )


def generate_dataset(
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    cache_dir: str = "data/skewness/random_yfcc100m",
    seed: int = 42,
):
    print(f"Generating dataset {dataset_key} ...")
    print(f"Storing ground truth in {cache_dir} ...")
    SkewnessDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        cache_dir=cache_dir,
        seed=seed,
    )


if __name__ == "__main__":
    fire.Fire()

