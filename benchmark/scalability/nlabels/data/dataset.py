import json
import pickle as pkl
from pathlib import Path

import fire
import numpy as np

from benchmark.profiler import Dataset
from dataset.utils import compute_ground_truth_cuda


class ScalabilityDataset(Dataset):
    def __init__(
        self,
        train_vecs: np.ndarray,
        train_mds: list[list[int]],
        test_vecs: np.ndarray,
        test_mds: list[list[int]],
        ground_truth: np.ndarray,
        dataset_key: str,
        test_size: float,
        n_labels: int,
        label_selectivity: float,
        seed: int = 42,
    ):
        super().__init__(
            train_vecs=train_vecs,
            train_mds=train_mds,
            test_vecs=test_vecs,
            test_mds=test_mds,
            ground_truth=ground_truth,
            all_labels=set(range(n_labels)),
        )

        self.dataset_key = dataset_key
        self.test_size = test_size
        self.n_labels = n_labels
        self.label_selectivity = label_selectivity
        self.seed = seed

    @classmethod
    def from_dataset_key(
        cls,
        dataset_key: str,
        test_size: float,
        n_labels: int,
        label_selectivity: float,
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ):
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

        train_mds = [[] for _ in dataset.train_vecs]
        test_mds = [[] for _ in dataset.test_vecs]

        np.random.seed(seed)
        for label in range(n_labels):
            train_mask = np.random.rand(dataset.train_vecs.shape[0]) < label_selectivity
            test_mask = np.random.rand(dataset.test_vecs.shape[0]) < label_selectivity

            for id in np.where(train_mask)[0]:
                train_mds[id].append(label)

            for id in np.where(test_mask)[0]:
                test_mds[id].append(label)

        ground_truth = compute_ground_truth_cuda(
            dataset.train_vecs,
            train_mds,
            dataset.test_vecs,
            test_mds,
            set(range(n_labels)),
        )

        result = cls(
            train_vecs=dataset.train_vecs,
            train_mds=train_mds,
            test_vecs=dataset.test_vecs,
            test_mds=test_mds,
            ground_truth=ground_truth,
            dataset_key=dataset_key,
            test_size=test_size,
            n_labels=n_labels,
            label_selectivity=label_selectivity,
            seed=seed,
        )

        if cache_dir is not None:
            result.save(cache_dir)

        return result

    @classmethod
    def load(cls, cache_dir: str | Path):
        print(f"Loading dataset from {cache_dir} ...", flush=True)

        cache_dir = Path(cache_dir)
        meta = json.load(open(cache_dir / "meta.json"))
        dataset = Dataset.from_dataset_key(
            meta["dataset_key"], test_size=meta["test_size"]
        )

        acces_lists = pkl.load(open(cache_dir / "access_lists.pkl", "rb"))
        train_mds, test_mds = acces_lists["train_mds"], acces_lists["test_mds"]
        ground_truth = np.load(cache_dir / "ground_truth.npy")

        print(f"  n_labels: {meta['n_labels']}")
        print(f"  label_selectivity: {meta['label_selectivity']}")

        return cls(
            train_vecs=dataset.train_vecs,
            train_mds=train_mds,
            test_vecs=dataset.test_vecs,
            test_mds=test_mds,
            ground_truth=ground_truth,
            dataset_key=meta["dataset_key"],
            test_size=meta["test_size"],
            n_labels=meta["n_labels"],
            label_selectivity=meta["label_selectivity"],
            seed=meta["seed"],
        )

    def save(self, cache_dir: str | Path):
        print(f"Saving dataset to {cache_dir} ...", flush=True)

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
                "n_labels": self.n_labels,
                "label_selectivity": self.label_selectivity,
                "seed": self.seed,
            },
            open(cache_dir / "meta.json", "w"),
            indent=4,
        )


def generate_dataset(
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    n_labels: int = 10000,
    label_selectivity: float = 0.01,
    cache_dir: str | Path = "data/scalability/random_yfcc100m",
    seed: int = 42,
):
    print(f"Generating dataset {dataset_key} ...")
    print(f"Storing ground truth in {cache_dir} ...")
    ScalabilityDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        n_labels=n_labels,
        label_selectivity=label_selectivity,
        cache_dir=cache_dir,
        seed=seed,
    )


if __name__ == "__main__":
    fire.Fire()
