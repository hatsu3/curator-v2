import json
import pickle as pkl
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark.profiler import Dataset
from dataset.utils import compute_ground_truth_cuda


class SelectivityDataset(Dataset):
    def __init__(
        self,
        train_vecs: np.ndarray,
        train_mds: list[list[int]],
        test_vecs: np.ndarray,
        test_mds: list[list[int]],
        ground_truth: np.ndarray,
        label_to_selectivity: dict[int, float],
        dataset_key: str,
        test_size: float,
        n_selectivities: int,
        n_labels_per_selectivity: int,
        seed: int,
    ):
        self.train_vecs = train_vecs
        self.train_mds = train_mds
        self.test_vecs = test_vecs
        self.test_mds = test_mds
        self.ground_truth = ground_truth
        self.label_to_selectivity = label_to_selectivity

        self.dataset_key = dataset_key
        self.test_size = test_size
        self.n_selectivities = n_selectivities
        self.n_labels_per_selectivity = n_labels_per_selectivity
        self.seed = seed

    @property
    def all_labels(self) -> set[int]:
        return set(self.label_to_selectivity.keys())
    
    @property
    def all_selectivities(self) -> set[float]:
        return set(self.label_to_selectivity.values())

    def get_selectivity_range_split(
        self, min_sel: float = 0.0, max_sel: float = 1.0
    ) -> "SelectivityDataset":
        split_labels = set(
            label
            for label, sel in self.label_to_selectivity.items()
            if min_sel <= sel <= max_sel
        )

        split_train_mds = [
            [label for label in access_list if label in split_labels]
            for access_list in self.train_mds
        ]
        split_test_mds = [
            [label for label in access_list if label in split_labels]
            for access_list in self.test_mds
        ]

        split_ground_truth = list()
        orig_gt_gen = iter(self.ground_truth)
        for access_list in self.test_mds:
            for label in access_list:
                gt = next(orig_gt_gen)
                if label in split_labels:
                    split_ground_truth.append(gt)
        split_ground_truth = np.array(split_ground_truth)

        return SelectivityDataset(
            train_vecs=self.train_vecs,
            train_mds=split_train_mds,
            test_vecs=self.test_vecs,
            test_mds=split_test_mds,
            ground_truth=split_ground_truth,
            label_to_selectivity={
                label: sel
                for label, sel in self.label_to_selectivity.items()
                if label in split_labels
            },
            dataset_key=self.dataset_key,
            test_size=self.test_size,
            n_selectivities=len(split_labels) // self.n_labels_per_selectivity,
            n_labels_per_selectivity=self.n_labels_per_selectivity,
            seed=self.seed,
        )

    @classmethod
    def from_dataset_key(
        cls,
        dataset_key: str,
        test_size: float,
        n_selectivities: int,
        n_labels_per_selectivity: int,
        cache_dir: str | Path | None = None,
        seed: int = 42,
    ):
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

        train_mds = [[] for _ in dataset.train_vecs]
        test_mds = [[] for _ in dataset.test_vecs]
        label_to_selectivity = {}
        cur_label = 0

        np.random.seed(seed)
        for selectivity in tqdm(
            np.linspace(0.01, 1.00, n_selectivities), total=n_selectivities
        ):
            for _ in range(n_labels_per_selectivity):
                train_mask = np.random.rand(dataset.train_vecs.shape[0]) < selectivity
                test_mask = np.random.rand(dataset.test_vecs.shape[0]) < selectivity

                for id in np.where(train_mask)[0]:
                    train_mds[id].append(cur_label)

                for id in np.where(test_mask)[0]:
                    test_mds[id].append(cur_label)

                label_to_selectivity[cur_label] = selectivity
                cur_label += 1

        ground_truth = compute_ground_truth_cuda(
            dataset.train_vecs,
            train_mds,
            dataset.test_vecs,
            test_mds,
            set(range(cur_label)),
        )

        result = cls(
            train_vecs=dataset.train_vecs,
            train_mds=train_mds,
            test_vecs=dataset.test_vecs,
            test_mds=test_mds,
            ground_truth=ground_truth,
            label_to_selectivity=label_to_selectivity,
            dataset_key=dataset_key,
            test_size=test_size,
            n_selectivities=n_selectivities,
            n_labels_per_selectivity=n_labels_per_selectivity,
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
        label_to_sel = pd.read_csv(cache_dir / "label_to_selectivity.csv")
        label_to_sel = label_to_sel.T.to_dict()[0]
        label_to_sel = {int(k): v for k, v in label_to_sel.items()}

        return cls(
            train_vecs=dataset.train_vecs,
            train_mds=train_mds,
            test_vecs=dataset.test_vecs,
            test_mds=test_mds,
            ground_truth=ground_truth,
            label_to_selectivity=label_to_sel,
            dataset_key=meta["dataset_key"],
            test_size=meta["test_size"],
            n_selectivities=meta["n_selectivities"],
            n_labels_per_selectivity=meta["n_labels_per_selectivity"],
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

        pd.DataFrame(self.label_to_selectivity, index=[0]).to_csv(
            cache_dir / "label_to_selectivity.csv", index=False
        )

        json.dump(
            {
                "dataset_key": self.dataset_key,
                "test_size": self.test_size,
                "n_selectivities": self.n_selectivities,
                "n_labels_per_selectivity": self.n_labels_per_selectivity,
                "seed": self.seed,
            },
            open(cache_dir / "meta.json", "w"),
            indent=4,
        )


def generate_dataset(
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    n_selectivities: int = 20,
    n_labels_per_selectivity: int = 10,
    cache_dir: str = "data/selectivity/random_yfcc100m",
    seed: int = 42,
):
    print(f"Generating dataset {dataset_key} ...")
    print(f"Storing ground truth in {cache_dir} ...")
    SelectivityDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        n_selectivities=n_selectivities,
        n_labels_per_selectivity=n_labels_per_selectivity,
        cache_dir=cache_dir,
        seed=seed,
    )


if __name__ == "__main__":
    fire.Fire()
