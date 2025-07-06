import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal

import fire
import numpy as np

from indexes.base import Index

proj_root = Path(__file__).resolve().parents[1]
acorn_path = proj_root / "3rd_party/ACORN"
acorn_binary_path = acorn_path / "build/demos"


def write_access_lists_to_json(
    access_lists: list[list[int]], output_path: str | Path
) -> None:
    flat_access_lists = list()
    for label_list in access_lists:
        flat_access_lists.extend(label_list)
        flat_access_lists.append(-1)

    with Path(output_path).open("w") as file:
        json.dump(flat_access_lists, file)


def write_vectors_to_binary(X: np.ndarray, output_path: str | Path) -> None:
    n_points, dim = X.shape

    with Path(output_path).open("wb") as file:
        for i in range(n_points):
            file.write(dim.to_bytes(4, byteorder="little", signed=False))
            file.write(X[i].astype(np.float32).tobytes())


def write_dataset(
    X: np.ndarray,
    access_lists: list[list[int]],
    dataset_dir: str | Path,
    split: Literal["train", "test"],
    overwrite: bool = False,
) -> None:
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata_fn = dataset_dir / f"{split}_metadata.json"
    vectors_fn = dataset_dir / f"{split}_vector.bin"

    if metadata_fn.is_file() and vectors_fn.is_file() and not overwrite:
        raise FileExistsError(
            f"Files {metadata_fn} and {vectors_fn} already exist. "
            "Set overwrite=True to overwrite."
        )

    print(f"Writing dataset to {dataset_dir} ...")
    write_access_lists_to_json(access_lists, metadata_fn)
    write_vectors_to_binary(X, vectors_fn)


def write_dataset_by_key(
    dataset_dir: str | Path,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    overwrite: bool = False,
):
    from benchmark.profiler import Dataset

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    write_dataset(
        dataset.train_vecs,
        dataset.train_mds,
        dataset_dir,
        "train",
        overwrite=overwrite,
    )
    write_dataset(
        dataset.test_vecs,
        dataset.test_mds,
        dataset_dir,
        "test",
        overwrite=overwrite,
    )


def load_preds_from_binary(preds_path: str | Path) -> list[list[int]]:
    with Path(preds_path).open("rb") as file:
        num_queries = int.from_bytes(file.read(8), byteorder="little", signed=False)
        k = int.from_bytes(file.read(8), byteorder="little", signed=False)
        nns = np.frombuffer(file.read(), dtype=np.int64)

    return nns.reshape(num_queries, k).tolist()


def write_complex_predicate_dataset_by_key(
    dataset_dir: str | Path,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
):
    from benchmark.complex_predicate.dataset import ComplexPredicateDataset

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset = ComplexPredicateDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        templates=templates,
        n_filters_per_template=n_filters_per_template,
        n_queries_per_filter=n_queries_per_filter,
        gt_cache_dir=gt_cache_dir,
    )

    write_vectors_to_binary(
        dataset.train_vecs,
        dataset_dir / "train_vector.bin",
    )

    write_vectors_to_binary(
        dataset.test_vecs,
        dataset_dir / "test_vector.bin",
    )

    bitmaps: dict[int, np.ndarray] = dict()
    unique_labels = set()
    for label_list in dataset.train_mds:
        unique_labels.update(label_list)

    num_vectors = dataset.train_vecs.shape[0]
    for label in unique_labels:
        bitmaps[label] = np.zeros(num_vectors, dtype=np.float32)

    for i, label_list in enumerate(dataset.train_mds):
        for label in label_list:
            bitmaps[label][i] = 1

    unique_labels_np = np.array(list(unique_labels))[:, None]
    bitmaps_np = np.array([bitmaps[label] for label in unique_labels_np.flatten()])
    bitmaps_np = np.concatenate([unique_labels_np, bitmaps_np], axis=1)
    write_vectors_to_binary(bitmaps_np, dataset_dir / "bitmap.bin")

    # bitmaps_flat: list[int] = list()
    # for label, bitmap in bitmaps.items():
    #     bitmaps_flat.append(label)
    #     bitmaps_flat.extend(bitmap.tolist())

    # with (dataset_dir / "bitmap.json").open("w") as file:
    #     json.dump(bitmaps_flat, file)

    with (dataset_dir / "filter.txt").open("w") as file:
        templates = sorted(dataset.template_to_filters.keys())
        for template in templates:
            for filter in dataset.template_to_filters[template]:
                file.write(filter + "\n")


class ACORN(Index):
    def __init__(
        self,
        dataset_dir: str | Path,
        index_dir: str | Path,
        m: int = 32,
        gamma: int = 10,
        m_beta: int = 64,
        search_ef: int = 16,
    ) -> None:
        super().__init__()

        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_dir}")

        self.index_dir = Path(index_dir)
        if self.index_dir.is_dir():
            print(f"WARN: Index directory {self.index_dir} already exists.")
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.m = m
        self.gamma = gamma
        self.m_beta = m_beta
        self.search_ef = search_ef

    @property
    def params(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "index_dir": str(self.index_dir),
            "m": self.m,
            "gamma": self.gamma,
            "m_beta": self.m_beta,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "search_ef": self.search_ef,
        }

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        if "search_ef" in params:
            self.search_ef = params["search_ef"]

    def create(self, x: np.ndarray, label: int) -> None:
        raise NotImplementedError

    def batch_create(
        self,
        X: np.ndarray,
        labels: list[int],
        access_lists: list[list[int]],
    ) -> None:
        subprocess.run(
            [
                str(acorn_binary_path / "acorn_build_index"),
                str(self.dataset_dir / "train_vector.bin"),
                str(self.index_dir / "index.bin"),
                str(self.index_dir / "memory_usage.json"),
                str(self.gamma),
                str(self.m),
                str(self.m_beta),
            ],
            check=True,
        )

    def grant_access(self, label: int, tenant_id: int) -> None:
        raise NotImplementedError

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        raise NotImplementedError

    def batch_query(
        self,
        X: np.ndarray,
        k: int,
        access_lists: list[list[int]],
        num_threads: int = 1,
    ) -> list[list[int]]:
        # Set environment variable to control OpenMP threads for FAISS
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_threads)

        subprocess.run(
            [
                str(acorn_binary_path / "acorn_search_index"),
                str(self.index_dir / "index.bin"),
                str(self.dataset_dir / "test_vector.bin"),
                str(self.dataset_dir / "test_metadata.json"),
                str(self.dataset_dir / "train_metadata.json"),
                str(self.index_dir / "preds.bin"),
                str(self.index_dir / "search_latency.bin"),
                str(self.index_dir / "ndists.bin"),
                str(self.search_ef),
            ],
            check=True,
            env=env,
        )

        return load_preds_from_binary(self.index_dir / "preds.bin")

    def batch_query_with_complex_predicate(
        self, X: np.ndarray, k: int, predicates: list[str], num_threads: int = 1
    ) -> list[list[int]]:
        # Set environment variable to control OpenMP threads for FAISS
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_threads)

        subprocess.run(
            [
                str(acorn_binary_path / "acorn_complex_search"),
                str(self.index_dir / "index.bin"),
                str(self.dataset_dir / "test_vector.bin"),
                str(self.dataset_dir / "bitmap.bin"),
                str(self.dataset_dir / "filter.txt"),
                str(self.index_dir / "preds.bin"),
                str(self.index_dir / "search_latency.bin"),
                str(self.search_ef),
            ],
            check=True,
            env=env,
        )

        return load_preds_from_binary(self.index_dir / "preds.bin")


if __name__ == "__main__":
    fire.Fire()
