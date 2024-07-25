from typing import Any

import fire
import hnswlib
import numpy as np

from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config
from dataset import Metadata
from indexes.base import Index


class HNSWIndex(Index):
    def __init__(
        self,
        tenant_id: int,
        construction_ef: int = 100,
        search_ef: int = 10,
        m: int = 16,
        num_threads: int = 1,
        max_elements: int = 2000000,
    ) -> None:
        super().__init__()

        self.tenant_id = tenant_id

        self.metric = "l2"
        self.construction_ef = construction_ef
        self.search_ef = search_ef
        self.m = m
        self.num_threads = num_threads
        self.max_elements = max_elements

        self.index: hnswlib.Index | None = None
        self.vectors: dict[int, np.ndarray] = dict()
        self.labels: set[int] = set()

    @property
    def params(self) -> dict[str, Any]:
        return {
            "construction_ef": self.construction_ef,
            "m": self.m,
            "num_threads": self.num_threads,
            "max_elements": self.max_elements,
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

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        raise NotImplementedError("hnswlib does not require training")

    def create(self, x: np.ndarray, label: int, tenant_id: int) -> None:
        self.vectors[label] = x

        if tenant_id != self.tenant_id:
            return

        if self.index is None:
            self.index = self._create_collection(x.shape[-1])
        self.index.add_items(x[None], np.asarray([label]), replace_deleted=True)
        self.labels.add(label)

    def grant_access(self, label: int, tenant_id: int) -> None:
        if tenant_id != self.tenant_id:
            return

        self.create(self.vectors[label], label, tenant_id)

    def shrink_to_fit(self) -> None:
        if self.index is None:
            raise ValueError("Index is not initialized")
        print("Resizing HNSW indexes to fit current number of elements...")
        cur_elem_num = self.index.get_current_count()
        print(f"Current number of elements: {cur_elem_num}")
        self.index.resize_index(cur_elem_num)

    def delete(self, label: int, tenant_id: int | None = None) -> None:
        raise NotImplementedError("Use delete_vector instead")

    def delete_vector(self, label: int, tenant_id: int) -> None:
        if self.index is None:
            raise ValueError("Index is not initialized")

        if label in self.labels:
            self.index.mark_deleted(label)

    def revoke_access(self, label: int, tenant_id: int) -> None:
        if self.index is None:
            raise ValueError("Index is not initialized")

        if label in self.labels:
            self.delete_vector(label, tenant_id)

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        if self.index is None:
            raise ValueError("Index is not initialized")

        if tenant_id != self.tenant_id:
            return []

        if self.index.get_current_count() < k:
            print("k is greater than number of elements, shrinking k")
            k = self.index.get_current_count()

        result_labels, __ = self.index.knn_query(x[None], k=k)
        return result_labels[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, access_lists: list[list[int]], num_threads: int = 1
    ) -> list[list[int]]:
        raise NotImplementedError("Batch querying is not supported for HNSW")

    def _create_collection(self, dim: int) -> hnswlib.Index:
        index = hnswlib.Index(space=self.metric, dim=dim)
        index.init_index(
            max_elements=self.max_elements,
            ef_construction=self.construction_ef,
            M=self.m,
            random_seed=100,
            allow_replace_deleted=True,
        )
        index.set_ef(self.search_ef)
        index.set_num_threads(self.num_threads)
        return index


def main(tenant_id):
    # load dataset
    dataset_key, test_size = "yfcc100m", 0.01
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_config = IndexConfig(
        index_cls=HNSWIndex,
        index_params={
            "tenant_id": tenant_id,
            "construction_ef": 32,
            "m": 48,
            "max_elements": 200000,
        },
        search_params={
            "search_ef": 16,
        },
        train_params=None,
    )

    profiler = IndexProfiler(multi_tenant=True)
    results = profiler.batch_profile([index_config], [dataset_config], timeout=600)
    print(results)


if __name__ == "__main__":
    fire.Fire(main)
