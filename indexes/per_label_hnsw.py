from typing import Any

import hnswlib
import numpy as np

from dataset import Metadata
from indexes.base import Index


class PerLabelHNSW(Index):
    """HNSW index with per-tenant indexing"""

    def __init__(
        self,
        construction_ef: int = 100,
        search_ef: int = 10,
        m: int = 16,
        num_threads: int = 1,
        max_elements: int = 2000000,
    ) -> None:
        super().__init__()

        self.metric = "l2"
        self.construction_ef = construction_ef
        self.search_ef = search_ef
        self.m = m
        self.num_threads = num_threads
        self.max_elements = max_elements

        self.vectors: dict[int, np.ndarray] = dict()
        self.indexes: dict[int, hnswlib.Index] = dict()

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

    @property
    def dim(self) -> int:
        return next(iter(self.vectors.values())).shape[-1]

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        raise NotImplementedError("hnswlib does not require training")

    def create(self, x: np.ndarray, label: int) -> None:
        self.vectors[label] = x

    def grant_access(self, label: int, tenant_id: int) -> None:
        if tenant_id not in self.indexes:
            self.indexes[tenant_id] = self._create_collection(self.dim)

        vec = self.vectors[label]
        self.indexes[tenant_id].add_items(
            vec[None], np.asarray([label]), replace_deleted=True
        )

    def shrink_to_fit(self) -> None:
        print("Resizing HNSW indexes to fit current number of elements...")
        for index in self.indexes.values():
            cur_elem_num = index.get_current_count()
            index.resize_index(cur_elem_num)

    def delete_vector(self, label: int) -> None:
        self.vectors.pop(label)

    def revoke_access(self, label: int, tenant_id: int) -> None:
        self.indexes[tenant_id].mark_deleted(label)

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        assert tenant_id is not None
        index = self.indexes[tenant_id]

        if index.get_current_count() < k:
            print("k is greater than number of elements, shrinking k")
            k = index.get_current_count()

        index.set_ef(self.search_ef)
        result_labels, __ = index.knn_query(x[None], k=k)
        return result_labels[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, access_lists: list[list[int]], num_threads: int = 1
    ) -> list[list[int]]:
        raise NotImplementedError(
            "Batch querying is not supported for PerLabelHNSW"
        )

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
