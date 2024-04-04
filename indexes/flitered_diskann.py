from pathlib import Path
from typing import Any

import diskannpy as dap
import numpy as np

from dataset import Metadata
from indexes.base import Index


class FilteredDiskANN(Index):
    def __init__(
        self,
        index_dir: str,
        d: int,
        ef_construct: int = 64,
        graph_degree: int = 32,
        alpha: float = 1.0,
        filter_ef_construct: int = 128,
        construct_threads: int = 16,
        search_threads: int = 1,
        ef_search: int = 64,
    ):
        super().__init__()

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=False)

        self.d = d
        self.ef_construct = ef_construct
        self.graph_degree = graph_degree
        self.alpha = alpha
        self.filter_ef_construct = filter_ef_construct
        self.construct_threads = construct_threads
        self.search_threads = search_threads
        self.ef_search = ef_search

        self.index: dap.StaticMemoryIndex | None = None

    @property
    def params(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "ef_construct": self.ef_construct,
            "graph_degree": self.graph_degree,
            "alpha": self.alpha,
            "filter_ef_construct": self.filter_ef_construct,
            "construct_threads": self.construct_threads,
            "search_threads": self.search_threads,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "ef_search": self.ef_search,
        }

    @search_params.setter
    def search_params(self, value: dict[str, Any]):
        if "ef_search" in value:
            self.ef_search = value["ef_search"]

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        raise NotImplementedError("diskann does not require training")

    def create(self, x: np.ndarray, label: int, tenant_id: int) -> None:
        raise NotImplementedError("diskann does not support creation")

    def batch_create(
        self, X: np.ndarray, labels: list[int], tenant_ids: list[list[int]]
    ) -> None:
        if self.index is not None:
            raise RuntimeError("Index has already been loaded")

        if len(labels) > 0:
            raise NotImplementedError("diskann does not support user-specific labels")

        tenant_ids_str = [[str(t) for t in ts] for ts in tenant_ids]
        dap.build_memory_index(
            data=X,
            distance_metric="l2",
            index_directory=str(self.index_dir),
            complexity=self.ef_construct,
            graph_degree=self.graph_degree,
            num_threads=self.construct_threads,
            alpha=self.alpha,
            use_pq_build=False,
            filter_labels=tenant_ids_str,
            universal_label="",
            filter_complexity=self.filter_ef_construct,
        )  # type: ignore

        self.index = dap.StaticMemoryIndex(
            index_directory=str(self.index_dir),
            num_threads=self.search_threads,
            initial_search_complexity=self.ef_search,
            enable_filters=True,
        )

    def grant_access(self, label: int, tenant_id: int) -> None:
        raise NotImplementedError("diskann does not support access control")

    def delete(self, label: int, tenant_id: int | None = None) -> None:
        raise NotImplementedError("diskann does not support deletion")

    def delete_vector(self, label: int, tenant_id: int) -> None:
        raise NotImplementedError("diskann does not support deletion")

    def revoke_access(self, label: int, tenant_id: int) -> None:
        raise NotImplementedError("diskann does not support access control")

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        if self.index is None:
            raise RuntimeError("Index has not been loaded")

        filter_label = "" if tenant_id is None else str(tenant_id)
        res = self.index.search(
            x, k, complexity=self.ef_search, filter_label=filter_label
        )
        return res.identifiers.tolist()

    def batch_query(
        self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1
    ) -> list[list[int]]:
        # diskann does not support batch queries with filtering
        return [self.query(x, k, tenant_id) for x in X]


if __name__ == "__main__":
    import shutil

    print("Testing FilteredDiskANN...")

    n, d, dtype, nt, share_deg = 1000, 128, np.float32, 10, 2
    index_dir = Path("index")

    data = np.random.rand(n, d).astype(dtype)
    tenant_ids = np.random.randint(0, nt, (n, share_deg)).tolist()

    index = FilteredDiskANN(str(index_dir), d)
    index.batch_create(data, labels=[], tenant_ids=tenant_ids)
    res = index.query(data[0], k=10, tenant_id=tenant_ids[0][0])
    print(res)

    shutil.rmtree(index_dir)
