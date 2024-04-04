from typing import Any

import faiss
import numpy as np

from dataset import Metadata
from indexes.base import Index


class HybridCurator(Index):
    def __init__(
        self,
        d: int,
        M: int,
        branch_factor: int = 4,
        buf_capacity: int = 32,
        alpha: float = 1.0,
        ef_construction: int = 40,
        tree_depth: int = 8,
        bf_capacity: int = 1000,
        bf_error_rate: float = 0.001,
    ):
        super().__init__()

        self.d = d
        self.M = M
        self.branch_factor = branch_factor
        self.buf_capacity = buf_capacity
        self.tree_depth = tree_depth
        self.alpha = alpha
        self.ef_construction = ef_construction
        self.bf_capacity = bf_capacity
        self.bf_error_rate = bf_error_rate

        self.index = faiss.HybridCuratorV2(
            self.d,
            self.M,
            self.tree_depth,
            self.branch_factor,
            self.alpha,
            self.ef_construction,
            self.bf_capacity,
            self.bf_error_rate,
            self.buf_capacity,
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "M": self.M,
            "tree_depth": self.tree_depth,
            "branch_factor": self.branch_factor,
            "ef_construction": self.ef_construction,
            "bf_capacity": self.bf_capacity,
            "bf_error_rate": self.bf_error_rate,
            "buf_capacity": self.buf_capacity,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {"alpha": self.alpha}

    @search_params.setter
    def search_params(self, value: dict[str, Any]):
        if "alpha" in value:
            self.alpha = value["alpha"]

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        self.index.train(X, 0)  # type: ignore

    def create(self, x: np.ndarray, label: int, tenant_id: int) -> None:
        self.index.add_vector_with_ids(x[None], [label], tenant_id)  # type: ignore

    def grant_access(self, label: int, tenant_id: int) -> None:
        self.index.grant_access(label, tenant_id)  # type: ignore

    def delete(self, label: int, tenant_id: int | None = None) -> None:
        raise NotImplementedError("use delete_vector instead")

    def delete_vector(self, label: int, tenant_id: int) -> None:
        pass

    def revoke_access(self, label: int, tenant_id: int) -> None:
        pass

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        top_dists, top_ids = self.index.search(x[None], k, tenant_id)  # type: ignore
        return top_ids[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1
    ) -> list[list[int]]:
        top_dists, top_ids = self.index.search(X, k, tenant_id)  # type: ignore
        return top_ids.tolist()


if __name__ == "__main__":
    print("Testing HybridCurator...")

    d, M, tree_depth, branch_factor = 512, 16, 4, 4
    index = HybridCurator(d, M, branch_factor, buf_capacity=4, tree_depth=tree_depth)

    index.train(np.random.rand(1000, d))

    for i in range(1000):
        index.create(np.random.rand(d), i, i % 100)
        index.grant_access(i, (i + 50) % 100)

    res = index.query(np.random.rand(d), 10, 0)
    print(res)
