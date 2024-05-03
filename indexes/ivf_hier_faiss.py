from typing import Any

import faiss
import numpy as np

from dataset import Metadata
from indexes.base import Index


class IVFFlatMultiTenantBFHierFaiss(Index):
    """Curator index"""

    def __init__(
        self,
        d: int,
        nlist: int,
        bf_capacity: int = 1000,
        bf_error_rate: float = 0.001,
        max_sl_size: int = 128,
        update_bf_interval: int = 100,
        clus_niter: int = 20,
        max_leaf_size: int = 128,
        nprobe: int = 40,
        prune_thres: float = 1.6,
        variance_boost: float = 0.2,
    ):
        """Initialize Curator index.

        Parameters
        ----------
        d : int
            Dimensionality of the vectors.
        nlist : int
            Number of cells/buckets in the inverted file.
        bf_capacity : int, optional
            The capacity of the Bloom filter.
        bf_error_rate : float, optional
            The error rate of the Bloom filter.
        """
        super().__init__()

        self.d = d
        self.nlist = nlist
        self.bf_capacity = bf_capacity
        self.bf_error_rate = bf_error_rate
        self.max_sl_size = max_sl_size
        self.update_bf_interval = update_bf_interval
        self.clus_niter = clus_niter
        self.max_leaf_size = max_leaf_size
        self.nprobe = nprobe
        self.prune_thres = prune_thres
        self.variance_boost = variance_boost

        self.quantizer = faiss.IndexFlatL2(self.d)
        self.index = faiss.MultiTenantIndexIVFHierarchical(
            self.quantizer,
            self.d,
            self.nlist,
            faiss.METRIC_L2,
            self.bf_capacity,
            self.bf_error_rate,
            self.max_sl_size,
            self.update_bf_interval,
            self.clus_niter,
            self.max_leaf_size,
            self.nprobe,
            self.prune_thres,
            self.variance_boost,
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "nlist": self.nlist,
            "bf_capacity": self.bf_capacity,
            "bf_error_rate": self.bf_error_rate,
            "max_sl_size": self.max_sl_size,
            "update_bf_interval": self.update_bf_interval,
            "clus_niter": self.clus_niter,
            "max_leaf_size": self.max_leaf_size,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "nprobe": self.nprobe,
            "prune_thres": self.prune_thres,
            "variance_boost": self.variance_boost,
        }

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        if "nprobe" in params:
            self.nprobe = params["nprobe"]
        if "prune_thres" in params:
            self.prune_thres = params["prune_thres"]
        if "variance_boost" in params:
            self.variance_boost = params["variance_boost"]

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        self.index.train(X, 0)  # type: ignore

    def create(self, x: np.ndarray, label: int, tenant_id: int) -> None:
        self.index.add_vector_with_ids(x[None], [label], tenant_id)  # type: ignore

    def grant_access(self, label: int, tenant_id: int) -> None:
        self.index.grant_access(label, tenant_id)  # type: ignore

    def delete(self, label: int, tenant_id: int | None = None) -> None:
        raise NotImplementedError("Use delete_vector instead")

    def delete_vector(self, label: int, tenant_id: int) -> None:
        self.index.remove_vector(label, tenant_id)  # type: ignore

    def revoke_access(self, label: int, tenant_id: int) -> None:
        self.index.revoke_access(label, tenant_id)  # type: ignore

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        top_dists, top_ids = self.index.search(x[None], k, tenant_id)  # type: ignore
        return top_ids[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1
    ) -> list[list[int]]:
        top_dists, top_ids = self.index.search(X, k, tenant_id)  # type: ignore
        return top_ids.tolist()


if __name__ == "__main__":
    print("Testing IVFFlatMultiTenantBFHierFaiss...")
    index = IVFFlatMultiTenantBFHierFaiss(10, 10)
    index.train(np.random.random((400, 10)))
    index.create(np.random.rand(10), 0, 0)
    index.create(np.random.rand(10), 1, 0)
    index.create(np.random.rand(10), 2, 1)
    index.grant_access(2, 2)
    res = index.query(np.random.rand(10), 2, 0)
    print(res)
