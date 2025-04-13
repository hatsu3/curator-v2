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
        clus_niter: int = 20,
        max_leaf_size: int = 128,
        nprobe: int = 1200,
        prune_thres: float = 1.6,
        variance_boost: float = 0.2,
        search_ef: int = 0,
        beam_size: int = 1,
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
        self.clus_niter = clus_niter
        self.max_leaf_size = max_leaf_size
        self.nprobe = nprobe
        self.prune_thres = prune_thres
        self.variance_boost = variance_boost
        self.search_ef = search_ef
        self.beam_size = beam_size

        self.index = faiss.MultiTenantIndexIVFHierarchical(
            self.d,
            self.nlist,
            faiss.METRIC_L2,
            self.bf_capacity,
            self.bf_error_rate,
            self.max_sl_size,
            self.clus_niter,
            self.max_leaf_size,
            self.nprobe,
            self.prune_thres,
            self.variance_boost,
            self.search_ef,
            self.beam_size,
        )

    @property
    def params(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "nlist": self.nlist,
            "bf_capacity": self.bf_capacity,
            "bf_error_rate": self.bf_error_rate,
            "max_sl_size": self.max_sl_size,
            "clus_niter": self.clus_niter,
            "max_leaf_size": self.max_leaf_size,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "nprobe": self.nprobe,
            "prune_thres": self.prune_thres,
            "variance_boost": self.variance_boost,
            "search_ef": self.search_ef,
            "beam_size": self.beam_size,
        }

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        if "nprobe" in params:
            self.nprobe = params["nprobe"]
            self.index.nprobe = self.nprobe
        if "prune_thres" in params:
            self.prune_thres = params["prune_thres"]
            self.index.prune_thres = self.prune_thres
        if "variance_boost" in params:
            self.variance_boost = params["variance_boost"]
            self.index.variance_boost = self.variance_boost
        if "search_ef" in params:
            self.search_ef = params["search_ef"]
            self.index.search_ef = self.search_ef
        if "beam_size" in params:
            self.beam_size = params["beam_size"]
            self.index.beam_size = self.beam_size

    def train(
        self, X: np.ndarray, tenant_ids: Metadata | None = None, **train_params
    ) -> None:
        self.index.train(X, 0)  # type: ignore

    def create(self, x: np.ndarray, label: int) -> None:
        self.index.add_vector_with_ids(x[None], [label])  # type: ignore

    def grant_access(self, label: int, tenant_id: int) -> None:
        self.index.grant_access(label, tenant_id)  # type: ignore

    def delete_vector(self, label: int) -> None:
        self.index.remove_vector(label)  # type: ignore

    def revoke_access(self, label: int, tenant_id: int) -> None:
        self.index.revoke_access(label, tenant_id)  # type: ignore

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        top_dists, top_ids = self.index.search(x[None], k, tenant_id)  # type: ignore
        return top_ids[0].tolist()

    def query_with_complex_predicate(self, x: np.ndarray, k: int, predicate: str) -> list[int]:
        top_dists, top_ids = self.index.search(x[None], k, predicate)  # type: ignore
        return top_ids[0].tolist()

    def index_filter(self, filter: str) -> None:
        self.index.build_index_for_filter(filter)

    def query_unfiltered(self, x: np.ndarray, k: int) -> list[int]:
        top_dists, top_ids = self.index.search(x[None], k, -1)  # type: ignore
        return top_ids[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, access_lists: list[list[int]], num_threads: int = 1
    ) -> list[list[int]]:
        raise NotImplementedError("Batch querying is not supported for IVFFlatMultiTenantBFHierFaiss")

    def enable_stats_tracking(self, enable: bool = True):
        self.index.enable_stats_tracking(enable)

    def get_search_stats(self) -> dict[str, int]:
        stats = self.index.get_search_stats()
        stats = [stats.at(i) for i in range(stats.size())]
        return dict(
            zip(
                [
                    "n_dists",
                    "n_nodes_visited",
                    "n_bucks_unpruned",
                    "n_bucks_pruned",
                    "n_vecs_visited",
                    "n_steps",
                    "n_bf_queries",
                    "n_var_queries",
                ],
                stats,
            )
        )
