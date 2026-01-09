from typing import Any

import faiss
import numpy as np

from benchmark.complex_predicate.utils import compute_qualified_labels
from dataset import Metadata
from indexes.base import Index


class Curator(Index):
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
        use_temp_index_caching: bool = False,
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
        use_temp_index_caching : bool, optional
            If True, build_index_for_filter will cache temporary index structures
            instead of using direct tenant-based indexing. This provides a more
            unified approach similar to search_with_bitmap_filter.
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
        self.use_temp_index_caching = use_temp_index_caching

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
            self.use_temp_index_caching,
        )

        # Map indexed filters to their assigned labels
        self.filter_to_label = dict()

        # Map indexed filters to their precomputed bitmaps
        self.filter_to_bitmap = dict()

        # Map predicate filters to their sorted internal vector IDs for optimized search
        self.filter_to_sorted_vids = dict()

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

    """ Complex predicate support """

    def query_with_complex_predicate(
        self, x: np.ndarray, k: int, predicate: str
    ) -> list[int]:
        # Priority order: tenant-based index > optimized bitmap > regular bitmap
        if predicate in self.filter_to_label:  # index already built
            return self.query(x, k, self.filter_to_label[predicate])
        elif (
            self.index.get_optimized_search_enabled()
            and predicate in self.filter_to_sorted_vids
        ):  # optimized search available
            top_dists, top_ids = self.index.search_with_bitmap_filter_optimized(
                x[None], k, self.filter_to_sorted_vids[predicate]
            )  # type: ignore
            return top_ids[0].tolist()
        elif predicate in self.filter_to_bitmap:  # bitmap already built
            return self.search_with_bitmap_filter(
                x, k, self.filter_to_bitmap[predicate]
            )
        else:
            raise ValueError(
                f"No index, optimized bitmap, or regular bitmap found for predicate {predicate}"
            )

    def index_filter(self, predicate: str, train_mds: list[list[int]]) -> None:
        """Build index for a given predicate."""
        qualified_labels = compute_qualified_labels(predicate, train_mds)
        self.index.build_index_for_filter(qualified_labels, predicate)  # type: ignore

        # Both approaches now store the tenant label mapping for tenant-based search
        self.filter_to_label[predicate] = self.index.get_filter_label(predicate)

    def build_filter_bitmap(self, predicate: str, train_mds: list[list[int]]) -> None:
        """Build bitmap for a given predicate."""
        qualified_labels = compute_qualified_labels(predicate, train_mds)
        self.filter_to_bitmap[predicate] = qualified_labels

    def build_filter_sorted_vids(
        self, predicate: str, train_mds: list[list[int]]
    ) -> None:
        """Build sorted internal vector IDs for optimized search for a given predicate."""
        qualified_labels = compute_qualified_labels(predicate, train_mds).astype(np.uint32)
        # Convert external labels to internal vector IDs
        internal_vids = self.index.get_label_to_vid_mapping(qualified_labels)
        # Filter out any -1 values (labels not found) and sort
        valid_vids = internal_vids[internal_vids != -1]
        sorted_vids = np.sort(valid_vids)
        self.filter_to_sorted_vids[predicate] = sorted_vids

    def search_with_bitmap_filter(
        self,
        x: np.ndarray,
        k: int,
        qualified_labels: np.ndarray | list[int],
    ) -> list[int]:
        """Search using bitmap filter with explicit list of qualified labels.

        Parameters
        ----------
        x : np.ndarray
            Query vector
        k : int
            Number of nearest neighbors to return
        qualified_labels : np.ndarray
            List of vector labels that are allowed in search results

        Returns
        -------
        list[int]
            List of nearest neighbor labels
        """
        qualified_labels = np.array(qualified_labels, dtype=np.uint32)
        top_dists, top_ids = self.index.search_with_bitmap_filter(
            x[None], k, qualified_labels
        )  # type: ignore
        return top_ids[0].tolist()

    def get_last_search_profile(self) -> dict[str, Any]:
        """Get profiling data from the last search_with_bitmap_filter call.

        Returns
        -------
        dict[str, Any]
            Dictionary containing timing information and statistics:
            - preproc_time_ms: Time spent preprocessing (converting labels to internal IDs)
            - sort_time_ms: Time spent sorting qualified vector IDs
            - build_temp_index_time_ms: Time spent building temporary index structure
            - search_time_ms: Time spent searching the temporary index
            - qualified_labels_count: Number of input qualified labels
            - temp_nodes_count: Number of nodes in the temporary index
        """
        return {
            "preproc_time_ms": self.index.get_last_preproc_time_ms(),  # type: ignore
            "sort_time_ms": self.index.get_last_sort_time_ms(),  # type: ignore
            "build_temp_index_time_ms": self.index.get_last_build_temp_index_time_ms(),  # type: ignore
            "search_time_ms": self.index.get_last_search_time_ms(),  # type: ignore
            "qualified_labels_count": self.index.get_last_qualified_labels_count(),  # type: ignore
            "temp_nodes_count": self.index.get_last_temp_nodes_count(),  # type: ignore
        }

    def query_unfiltered(self, x: np.ndarray, k: int) -> list[int]:
        top_dists, top_ids = self.index.search(x[None], k, -1)  # type: ignore
        return top_ids[0].tolist()

    def batch_query(
        self, X: np.ndarray, k: int, access_lists: list[list[int]], num_threads: int = 1
    ) -> list[list[int]]:
        raise NotImplementedError(
            "Batch querying is not supported for Curator"
        )

    def enable_stats_tracking(self, enable: bool = True) -> None:
        self.index.set_profiling_enabled(enable)

    def get_search_stats(self) -> dict[str, Any]:
        """Get search statistics. For compatibility, returns last search profile data."""
        return self.get_last_search_profile()

    def get_cached_temp_index_memory_usage(self) -> int:
        """Get total memory usage of all cached temporary indexes in bytes.

        Only relevant when use_temp_index_caching=True.

        Returns
        -------
        int
            Total memory usage in bytes
        """
        return self.index.get_cached_temp_index_memory_usage()  # type: ignore
