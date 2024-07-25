import numpy as np

from indexes.base import Index


class PreFilteringIndex(Index):
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.access_lists: list[list[int]] | None = None
        self.tenant_to_vectors: dict[int, list[int]] = dict()

        self.track_stats: bool = False
        self.n_dists: int = 0

    @property
    def params(self):
        return {}

    @property
    def search_params(self):
        return {}

    @search_params.setter
    def search_params(self, params: dict):
        pass

    def batch_create(
        self, X: np.ndarray, labels: list[int], access_lists: list[list[int]]
    ):
        self.vectors = X
        self.labels = np.array(labels)
        self.access_lists = access_lists

        for i, access_list in enumerate(access_lists):
            for tenant_id in access_list:
                if tenant_id not in self.tenant_to_vectors:
                    self.tenant_to_vectors[tenant_id] = list()
                self.tenant_to_vectors[tenant_id].append(i)

    def query(self, x: np.ndarray, k: int, tenant_id: int) -> list[int]:
        assert (
            self.vectors is not None and self.labels is not None
        ), "Index not constructed"
        qual_vec_idxs = np.array(self.tenant_to_vectors[tenant_id])
        dists = np.linalg.norm(self.vectors[qual_vec_idxs] - x, axis=1)
        if self.track_stats:
            self.n_dists = len(dists)
        sorted_idxs = qual_vec_idxs[np.argsort(dists)[:k]]
        return self.labels[sorted_idxs].tolist()

    def enable_stats_tracking(self, enable: bool) -> None:
        self.track_stats = enable

    def get_search_stats(self) -> dict:
        return {
            "n_dists": self.n_dists,
        }
