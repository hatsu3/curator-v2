from typing import Any

import faiss
from numpy import ndarray

from indexes.base import Index


class HybridCurator(Index):
    def __init__(
        self,
        dim: int,
        M: int,
        gamma: int,
        M_beta: int,
        n_branches: int,
        leaf_size: int,
        n_uniq_labels: int,
        sel_threshold: float,
    ):
        super().__init__()

        self.dim = dim
        self.M = M
        self.gamma = gamma
        self.M_beta = M_beta
        self.n_branches = n_branches
        self.leaf_size = leaf_size
        self.n_uniq_labels = n_uniq_labels
        self.sel_threshold = sel_threshold

        self.index = faiss.HybridCurator(
            dim, M, gamma, M_beta, n_branches, leaf_size, n_uniq_labels
        )

    @property
    def params(self) -> dict[str, int]:
        return {
            "dim": self.dim,
            "M": self.M,
            "gamma": self.gamma,
            "M_beta": self.M_beta,
            "n_branches": self.n_branches,
            "leaf_size": self.leaf_size,
            "n_uniq_labels": self.n_uniq_labels,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "sel_threshold": self.sel_threshold,
            "curator_search_ef": self.index.curator.search_ef,
            "acorn_search_ef": self.index.acorn.acorn.efSearch,
        }

    @search_params.setter
    def search_params(self, params: dict[str, Any]) -> None:
        if "sel_threshold" in params:
            self.sel_threshold = params["sel_threshold"]
            self.index.sel_threshold = params["sel_threshold"]
        if "curator_search_ef" in params:
            self.index.curator.search_ef = params["curator_search_ef"]
        if "acorn_search_ef" in params:
            self.index.acorn.acorn.efSearch = params["acorn_search_ef"]

    def train(
        self, X: ndarray, tenant_ids: list[list[int]] | None = None, **train_params
    ) -> None:
        self.index.train(X, 0)  # type: ignore

    def create(self, x: ndarray, label: int) -> None:
        self.index.add_vector_with_ids(x[None], [label])  # type: ignore

    def grant_access(self, label: int, tenant_id: int) -> None:
        self.index.grant_access(label, tenant_id)  # type: ignore

    def delete_vector(self, label: int) -> None:
        raise NotImplementedError("HybridCurator does not support delete_vector.")

    def revoke_access(self, label: int, tenant_id: int) -> None:
        self.index.revoke_access(label, tenant_id)  # type: ignore

    def query(self, x: ndarray, k: int, tenant_id: int) -> list[int]:
        top_dists, top_ids = self.index.search(x[None], k, tenant_id)  # type: ignore
        return top_ids[0].tolist()
