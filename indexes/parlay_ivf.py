import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any

proj_root = Path(__file__).resolve().parents[1]
parlay_path = proj_root / "3rd_party/ParlayANN/python"
if not Path(parlay_path).is_dir():
    raise FileNotFoundError("ParlayANN not found")

sys.path.append(str(parlay_path))

import numpy as np
import wrapper as wp

from indexes.base import Index


class ParlayIVF(Index):
    def __init__(
        self,
        index_dir: str = "",
        cluster_size: int = 5000,
        cutoff: int = 10000,
        max_iter: int = 10,
        weight_classes: list[int] = [100000, 400000],
        max_degrees: list[int] = [8, 10, 12],
        bitvector_cutoff: int = 10000,
        target_points: int = 5000,
        tiny_cutoff: int = 1000,
        beam_widths: list[int] = [100, 100, 100],
        search_limits: list[int] = [100000, 400000, 3000000],
        build_threads: int = 8,
        search_threads: int = 1,
    ):
        self.index_dir = index_dir
        
        if index_dir:
            print(f"Initializing index directory at {index_dir} ...")
            Path(index_dir).mkdir(parents=True, exist_ok=False)

        self.cluster_size = cluster_size
        self.cutoff = cutoff
        self.max_iter = max_iter

        self.weight_classes = weight_classes
        self.max_degrees = max_degrees
        self.bitvector_cutoff = bitvector_cutoff

        self.target_points = target_points
        self.tiny_cutoff = tiny_cutoff
        self.beam_widths = beam_widths
        self.search_limits = search_limits

        self.build_threads = build_threads
        self.search_threads = search_threads

        self.index = None

    @property
    def params(self) -> dict[str, Any]:
        return {
            "index_dir": self.index_dir,
            "cluster_size": self.cluster_size,
            "cutoff": self.cutoff,
            "max_iter": self.max_iter,
            "weight_classes": self.weight_classes,
            "max_degrees": self.max_degrees,
            "bitvector_cutoff": self.bitvector_cutoff,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "target_points": self.target_points,
            "tiny_cutoff": self.tiny_cutoff,
            "beam_widths": self.beam_widths,
            "search_limits": self.search_limits,
        }

    @search_params.setter
    def search_params(self, value: dict[str, Any]):
        assert (
            self.index is not None
        ), "Index must be initialized before setting search params"

        if "target_points" in value:
            self.target_points = value["target_points"]
            self.index.set_target_points(self.target_points)

        if "tiny_cutoff" in value:
            self.tiny_cutoff = value["tiny_cutoff"]
            self.index.set_tiny_cutoff(self.tiny_cutoff)

        if "beam_widths" in value:
            self.beam_widths = value["beam_widths"]

        if "search_limits" in value:
            self.search_limits = value["search_limits"]

    def _set_num_threads(self, num_threads: int):
        print(f"Setting PARLAY_NUM_THREADS to {num_threads} ...")
        os.environ["PARLAY_NUM_THREADS"] = str(num_threads)

    def _set_build_params(self):
        assert (
            self.index is not None
        ), "Index must be initialized before setting build params"
        build_params = [
            wp.BuildParams(max_degree, 200, 1.175) for max_degree in self.max_degrees  # type: ignore
        ]
        for i, bp in enumerate(build_params):
            self.index.set_build_params(bp, i)

    def _set_query_params(self, k: int = 10):
        assert self.search_limits is not None
        assert (
            self.index is not None
        ), "Index must be trained before setting query params"

        for i, (beam_width, search_limit, max_degree) in enumerate(
            zip(self.beam_widths, self.search_limits, self.max_degrees)
        ):
            self.index.set_query_params(
                wp.QueryParams(k, beam_width, 1.35, search_limit, max_degree),  # type: ignore
                i,
            )

    def _write_access_lists_csr_bin(
        self, access_lists: list[list[int]], output_path: str | Path
    ):
        """
        Write access lists to a binary file in CSR format that can be parsed by ParlayANN.
        See ParlayANN/algorithms/utils/filters.h:csr_filters.
        """
        n_points = len(access_lists)
        n_nonzero = sum(len(row) for row in access_lists)

        labels = set()
        for row in access_lists:
            labels.update(row)
        n_labels = len(labels)
        assert labels == set(range(n_labels)), "Labels must be contiguous integers"

        row_offsets = [0]
        for row in access_lists:
            row_offsets.append(row_offsets[-1] + len(row))

        with Path(output_path).open("wb") as file:
            file.write(struct.pack("qqq", n_points, n_labels, n_nonzero))
            file.write(struct.pack("q" * (n_points + 1), *row_offsets))
            for row in access_lists:
                if row:
                    file.write(struct.pack("i" * len(row), *row))

    def _write_vectors_bin(self, X: np.ndarray, output_path: str | Path):
        """
        Write vectors to a binary file that can be parsed by ParlayANN.
        See ParlayANN/algorithms/utils/point_range.h:PointRange.
        """
        n_points, dim = X.shape

        with Path(output_path).open("wb") as file:
            file.write(struct.pack("II", n_points, dim))
            file.write(X.astype(np.float32).tobytes())

    def train(
        self, X: np.ndarray, tenant_ids: list[list[int]] | None = None, **train_params
    ) -> None:
        assert tenant_ids is not None, "ParlayIVF requires tenant_ids during training"

        self._set_num_threads(self.build_threads)
        self.index = wp.init_squared_ivf_index("Euclidian", "float")

        self.index.set_max_iter(self.max_iter)
        self.index.set_bitvector_cutoff(self.bitvector_cutoff)
        self._set_build_params()

        self.index.set_target_points(self.target_points)
        self.index.set_tiny_cutoff(self.tiny_cutoff)

        with tempfile.TemporaryDirectory() as dataset_dir:
            access_lists_fn = Path(dataset_dir) / "access_lists.bin"
            vectors_fn = Path(dataset_dir) / "data.bin"
            self._write_access_lists_csr_bin(tenant_ids, access_lists_fn)
            self._write_vectors_bin(X, vectors_fn)

            self.index.fit_from_filename(
                str(vectors_fn),
                str(access_lists_fn),
                self.cutoff,
                self.cluster_size,
                self.index_dir,
                self.weight_classes,
                False,
            )

    def delete(self, label: int, tenant_id: int | None = None) -> None:
        raise NotImplementedError("Deleting is not supported for ParlayIVF")

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        raise NotImplementedError("Querying is not supported for ParlayIVF")

    def batch_query(
        self,
        X: np.ndarray,
        k: int,
        access_lists: list[list[int]],
    ) -> list[list[int]]:
        assert self.index is not None, "Index must be trained before querying"

        self._set_num_threads(self.search_threads)
        self._set_query_params(k)

        filters = [wp.QueryFilter(tenant) for access_list in access_lists for tenant in access_list]  # type: ignore
        X_expanded = np.repeat(
            X, [len(access_list) for access_list in access_lists], axis=0
        )

        results, __ = self.index.batch_filter_search(
            X_expanded, filters, X_expanded.shape[0], k
        )
        self.index.reset()

        return results

    def __del__(self):
        if self.index_dir and Path(self.index_dir).is_dir():
            print(f"Deleting index directory at {self.index_dir} ...")
            shutil.rmtree(self.index_dir)


if __name__ == "__main__":
    index = ParlayIVF("parlay_ivf_test")
    index.train(
        np.random.rand(1000, 100).astype(np.float32), [[i] for i in range(1000)]
    )
    print(index.batch_query(np.random.rand(10, 100), 10, [[0]] * 10))
