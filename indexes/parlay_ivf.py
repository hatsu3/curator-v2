import os
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

import parse

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
        dataset_dir: str | Path,
        index_dir: str | Path = "",
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
        self.dataset_dir = Path(dataset_dir)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

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
            "dataset_dir": self.dataset_dir,
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
            "search_threads": self.search_threads,
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

        if "search_threads" in value:
            self.search_threads = value["search_threads"]

        self._set_num_threads(self.search_threads)
        self._set_query_params(k=10)

    @property
    def dataset_exists(self) -> bool:
        return (self.dataset_dir / "access_lists.bin").is_file() and (
            self.dataset_dir / "data.bin"
        ).is_file()

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

    def write_dataset(
        self, X: np.ndarray, access_lists: list[list[int]], overwrite: bool = False
    ):
        assert self.dataset_dir.is_dir(), f"Directory {self.dataset_dir} does not exist"
        access_lists_fn = self.dataset_dir / "access_lists.bin"
        vectors_fn = self.dataset_dir / "data.bin"

        if access_lists_fn.is_file() and vectors_fn.is_file() and not overwrite:
            raise FileExistsError(
                f"Files {access_lists_fn} and {vectors_fn} already exist. "
                "Set overwrite=True to overwrite."
            )

        print(f"Writing dataset to {self.dataset_dir} ...")
        self._write_access_lists_csr_bin(access_lists, access_lists_fn)
        self._write_vectors_bin(X, vectors_fn)

    def batch_create(
        self,
        X: np.ndarray | None = None,
        labels: list[int] | None = None,
        access_lists: list[list[int]] | None = None,
    ) -> None:
        if X is not None or access_lists is not None:
            raise ValueError(
                "ParlayIVF loads data from files directly during batch insertion"
            )

        data_path = self.dataset_dir / "data.bin"
        access_lists_path = self.dataset_dir / "access_lists.bin"

        if not data_path.is_file() or not access_lists_path.is_file():
            raise FileNotFoundError(
                f"Data files not found at {data_path} and {access_lists_path}. "
                f"Use write_dataset to write the dataset to disk first."
            )

        self._set_num_threads(self.build_threads)
        self.index = wp.init_squared_ivf_index("Euclidian", "float")

        self.index.set_max_iter(self.max_iter)
        self.index.set_bitvector_cutoff(self.bitvector_cutoff)
        self._set_build_params()

        self.index.set_target_points(self.target_points)
        self.index.set_tiny_cutoff(self.tiny_cutoff)

        self.index.fit_from_filename(
            str(data_path),
            str(access_lists_path),
            self.cutoff,
            self.cluster_size,
            self.index_dir,
            self.weight_classes,
            False,
        )

        self._set_num_threads(self.search_threads)
        self._set_query_params(k=10)

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        assert self.index is not None, "Index must be trained before querying"

        if k != 10:
            self._set_num_threads(self.search_threads)
            self._set_query_params(k)

        results, __ = self.index.batch_filter_search(
            x[None], [wp.QueryFilter(tenant_id)], 1, k  # type: ignore
        )
        self.index.reset()

        return results[0]

    def batch_query(
        self,
        X: np.ndarray,
        k: int,
        access_lists: list[list[int]],
        num_threads: int = 1,
    ) -> list[list[int]]:
        assert self.index is not None, "Index must be trained before querying"

        if k != 10 or num_threads != self.search_threads:
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

    def query_with_complex_predicate(
        self, x: np.ndarray, k: int, predicate: str
    ) -> list[int]:
        assert self.index is not None, "Index must be trained before querying"

        self._set_num_threads(self.search_threads)
        self._set_query_params(k)

        res = parse.parse("AND {} {}", predicate)
        if res is None:
            raise ValueError(f"Invalid predicate: {predicate}. Only AND is supported.")
        assert isinstance(res, parse.Result)
        t1, t2 = res.fixed

        filter = wp.QueryFilter(int(t1), int(t2))  # type: ignore
        results, __ = self.index.batch_filter_search(x[None], [filter], 1, k)
        self.index.reset()

        return results[0]

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
