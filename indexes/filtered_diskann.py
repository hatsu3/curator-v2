import shutil
import struct
from pathlib import Path
from typing import Any

import diskannpy as dap
import numpy as np
import pandas as pd

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
        cache_index: bool = False,
    ):
        super().__init__()

        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=cache_index)

        self.d = d
        self.ef_construct = ef_construct
        self.graph_degree = graph_degree
        self.alpha = alpha
        self.filter_ef_construct = filter_ef_construct
        self.construct_threads = construct_threads
        self.search_threads = search_threads
        self.ef_search = ef_search
        self.cache_index = cache_index

        self.index: dap.StaticMemoryIndex | None = None
        if self.cache_index and any(self.index_dir.iterdir()):
            print(f"Loading index from {self.index_dir} ...")
            self.index = dap.StaticMemoryIndex(
                index_directory=str(self.index_dir),
                num_threads=self.search_threads,
                initial_search_complexity=self.ef_search,
                enable_filters=True,
            )

        self.track_stats: bool = False
        self.search_stats: dict[str, int] = {}

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

    def delete_vector(self, label: int, tenant_id: int) -> None:
        raise NotImplementedError("diskann does not support deletion")

    def revoke_access(self, label: int, tenant_id: int) -> None:
        raise NotImplementedError("diskann does not support access control")

    def query(self, x: np.ndarray, k: int, tenant_id: int | None = None) -> list[int]:
        if self.index is None:
            raise RuntimeError("Index has not been loaded")

        filter_label = "" if tenant_id is None else str(tenant_id)
        res = self.index.search(
            x,
            k,
            complexity=self.ef_search,
            filter_label=filter_label,
            return_stats=self.track_stats,
        )

        if self.track_stats:
            assert isinstance(res, dap.QueryResponseWithStats)
            self.search_stats = {
                "n_hops": res.hops,
                "n_ndists": res.dist_comps,
            }

        return res.identifiers.tolist()

    def batch_query(
        self, X: np.ndarray, k: int, tenant_id: int | None = None, num_threads: int = 1
    ) -> list[list[int]]:
        # diskann does not support batch queries with filtering
        return [self.query(x, k, tenant_id) for x in X]

    def enable_stats_tracking(self, enable: bool = True):
        self.track_stats = enable

    def get_search_stats(self) -> dict[str, int]:
        return self.search_stats

    def __del__(self):
        if not self.cache_index and self.index_dir.exists():
            shutil.rmtree(self.index_dir, ignore_errors=True)


class NeighborPriorityQueue:
    def __init__(self, capacity: int = 0):
        self.size = 0  # current queue size
        self.capacity = capacity  # maximum queue size
        self.cur = 0  # the first neighbor that is unexpanded
        self.data = []

    def push(self, dist: float, id: int):
        neigh = (dist, id, False)

        if self.size == self.capacity and self.data[self.size - 1] < neigh:
            return

        # binary search to find the position to insert
        lo, hi = 0, self.size
        while lo < hi:
            mid = (lo + hi) // 2
            if neigh < self.data[mid]:
                hi = mid
            elif neigh[1] == self.data[mid][1]:
                return
            else:
                lo = mid + 1

        self.data.insert(lo, neigh)
        self.data = self.data[: self.capacity + 1]

        # update the queue size
        if self.size < self.capacity:
            self.size += 1

        if lo < self.cur:
            self.cur = lo

    def closest_unexpanded(self):
        # mark the neighbor to be returned as expanded
        pre = self.cur
        self.data[pre] = (*self.data[pre][:-1], True)

        # find the next unexpanded neighbor
        while self.cur < self.size and self.data[self.cur][2]:
            self.cur += 1

        return self.data[pre][:-1]

    def has_unexpanded_nodes(self):
        return self.cur < self.size

    def topk_neighbors(self, k: int):
        neighs = self.data.copy()
        neighs.sort(key=lambda x: x[0])
        return [id for _, id, _ in neighs[:k]]


class FilteredDiskANNPy:
    def __init__(
        self,
        data: np.ndarray,
        access_lists: list[list[int]],
        label_map: dict[int, int],
        graph: list[list[int]],
        start_point: int,
        label_start_points: dict[int, int],
    ):
        self.data = data
        self.access_lists = access_lists
        self.label_map = label_map
        self.graph = graph
        self.start_point = start_point
        self.label_start_points = label_start_points

        self.search_stats = {}

    @property
    def num_edges(self):
        return sum(len(neighbors) for neighbors in self.graph)

    @classmethod
    def load(cls, index_dir: Path | str, prefix: str = "ann"):
        index_dir = Path(index_dir)
        if not index_dir.is_dir():
            raise ValueError(f"Index directory {index_dir} does not exist")

        data = cls.parse_data_file(index_dir / f"{prefix}.data")
        access_lists = cls.parse_access_list_file(index_dir / f"{prefix}_labels.txt")
        label_map = cls.parse_label_map_file(index_dir / f"{prefix}_labels_map.txt")
        graph, start_point = cls.parse_graph_store_file(index_dir / f"{prefix}")
        label_start_points = cls.parse_label_start_points_file(
            index_dir / f"{prefix}_labels_to_medoids.txt"
        )

        return cls(
            data, access_lists, label_map, graph, start_point, label_start_points
        )

    @staticmethod
    def parse_graph_store_file(filename: Path | str) -> tuple[list[list[int]], int]:
        graph: list[list[int]] = []

        with open(filename, "rb") as file:
            expected_file_size = struct.unpack("Q", file.read(8))[0]
            max_observed_degree = struct.unpack("I", file.read(4))[0]
            start = struct.unpack("I", file.read(4))[0]
            file_frozen_pts = struct.unpack("Q", file.read(8))[0]
            assert file_frozen_pts == 0, "We expect no frozen points in the graph store"

            bytes_read = 8 + 4 + 4 + 8
            nodes_read = 0
            total_edges = 0

            while bytes_read != expected_file_size:
                nodes_read += 1

                k = struct.unpack("I", file.read(4))[0]
                total_edges += k

                if k == 0:
                    raise RuntimeError("ERROR: Point found with no out-neighbours")

                neighbors = list(struct.unpack(f"{k}I", file.read(k * 4)))
                graph.append(neighbors)
                bytes_read += 4 * (k + 1)

            print(f"Read index graph with {nodes_read} nodes and {total_edges} edges")
            print(f"Max observed degree: {max_observed_degree}")
            print(f"Start point: {start}")

            return graph, start

    @staticmethod
    def parse_label_start_points_file(filename: Path | str) -> dict[int, int]:
        df = pd.read_csv(filename, header=None, names=["label", "start_point"])
        label_start_points = df.set_index("label")["start_point"].to_dict()
        print(f"Read start points of {len(label_start_points)} labels")
        return label_start_points

    @staticmethod
    def parse_data_file(filename: Path | str) -> np.ndarray:
        with open(filename, "rb") as file:
            nrows = struct.unpack("i", file.read(4))[0]
            ncols = struct.unpack("i", file.read(4))[0]
            print(f"Reading {nrows} vectors of dimension {ncols}")

            file.seek(0, 2)
            file_size = file.tell()
            file.seek(8)
            assert file_size == 8 + nrows * ncols * 4, "File size mismatch"

            data = np.frombuffer(file.read(nrows * ncols * 4), dtype=np.float32)
            data = data.reshape(nrows, ncols)
            return data

    @staticmethod
    def parse_label_map_file(filename: Path | str) -> dict[int, int]:
        df = pd.read_csv(filename, header=None, names=["ext_lbl", "int_lbl"], sep="\t")
        label_map = df.set_index("ext_lbl")["int_lbl"].to_dict()
        label_map.pop("default", None)
        label_map = {int(k): int(v) for k, v in label_map.items()}
        print(f"Read label map of {len(label_map)} labels")
        return label_map

    @staticmethod
    def parse_access_list_file(filename: Path | str) -> list[list[int]]:
        access_lists = []

        with open(filename, "r") as file:
            for line in file:
                access_list = [int(tenant) for tenant in line.strip().split(",")]
                access_lists.append(access_list)

        print(f"Read access lists of {len(access_lists)} points")
        return access_lists

    def get_converted_access_lists(
        self, access_lists: list[list[int]]
    ) -> list[list[int]]:
        return [
            [self.label_map[tenant] for tenant in access_list]
            for access_list in access_lists
        ]

    def query(
        self,
        q: np.ndarray,
        tenant: int,
        search_ef: int,
        k: int = 10,
    ) -> list[int]:
        self.search_stats = {
            "n_dists": 0,
            "hops": [],
        }

        tenant = self.label_map[tenant]
        frontier = NeighborPriorityQueue(capacity=search_ef)
        visited = set()

        # initialize the candidate pool
        init_ids = {self.start_point}
        if tenant in self.label_start_points:
            init_ids.add(self.label_start_points[tenant])

        for id in init_ids:
            if tenant not in self.access_lists[id]:
                continue

            visited.add(id)
            distance = np.linalg.norm(self.data[id] - q).item()
            self.search_stats["n_dists"] += 1
            frontier.push(distance, id)

        # traverse the graph
        while frontier.has_unexpanded_nodes():
            _, n = frontier.closest_unexpanded()
            self.search_stats["hops"].append(n)

            ids, dists = [], []

            for id in self.graph[n]:
                if tenant not in self.access_lists[id]:
                    continue

                if id not in visited:
                    ids.append(id)

            visited.update(ids)
            dists = np.linalg.norm(self.data[ids] - q, axis=1).tolist()
            self.search_stats["n_dists"] += len(ids)

            for id, dist in zip(ids, dists):
                frontier.push(dist, id)

        return frontier.topk_neighbors(k)


if __name__ == "__main__":
    import shutil

    print("Testing FilteredDiskANN...")

    n, d, dtype, nt, share_deg = 1000, 128, np.float32, 10, 2
    index_dir = Path("index")

    data = np.random.rand(n, d).astype(dtype)
    tenant_ids = np.random.randint(0, nt, (n, share_deg)).tolist()

    index = FilteredDiskANN(str(index_dir), d)
    index.batch_create(data, labels=[], tenant_ids=tenant_ids)
    index.enable_stats_tracking(True)
    res = index.query(data[0], k=10, tenant_id=tenant_ids[0][0])
    print(res)
    print(index.get_search_stats())

    shutil.rmtree(index_dir)
