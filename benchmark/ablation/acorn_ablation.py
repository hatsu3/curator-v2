import pickle as pkl
import subprocess
from itertools import product
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from benchmark.profiler import Dataset
from indexes.acorn import ACORN, acorn_binary_path


def build_acorn_index(
    dataset_dir: str,
    index_dir: str,
    m: int = 32,
    gamma: int = 20,
    m_beta: int = 128,
):
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    index = ACORN(
        dataset_dir=dataset_dir,
        index_dir=index_dir,
        m=m,
        gamma=gamma,
        m_beta=m_beta,
    )
    print(f"Building index with m = {m}, gamma = {gamma}, m_beta = {m_beta} ...")
    index.batch_create(np.array([]), [], [])
    print(f"Index saved to {index.index_dir} ...")


class ACORNGraph:
    def __init__(
        self,
        levels: np.ndarray,
        neighbors: np.ndarray,
        offsets: np.ndarray,
        nneighbors: np.ndarray,
        index_dir: str | Path,
        graphs: list[nx.DiGraph] = [],
    ):
        """
        Args:
            levels: np.ndarray, shape (n_nodes,)
                Maximum level of each node, starting from 0 (base level)
            neighbors: np.ndarray, shape (n_edges,)
                Neighbors of each node, across all levels it appears.
            offsets: np.ndarray, shape (n_nodes + 1,)
                Offset of each node in the neighbors array
            nneighbors: np.ndarray, shape (n_levels + 1,)
                Cumulative number of neighbors for each level.
        """

        self.levels = levels - 1
        self.neighbors = neighbors
        self.offsets = offsets
        self.nneighbors = nneighbors
        self.index_dir = index_dir

        # Sanity checks
        self.n_nodes = len(levels)
        assert len(offsets) == self.n_nodes + 1
        self.n_levels = max(levels) + 1

        for node_id in range(self.n_nodes):
            nneigh = self.nneighbors[self.levels[node_id] + 1]
            assert self.offsets[node_id] + nneigh == self.offsets[node_id + 1]

        assert self.offsets[-1] == len(self.neighbors)

        # Build graphs (HNSW graphs are directed)
        if graphs:
            self.graphs = graphs
        else:
            self.graphs = [nx.DiGraph() for _ in range(self.n_levels)]
            for node_id, node_level in enumerate(self.levels):
                self.graphs[node_level].add_node(node_id)

            for node_id, node_level in tqdm(
                enumerate(self.levels), total=self.n_nodes, desc="Building graphs"
            ):
                for level in range(node_level):
                    neigh_beg = int(self.offsets[node_id] + self.nneighbors[level])
                    neigh_end = int(self.offsets[node_id] + self.nneighbors[level + 1])
                    for neigh in self.neighbors[neigh_beg:neigh_end]:
                        self.graphs[level].add_edge(node_id, neigh)

            self.save_to_gpickle(Path(self.index_dir) / "graphs")

    @classmethod
    def from_index_dir(cls, index_dir: str | Path):
        print(f"Loading index from {index_dir} ...")
        index_dir = Path(index_dir)
        tmp_index_path = Path(index_dir) / "index.bin.tmp"

        if not tmp_index_path.exists():
            subprocess.run(
                [
                    str(acorn_binary_path / "acorn_read_index"),
                    str(index_dir / "index.bin"),
                    str(tmp_index_path),
                ],
                check=True,
            )

        with open(tmp_index_path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            levels = np.fromfile(f, dtype=np.int32, count=n)
            n = int.from_bytes(f.read(8), "little")
            neighbors = np.fromfile(f, dtype=np.int32, count=n)
            n = int.from_bytes(f.read(8), "little")
            offsets = np.fromfile(f, dtype=np.uint64, count=n)
            n = int.from_bytes(f.read(8), "little")
            nneighbors = np.fromfile(f, dtype=np.int32, count=n)

        graph_dir = index_dir / "graphs"
        if graph_dir.is_dir():
            print(f"Loading graphs from {graph_dir} ...")
            graphs = [
                pkl.load(open(graph_dir / f"level_{level}.gpickle", "rb"))
                for level in range(max(levels))
            ]
            return cls(levels, neighbors, offsets, nneighbors, index_dir, graphs)

        return cls(levels, neighbors, offsets, nneighbors, index_dir)

    def save_to_gpickle(self, output_dir: str | Path):
        print(f"Saving graphs to {output_dir} ...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for level, graph in enumerate(self.graphs):
            pkl.dump(graph, open(output_dir / f"level_{level}.gpickle", "wb"))


def load_acorn_index(index_dir: str):
    return ACORNGraph.from_index_dir(index_dir)


def plot_neighborhood(
    index_dir: str,
    query_idx: int,
    label: int,
    output_path: str,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    n_neighbors: int = 100,
):
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    index = load_acorn_index(index_dir)
    assert len(dataset.train_vecs) == index.n_nodes

    print(f"Plotting neighborhood of query {query_idx} ...")
    print(f"Label list of query: {dataset.test_mds[query_idx]}")

    # find neighbors of query with brute force search
    query = dataset.test_vecs[query_idx]
    dists = np.linalg.norm(dataset.train_vecs - query, axis=1)
    neighbors = np.argsort(dists)[:n_neighbors]
    neighbors_filtered = [n for n in neighbors if label in dataset.train_mds[n]]
    entry_point = neighbors_filtered[0]

    # project vectors to 2D with PCA
    pca = PCA(n_components=2).fit(dataset.train_vecs)
    neighbors_projected = pca.transform(dataset.train_vecs[neighbors])
    neighbors_filtered_projected = pca.transform(dataset.train_vecs[neighbors_filtered])
    query_projected = pca.transform(query.reshape(1, -1))

    # plot nodes
    plt.rcParams.update({"font.size": 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    for ax in (ax1, ax2):
        ax.scatter(
            neighbors_projected[:, 0],
            neighbors_projected[:, 1],
            label="neighbors",
            c="#d3d3d3",  # light gray
            s=5,
        )
        ax.scatter(
            neighbors_filtered_projected[:, 0],
            neighbors_filtered_projected[:, 1],
            label="neighbors_filtered",
            c="#1f77b4",  # light blue
            s=5,
        )
        ax.scatter(
            query_projected[:, 0], query_projected[:, 1], label="query", c="red", s=5
        )

    # plot edges
    for ax in (ax1, ax2):
        for src, dst in product(neighbors, neighbors):
            if index.graphs[0].has_edge(src, dst):
                src_projected = pca.transform(dataset.train_vecs[src].reshape(1, -1))
                dst_projected = pca.transform(dataset.train_vecs[dst].reshape(1, -1))
                ax.plot(
                    [src_projected[0, 0], dst_projected[0, 0]],
                    [src_projected[0, 1], dst_projected[0, 1]],
                    c="gray",
                    alpha=(0.2 if ax == ax1 else 0.1),
                    linewidth=0.5,
                )

    # sssp from entry point to neighbors_filtered
    for neighbor in neighbors_filtered:
        try:
            path = nx.shortest_path(index.graphs[0], entry_point, neighbor)
        except nx.NetworkXNoPath:
            print(f"No path from {entry_point} to {neighbor}")
            continue

        for src, dst in zip(path[:-1], path[1:]):
            src_projected = pca.transform(dataset.train_vecs[src].reshape(1, -1))
            dst_projected = pca.transform(dataset.train_vecs[dst].reshape(1, -1))

            if src not in neighbors:
                ax2.scatter(
                    src_projected[0, 0],
                    src_projected[0, 1],
                    label="neighbors",
                    c="#d3d3d3",  # light gray
                    s=5,
                )

            ax2.plot(
                [src_projected[0, 0], dst_projected[0, 0]],
                [src_projected[0, 1], dst_projected[0, 1]],
                c="red",
                alpha=0.5,
                linewidth=0.5,
            )

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    fire.Fire()
