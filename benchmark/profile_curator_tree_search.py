import hashlib
import heapq
import pickle as pkl
from collections import Counter
from pathlib import Path

import fire
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

from benchmark.config import DatasetConfig
from benchmark.utils import get_dataset_config
from dataset import get_dataset, get_metadata


def load_dataset(dataset_config: DatasetConfig):
    train_vecs, test_vecs, metadata = get_dataset(
        dataset_name=dataset_config.dataset_name, **dataset_config.dataset_params
    )

    train_mds, test_mds = get_metadata(
        synthesized=dataset_config.synthesize_metadata,
        train_vecs=train_vecs,
        test_vecs=test_vecs,
        dataset_name=dataset_config.dataset_name,
        **dataset_config.metadata_params,
    )

    return train_vecs, test_vecs, train_mds, test_mds, metadata


class KMeansTree:
    class TreeNode:
        def __init__(self, parent=None):
            self.parent = parent
            self.sibling_id = -1
            self.kmeans: KMeans | None = None
            self.children = []
            self.vectors: list[int] = []

        @property
        def is_leaf(self):
            return len(self.children) == 0

        def predict(self, x: np.ndarray) -> int:
            assert self.kmeans is not None, "predict called on a leaf node"
            assign = self.kmeans.predict(x[None]).item()
            return assign

        def get_path(self) -> list[int]:
            path = []
            node = self
            while node.parent is not None:
                path.append(node.sibling_id)
                node = node.parent
            return path[::-1]

        def centroid(self) -> np.ndarray:
            assert self.parent is not None, "Root node has no centroid"
            return self.parent.kmeans.cluster_centers_[self.sibling_id]

        def __lt__(self, other):
            return hash(self) < hash(other)

    def __init__(
        self,
        n_clusters: int = 16,
        max_leaf_size: int = 128,
        n_init: int = 1,
        max_iter: int = 20,
    ):
        self.n_clusters = n_clusters
        self.max_leaf_size = max_leaf_size
        self.n_init = n_init
        self.max_iter = max_iter

        self.vec_to_leaf = {}
        self.root = None
        self.vectors: np.ndarray | None = None

        self.query_stat = {
            "n_dists": 0,
            "visited_nodes": [],
        }

    def fit(self, X: np.ndarray, labels: np.ndarray | None = None):
        if labels is None:
            labels = np.arange(len(X))
        self.vectors = X
        self.root = self._build_tree(X, labels)

    @property
    def average_depth(self):
        all_leafs = set(self.vec_to_leaf.values())
        return np.mean([len(leaf.get_path()) for leaf in all_leafs]).item()

    def _build_tree(
        self, X: np.ndarray, labels: np.ndarray, parent: TreeNode | None = None
    ) -> TreeNode:
        node = self.TreeNode(parent)

        if len(X) <= self.max_leaf_size:
            node.vectors = labels.tolist()
            self.vec_to_leaf.update({label: node for label in labels})
        else:
            node.kmeans = KMeans(
                n_clusters=self.n_clusters,
                init="random",
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=42,
            )
            assign = node.kmeans.fit_predict(X)

            for i in range(self.n_clusters):
                child = self._build_tree(X[assign == i], labels[assign == i], node)
                child.sibling_id = i
                node.children.append(child)

        return node

    def closest_leaf(self, x: np.ndarray) -> TreeNode:
        assert self.root is not None, "Tree not trained"
        node = self.root
        while not node.is_leaf:
            assign = node.predict(x)
            node = node.children[assign]
        return node

    def query(self, x: np.ndarray, k: int = 10, nprobe: int = 3000) -> list[int]:
        assert self.root is not None, "Tree not trained"
        assert self.vectors is not None, "Tree not trained"

        results = []
        frontier = [(0.0, self.root)]
        num_cand_vecs = 0

        self.query_stat = {
            "n_dists": 0,
            "visited_nodes": [],
        }

        while len(frontier) and num_cand_vecs < nprobe:
            dist, node = heapq.heappop(frontier)
            self.query_stat["visited_nodes"].append(node.get_path())

            if node.is_leaf:
                num_cand_vecs += len(node.vectors)
                self.query_stat["n_dists"] += len(node.vectors)

                for vec in node.vectors:
                    dist = np.linalg.norm(x - self.vectors[vec]).item()
                    if len(results) < k:
                        heapq.heappush(results, (-dist, vec))
                    else:
                        heapq.heappushpop(results, (-dist, vec))
            else:
                for child in node.children:
                    assert isinstance(child, self.TreeNode)
                    dist = np.linalg.norm(x - child.centroid()).item()
                    self.query_stat["n_dists"] += 1
                    heapq.heappush(frontier, (dist, child))

        return [vec for _, vec in sorted(results, reverse=True)]

    def query_beam_nobacktrack(
        self, x: np.ndarray, k: int = 10, beam_size: int = 4, nprobe: int = 3000
    ):
        assert self.root is not None, "Tree not trained"
        assert self.vectors is not None, "Tree not trained"

        results = []
        beam = [(0.0, self.root)]

        self.query_stat = {
            "n_dists": 0,
            "visited_nodes": [],
        }

        while True:
            expanded_beam = []
            expanded = False

            for dist, node in beam:
                if node.is_leaf:
                    self.query_stat["n_dists"] += len(node.vectors)

                    for vec in node.vectors:
                        dist = np.linalg.norm(x - self.vectors[vec]).item()
                        if len(results) < k:
                            heapq.heappush(results, (-dist, vec))
                        else:
                            heapq.heappushpop(results, (-dist, vec))
                else:
                    expanded = True
                    for child in node.children:
                        assert isinstance(child, self.TreeNode)
                        dist = np.linalg.norm(x - child.centroid()).item()
                        self.query_stat["n_dists"] += 1
                        expanded_beam.append((dist, child))

            if not expanded:
                break

            expanded_beam.sort(key=lambda x: x[0])
            beam = expanded_beam[:beam_size]

        return [vec for _, vec in sorted(results, reverse=True)]

    def query_beam(
        self, x: np.ndarray, k: int = 10, beam_size: int = 4, nprobe: int = 3000
    ):
        assert self.root is not None, "Tree not trained"
        assert self.vectors is not None, "Tree not trained"

        frontier = [(0.0, self.root)]
        beam = [(0.0, self.root)]

        self.query_stat = {
            "n_dists": 0,
            "visited_nodes": [],
        }

        while True:
            expanded_beam = []
            expanded = False

            for dist, node in beam:
                if node.is_leaf:
                    expanded_beam.append((dist, node))
                else:
                    expanded = True
                    for child in node.children:
                        assert isinstance(child, self.TreeNode)
                        dist = np.linalg.norm(x - child.centroid()).item()
                        self.query_stat["n_dists"] += 1
                        expanded_beam.append((dist, child))
                        heapq.heappush(frontier, (dist, child))

            if not expanded:
                break

            expanded_beam.sort(key=lambda x: x[0])
            beam = expanded_beam[:beam_size]

        visited = {node for _, node in frontier}
        results = []
        num_cand_vecs = 0

        while frontier and num_cand_vecs < nprobe:
            dist, node = heapq.heappop(frontier)
            self.query_stat["visited_nodes"].append(node.get_path())

            if node.is_leaf:
                num_cand_vecs += len(node.vectors)
                self.query_stat["n_dists"] += len(node.vectors)

                for vec in node.vectors:
                    dist = np.linalg.norm(x - self.vectors[vec]).item()
                    if len(results) < k:
                        heapq.heappush(results, (-dist, vec))
                    else:
                        heapq.heappushpop(results, (-dist, vec))
            else:
                for child in node.children:
                    if child in visited:
                        continue

                    assert isinstance(child, self.TreeNode)
                    dist = np.linalg.norm(x - child.centroid()).item()
                    self.query_stat["n_dists"] += 1
                    heapq.heappush(frontier, (dist, child))
                    visited.add(child)

        return [vec for _, vec in sorted(results, reverse=True)]

    def sanity_check(self):
        assert self.root is not None, "Tree not trained"
        assert self.vectors is not None, "Tree not trained"

        all_vecs = list()
        for vec, leaf in self.vec_to_leaf.items():
            all_vecs.append(vec)
        assert len(all_vecs) == len(self.vectors)
        assert set(all_vecs) == set(range(len(self.vectors)))


def recall_at_k(preds: list[int], ground_truth: list[int], k: int) -> float:
    return len(set(preds[:k]).intersection(set(ground_truth))) / k


def brute_force_knn(X: np.ndarray, q: np.ndarray, k: int) -> list[int]:
    distances = np.linalg.norm(X - q, axis=1)
    return np.argsort(distances)[:k].tolist()


def brute_force_knn_cuda(
    X: np.ndarray, Q: np.ndarray, k: int, batch_size: int = 8
) -> list[list[int]]:
    cache_key = hashlib.md5(X.tobytes() + Q.tobytes()).hexdigest()
    cache_path = Path(f"unfiltered_gt_{cache_key}.npy")

    if cache_path.exists():
        print(f"Loading cached ground truth from {cache_path}")
        return np.load(cache_path).tolist()

    import torch

    X_pth = torch.from_numpy(X).unsqueeze(0).cuda()
    Q_pth = torch.from_numpy(Q).unsqueeze(1).cuda()

    ground_truth = list()
    for i in tqdm(range(0, len(Q), batch_size), desc="Computing ground truth"):
        Q_batch = Q_pth[i : i + batch_size]
        dists_batch = torch.norm(X_pth - Q_batch, dim=2)
        topk_batch = torch.argsort(dists_batch, dim=1)[:, :k]
        ground_truth.extend(topk_batch.cpu().numpy().tolist())

    print(f"Saving ground truth to {cache_path}")
    np.save(cache_path, np.array(ground_truth))

    return ground_truth


def exp_vectors_on_boundary():
    print("Loading dataset")
    dataset_config, dim = get_dataset_config("yfcc100m", test_size=0.01)
    train_vecs, test_vecs, train_mds, test_mds, metadata = load_dataset(dataset_config)

    print("Training KMeansTree")
    kmeans_tree = KMeansTree(n_clusters=16, max_leaf_size=128)
    kmeans_tree.fit(train_vecs)

    ground_truth = brute_force_knn_cuda(train_vecs, test_vecs, k=10)

    common_prefix_lens = []
    for query, topk in tqdm(
        zip(test_vecs, ground_truth), total=len(test_vecs), desc="Querying"
    ):
        query_path = kmeans_tree.closest_leaf(query).get_path()

        for neigh in topk:
            neigh_path = kmeans_tree.vec_to_leaf[neigh].get_path()
            common_prefix_len = sum(
                [1 for i, j in zip(query_path, neigh_path) if i == j]
            )
            common_prefix_lens.append(common_prefix_len)

    print(f"Average tree depth: {kmeans_tree.average_depth}")
    print(f"Average common prefix length: {np.mean(common_prefix_lens)}")
    print("Common prefix length distribution:", Counter(common_prefix_lens))


def exp_unfiltered_search(
    sample_size: int = 10000,
    beam_size: int = 0,
    nprobe: int = 3000,
    output_path: str | None = "exp_unfiltered_search.pkl",
    random_seed: int = 42,
):
    np.random.seed(random_seed)

    print("Loading dataset")
    dataset_config, dim = get_dataset_config("yfcc100m", test_size=0.01)
    train_vecs, test_vecs, train_mds, test_mds, metadata = load_dataset(dataset_config)
    sampled_idxs = np.random.choice(len(test_vecs), sample_size, replace=False)
    test_vecs = test_vecs[sampled_idxs]

    print("Training KMeansTree")
    kmeans_tree = KMeansTree(n_clusters=16, max_leaf_size=128)
    kmeans_tree.fit(train_vecs)

    ground_truth = brute_force_knn_cuda(train_vecs, test_vecs, k=10, batch_size=8)

    recalls = []
    n_dists = []
    query_stats = []

    for query, topk in tqdm(
        zip(test_vecs, ground_truth), total=len(test_vecs), desc="Querying"
    ):
        if beam_size <= 0:
            pred = kmeans_tree.query(query, k=10, nprobe=nprobe)
        else:
            pred = kmeans_tree.query_beam(
                query, k=10, beam_size=beam_size, nprobe=nprobe
            )

        n_dists.append(kmeans_tree.query_stat["n_dists"])
        recall = recall_at_k(pred, topk, k=10)
        recalls.append(recall)

        query_stats.append(
            {
                **kmeans_tree.query_stat,
                "query_path": kmeans_tree.closest_leaf(query).get_path(),
                "topk_path": [
                    kmeans_tree.vec_to_leaf[neigh].get_path() for neigh in topk
                ],
                "recall": recall,
            }
        )

    print(f"Average recall@10: {np.mean(recalls):.4f}")
    print(f"Average number of distance computations: {np.mean(n_dists):.4f}")

    if output_path is not None:
        with open(output_path, "wb") as f:
            print(f"Saving query stats to {output_path}")
            pkl.dump(query_stats, f)


if __name__ == "__main__":
    fire.Fire()
