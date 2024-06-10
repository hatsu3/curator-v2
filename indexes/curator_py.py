import enum
import heapq
import math
import os
import pickle as pkl
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Pool, shared_memory
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


@dataclass
class CuratorParam:
    # Construction parameters
    n_clusters: int = 16
    max_sl_size: int = 128
    max_leaf_size: int = 128
    clus_niter: int = 20
    bf_capacity: int = 1000
    bf_error_rate: float = 0.01

    # Search parameters
    nprobe: int = 3000
    prune_thres: float = 1.6
    var_boost: float = 0.2

    @property
    def construct_params(self) -> dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "max_sl_size": self.max_sl_size,
            "max_leaf_size": self.max_leaf_size,
            "clus_niter": self.clus_niter,
            "bf_capacity": self.bf_capacity,
            "bf_error_rate": self.bf_error_rate,
        }

    @property
    def search_params(self) -> dict[str, Any]:
        return {
            "nprobe": self.nprobe,
            "prune_thres": self.prune_thres,
            "var_boost": self.var_boost,
        }


class RunningMean:
    def __init__(self):
        self.sum = 0
        self.n = 0

    def add(self, x):
        self.sum += x
        self.n += 1

    def batch_add(self, xs):
        self.sum += np.sum(xs).item()
        self.n += len(xs)

    def remove(self, x):
        if self.n == 0:
            raise ValueError("Cannot remove from empty RunningMean")

        self.sum -= x
        self.n -= 1

        # reset to avoid numerical issues
        if self.n == 0:
            self.sum = 0

    def get(self):
        if self.n == 0:
            return 0
        return self.sum / self.n


class CuratorNode:
    def __init__(self, parent: Optional["CuratorNode"] = None):  # type: ignore
        # clustering tree
        self.parent = parent
        self.children: list["CuratorNode"] = []  # type: ignore
        self.centroid: np.ndarray | None = None
        self.variance = RunningMean()

        # access information
        self.shortlists: defaultdict[int, list[int]] = defaultdict(list)
        self.bloom_filter: set[int] = set()

        # storage at leaf nodes
        self.vectors: list[int] = []

    @property
    def level(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.level + 1

    @property
    def sibid(self) -> int:
        if self.parent is None:
            raise ValueError("Root node has no siblings")
        return self.parent.children.index(self)

    @property
    def path(self) -> list[int]:
        if self.parent is None:
            return []
        return self.parent.path + [self.sibid]

    def closest_child(self, x: np.ndarray) -> "CuratorNode":  # type: ignore
        return min(self.children, key=lambda c: np.linalg.norm(c.centroid - x))


class CuratorIndexPy:
    def __init__(self, dim: int, param: CuratorParam | None = None):
        self.dim = dim
        self.param = param or CuratorParam()
        self.root: "CuratorNode" | None = None  # type: ignore

        self.vec2data: dict[int, np.ndarray] = {}
        self.vec2leaf: dict[int, "CuratorNode"] = {}  # type: ignore
        self.vec2tnts: dict[int, set[int]] = {}
        self.tnt2nvecs: dict[int, int] = defaultdict(int)

    @property
    def all_tenants(self) -> set[int]:
        return set(self.tnt2nvecs.keys())

    @property
    def next_available_tenant(self) -> int:
        return max(self.all_tenants) + 1

    def train(self, X: np.ndarray):
        def _train_node(node: CuratorNode, X: np.ndarray):
            node.centroid = np.mean(X, axis=0)

            if X.shape[0] <= self.param.max_leaf_size:
                return

            kmeans = KMeans(
                n_clusters=self.param.n_clusters,
                init="random",
                n_init=1,
                max_iter=self.param.clus_niter,
                random_state=42,
            ).fit(X)

            for i in range(self.param.n_clusters):
                child = CuratorNode(parent=node)
                node.children.append(child)
                _train_node(child, X[kmeans.labels_ == i])

        self.root = CuratorNode()
        assert self.root is not None
        _train_node(self.root, X)

    def batch_insert(
        self,
        X: np.ndarray,
        access_lists: list[list[int]],
        labels: list[int] | None = None,
    ):
        if labels is None:
            labels = list(range(len(X)))

        def _batch_insert_node(
            node: CuratorNode,
            X: np.ndarray,
            labels: np.ndarray,
            access_lists: np.ndarray,
        ):
            if not node.children:
                node.vectors.extend(labels)
                self.vec2leaf.update({label: node for label in labels})
                return

            centroids = np.array([child.centroid for child in node.children])
            dists = np.linalg.norm(X[:, None] - centroids, axis=2)
            assigns = np.argmin(dists, axis=1)

            for i, child in enumerate(node.children):
                mask = assigns == i
                child.variance.batch_add(np.linalg.norm(X[mask] - centroids[i], axis=1))
                _batch_insert_node(child, X[mask], labels[mask], access_lists[mask])

        print("Batch inserting vectors")
        assert self.root is not None and self.root.centroid is not None
        self.root.variance.batch_add(np.linalg.norm(X - self.root.centroid, axis=1))

        _batch_insert_node(
            self.root, X, np.array(labels), np.array(access_lists, dtype=object)
        )

        for x, access_list, label in tqdm(
            zip(X, access_lists, labels), total=len(X), desc="Granting access"
        ):
            self.vec2data[label] = x
            self.vec2tnts[label] = set(access_list)
            for tenant in access_list:
                self._grant_access_at_node(self.root, label, tenant)

    def insert(self, x: np.ndarray, label: int):
        if label in self.vec2data:
            raise ValueError(f"Vector with label {label} already exists")

        path = self._get_path(x)
        for node in path:
            dist = np.linalg.norm(node.centroid - x)
            node.variance.add(dist)

        leaf = path[-1]
        leaf.vectors.append(label)

        self.vec2data[label] = x
        self.vec2leaf[label] = leaf
        self.vec2tnts[label] = set()

    def _get_path(self, x: np.ndarray) -> list[CuratorNode]:
        assert self.root is not None
        path = [self.root]
        curr = self.root
        while curr.children:
            curr = curr.closest_child(x)
            path.append(curr)
        return path

    def grant_access(self, label: int, tenant: int):
        assert self.root is not None, "Index is not trained yet"
        self._check_vector_exists(label)

        if tenant in self.vec2tnts[label]:
            raise ValueError(
                f"Vector with label {label} already granted to tenant {tenant}"
            )

        self._grant_access_at_node(self.root, label, tenant)
        self.vec2tnts[label].add(tenant)
        self.tnt2nvecs[tenant] += 1

    def _check_vector_exists(self, label: int):
        if label not in self.vec2data:
            raise ValueError(f"Vector with label {label} does not exist")

    def _get_closest_child(self, node: CuratorNode, label: int) -> CuratorNode:
        path = self._get_path_to_leaf(self.vec2leaf[label])
        return path[path.index(node) + 1]

    def _grant_access_at_node(self, node: CuratorNode, label: int, tenant: int):
        if not node.children:
            # if this is a leaf node
            node.shortlists[tenant].append(label)
        elif tenant not in node.bloom_filter or tenant in node.shortlists:
            # if this node contains no or few vectors of the tenant
            node.shortlists[tenant].append(label)

            # if the shortlist is too large, push down the vectors
            if len(node.shortlists[tenant]) > self.param.max_sl_size:
                for vec in node.shortlists[tenant]:
                    child = self._get_closest_child(node, vec)
                    self._grant_access_at_node(child, vec, tenant)
                del node.shortlists[tenant]
        else:
            # if this node contains many vectors of the tenant
            child = self._get_closest_child(node, label)
            self._grant_access_at_node(child, label, tenant)

        node.bloom_filter.add(tenant)

    def delete(self, label: int):
        self._check_vector_exists(label)

        leaf = self.vec2leaf[label]
        leaf.vectors.remove(label)

        path = self._get_path_to_leaf(leaf)
        for node in path:
            dist = np.linalg.norm(node.centroid - self.vec2data[label])
            node.variance.remove(dist)

        del self.vec2data[label]
        del self.vec2leaf[label]
        del self.vec2tnts[label]

    def _get_path_to_leaf(self, leaf: CuratorNode) -> list[CuratorNode]:
        path = [leaf]
        curr = leaf
        while curr.parent:
            curr = curr.parent
            path.append(curr)
        return path[::-1]

    def revoke_access(self, label: int, tenant: int):
        self._check_vector_exists(label)

        if tenant not in self.vec2tnts[label]:
            raise ValueError(
                f"Vector with label {label} not granted to tenant {tenant}"
            )

        leaf = self.vec2leaf[label]
        path = self._get_path_to_leaf(leaf)

        # update access list
        self.vec2tnts[label].remove(tenant)
        self.tnt2nvecs[tenant] -= 1

        # update shortlists
        merging = False
        for node in path[::-1]:
            # phase 2: recursively merge shortlists
            if merging:
                total_sl_size = 0
                for child in node.children:
                    # should not merge if a subtree contain more than limit relevant vectors
                    if tenant in child.bloom_filter and tenant not in child.shortlists:
                        total_sl_size += self.param.max_sl_size + 1
                        break

                    if tenant in child.shortlists:
                        total_sl_size += len(child.shortlists[tenant])

                # stop merging if the total shortlist size exceeds the limit
                if total_sl_size > self.param.max_sl_size:
                    break

                node.shortlists[tenant] = []
                for child in node.children:
                    if tenant in child.shortlists:
                        node.shortlists[tenant].extend(child.shortlists[tenant])
                        del child.shortlists[tenant]
                        child.bloom_filter.remove(tenant)

                assert node.shortlists[tenant]

            # phase 1: remove from shortlist
            elif tenant in node.shortlists:
                assert node.shortlists[tenant]
                node.shortlists[tenant].remove(label)
                if not node.shortlists[tenant]:
                    del node.shortlists[tenant]
                    node.bloom_filter.remove(tenant)

                merging = True  # enter phase 2

    def search(
        self, x: np.ndarray, k: int, tenant: int, return_stats: bool = False
    ) -> tuple[list[int], dict] | list[int]:
        assert self.root is not None, "Index is not trained yet"

        buckets = []
        frontier = [(np.linalg.norm(self.root.centroid - x), self.root)]
        n_cands = 0
        n_dists = 1

        while frontier and n_cands < self.param.nprobe:
            score, node = heapq.heappop(frontier)

            if tenant not in node.bloom_filter:
                continue
            elif tenant in node.shortlists:
                buckets.append((score, node))
                n_cands += len(node.shortlists[tenant])
            elif node.children:
                for child in node.children:
                    n_dists += 1
                    dist = np.linalg.norm(child.centroid - x)
                    var = child.variance.get()
                    score = dist - self.param.var_boost * var
                    heapq.heappush(frontier, (score, child))
            else:
                # false positive in Bloom filter
                pass

        if not buckets:
            return []

        buckets.sort()
        best_buck_score = buckets[0][0]
        results = []

        for score, node in buckets:
            if score > best_buck_score * self.param.prune_thres:
                break

            assert node.shortlists[tenant]

            for label in node.shortlists[tenant]:
                n_dists += 1
                dist = np.linalg.norm(self.vec2data[label] - x)
                if len(results) < k:
                    heapq.heappush(results, (-dist, label))
                else:
                    heapq.heappushpop(results, (-dist, label))

        if return_stats:
            return [label for _, label in sorted(results, reverse=True)], {
                "ndists": n_dists
            }
        else:
            return [label for _, label in sorted(results, reverse=True)]

    def search_v2(
        self, x: np.ndarray, k: int, tenant: int, return_stats: bool = False
    ) -> tuple[list[int], dict] | list[int]:
        assert self.root is not None, "Index is not trained yet"

        buckets = []
        frontier = [(np.linalg.norm(self.root.centroid - x), self.root)]
        n_dists = 1

        while frontier:
            score, node = heapq.heappop(frontier)

            if buckets and score > buckets[0][0] * self.param.prune_thres:
                break

            if tenant not in node.bloom_filter:
                continue
            elif tenant in node.shortlists:
                heapq.heappush(buckets, (score, node))
            elif node.children:
                for child in node.children:
                    n_dists += 1
                    dist = np.linalg.norm(child.centroid - x)
                    var = child.variance.get()
                    score = dist - self.param.var_boost * var
                    heapq.heappush(frontier, (score, child))
            else:
                # false positive in Bloom filter
                pass

        if not buckets:
            return []

        buckets.sort()
        best_buck_score = buckets[0][0]
        results = []

        for score, node in buckets:
            if score > best_buck_score * self.param.prune_thres:
                break

            assert node.shortlists[tenant]

            for label in node.shortlists[tenant]:
                n_dists += 1
                dist = np.linalg.norm(self.vec2data[label] - x)
                if len(results) < k:
                    heapq.heappush(results, (-dist, label))
                else:
                    heapq.heappushpop(results, (-dist, label))

        if return_stats:
            return [label for _, label in sorted(results, reverse=True)], {
                "ndists": n_dists
            }
        else:
            return [label for _, label in sorted(results, reverse=True)]

    def batch_search(
        self, X: np.ndarray, access_lists: list[list[int]], k: int
    ) -> tuple[list[list[int]], list[dict]]:
        results = []
        stats = []
        for x, access_list in zip(X, access_lists):
            for tenant in access_list:
                result, stat = self.search(x, k, tenant, return_stats=True)
                results.append(result)
                stats.append(stat)

        return results, stats

    # TODO: does not work efficiently now
    # NumPy arrays, especially large ones, can be costly to serialize and deserialize.
    # Use shared memory for passing large arrays to processes.
    def parallel_search(
        self,
        X: np.ndarray,
        access_lists: list[list[int]],
        k: int,
        num_processes: int | None = None,
    ) -> tuple[list[list[int]], list[dict]]:
        if num_processes is None:
            ncpus = os.cpu_count()
            assert ncpus is not None, "Cannot determine the number of CPUs"
            num_processes = ncpus - 1

        shm = shared_memory.SharedMemory(create=True, size=X.nbytes)
        X_shared = np.ndarray(X.shape, dtype=X.dtype, buffer=shm.buf)
        X_shared[:] = X

        chunk_size = math.ceil(len(X) / num_processes)
        X_chunks = [X_shared[i : i + chunk_size] for i in range(0, len(X), chunk_size)]
        access_lists_chunks = [
            access_lists[i : i + chunk_size]
            for i in range(0, len(access_lists), chunk_size)
        ]

        with Pool(num_processes) as pool:
            results_stats = pool.starmap(
                self.batch_search,
                [
                    (X_chunk, access_lists_chunk, k)
                    for X_chunk, access_lists_chunk in zip(
                        X_chunks, access_lists_chunks
                    )
                ],
            )

        results = [res for chunk in results_stats for res in chunk[0]]
        stats = [stat for chunk in results_stats for stat in chunk[1]]
        return results, stats

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pkl.dump(self, open(path, "wb"))

    @classmethod
    def load(cls, path: str) -> "CuratorIndexPy":
        index = pkl.load(open(path, "rb"))
        return index

    def print_index(self):
        def print_node(node, indent):
            print(
                "  " * indent,
                node.level,
                node.bloom_filter,
                dict(node.shortlists),
                node.vectors,
            )
            for child in node.children:
                print_node(child, indent + 1)

        print_node(self.root, 0)

    def sanity_check(self, access_lists: dict[int, list[int]] | None = None):
        def check_bloom_filter(node) -> bool:
            for child in node.children:
                if not check_bloom_filter(child):
                    return False

            expected_bf = set(node.shortlists.keys())
            for child in node.children:
                expected_bf.update(child.bloom_filter)

            return node.bloom_filter == expected_bf

        def check_shortlists(node) -> tuple[bool, list[int]]:
            sl_in_desc = set()
            for child in node.children:
                success, sls = check_shortlists(child)
                if not success:
                    return False, []

                sl_in_desc.update(sls)

            # oversize check
            for shortlist in node.shortlists.values():
                if len(shortlist) > self.param.max_sl_size:
                    print("Oversized shortlist")
                    return False, []

            # fail to merge check
            if node.children:
                all_tenants = set()
                for child in node.children:
                    for tenant in child.shortlists.keys():
                        all_tenants.add(tenant)

                for tenant in all_tenants:
                    total_sl_size = 0

                    for child in node.children:
                        if tenant not in child.bloom_filter:
                            continue
                        elif tenant in child.shortlists:
                            total_sl_size += len(child.shortlists[tenant])
                        else:
                            total_sl_size += self.param.max_sl_size + 1
                            break

                    if total_sl_size <= self.param.max_sl_size:
                        print("Fail to merge")
                        return False, []

            # duplicate check
            sl_in_node = set(node.shortlists.keys())
            if sl_in_desc & sl_in_node:
                print("Duplicate shortlists")
                return False, []

            all_shortlists = list(sl_in_desc | sl_in_node)
            return True, all_shortlists

        def gather_stored_vectors(node):
            stored_vectors: dict[int, list[int]] = dict()

            for child in node.children:
                child_data = gather_stored_vectors(child)
                for label, tenants in child_data.items():
                    if label in stored_vectors:
                        stored_vectors[label].extend(tenants)
                    else:
                        stored_vectors[label] = tenants

            for tenant, shortlist in node.shortlists.items():
                for label in shortlist:
                    if label in stored_vectors:
                        stored_vectors[label].append(tenant)
                    else:
                        stored_vectors[label] = [tenant]

            return stored_vectors

        def check_storage(access_lists):
            stored_vectors = gather_stored_vectors(self.root)
            if set(stored_vectors.keys()) != set(access_lists.keys()):
                print("Stored vectors are incorrect")
                return False

            for label, tenants in stored_vectors.items():
                if len(tenants) != len(set(tenants)):
                    print("Duplicate tenants in access lists")
                    return False

                if set(tenants) != set(access_lists[label]):
                    print("Access lists are incorrect")
                    return False

            return True

        assert self.root is not None, "Index is not trained yet"

        if not check_bloom_filter(self.root):
            print("Bloom filters are incorrect")
            return False

        if not check_shortlists(self.root)[0]:
            print("Shortlists are incorrect")
            return False

        if access_lists is not None:
            if not check_storage(access_lists):
                print("Storage is incorrect")
                return False

        return True

    def search_with_complex_predicate(
        self, x: np.ndarray, k: int, filter_str: str
    ) -> list[int]:
        assert self.root is not None, "Index is not trained yet"

        def update_var_map(node: CuratorNode, var_map: dict[int, Subset]):
            for label, subset in var_map.items():
                if subset.type != SubsetType.UNK:
                    continue

                if label not in node.bloom_filter:
                    var_map[label] = Subset(SubsetType.NONE)
                elif label in node.shortlists:
                    var_map[label] = Subset(
                        SubsetType.SOME, set(node.shortlists[label])
                    )

        def get_child_data(
            node: CuratorNode, child: CuratorNode, data: set[int]
        ) -> set[int]:
            results = set()

            for id in data:
                path = self._get_path_to_leaf(self.vec2leaf[id])
                assert node in path and child in path
                if path.index(node) + 1 == path.index(child):
                    results.add(id)

            return results

        def get_child_var_map(
            node: CuratorNode, child: CuratorNode, var_map: dict[int, Subset]
        ):
            child_var_map = var_map.copy()

            for label, subset in var_map.items():
                if subset.type in [SubsetType.SOME, SubsetType.MOST]:
                    assert subset.data is not None
                    child_data = get_child_data(node, child, subset.data)
                    child_var_map[label] = Subset(subset.type, child_data)

            return child_var_map

        filter = SetValuedBoolExprFilter(filter_str)
        
        var_maps = {
            self.root: {label: Subset(SubsetType.UNK) for label in filter.labels}
        }

        root_dist = np.linalg.norm(self.root.centroid - x)
        root_score = root_dist - self.param.var_boost * self.root.variance.get()
        frontier = [(root_score, self.root)]
        
        buckets = []
        n_cands = 0

        while frontier and n_cands < self.param.nprobe:
            score, node = heapq.heappop(frontier)

            var_map = var_maps[node]
            update_var_map(node, var_map)
            
            subset = filter.evaluate(var_map)

            if subset.type == SubsetType.NONE:
                continue
            elif subset.type == SubsetType.SOME:
                assert subset.data is not None
                buckets.append((score, subset.data))
                n_cands += len(subset.data)
            else:  # MOST, ALL, UNKNOWN
                if node.children:
                    for child in node.children:
                        var_maps[child] = get_child_var_map(node, child, var_map)
                        
                        dist = np.linalg.norm(child.centroid - x)
                        score = dist - self.param.var_boost * child.variance.get()
                        heapq.heappush(frontier, (score, child))
                else:
                    if subset.type == SubsetType.MOST:
                        assert subset.data is not None
                        buckets.append((score, set(node.vectors) - subset.data))
                        n_cands += len(set(node.vectors) - subset.data)
                    elif subset.type == SubsetType.ALL:
                        buckets.append((score, set(node.vectors)))
                        n_cands += len(node.vectors)
                    else:
                        raise ValueError("Subset type should not be UNK in leaf node")

        if not buckets:
            return []

        buckets.sort()
        best_buck_score = buckets[0][0]
        results = []

        for score, vecs in buckets:
            if score > best_buck_score * self.param.prune_thres:
                break

            for label in vecs:
                dist = np.linalg.norm(self.vec2data[label] - x)
                if len(results) < k:
                    heapq.heappush(results, (-dist, label))
                else:
                    heapq.heappushpop(results, (-dist, label))

        return [label for _, label in sorted(results, reverse=True)]

    def exhaustive_search_with_complex_predicate(
        self, filter_str: str
    ) -> list[int]: 
        filter = BoolExprFilter(filter_str)
        results = []
        
        for id, access_list in self.vec2tnts.items():
            if filter.evaluate(list(access_list)):
                results.append(id)
        
        return results

    def batch_grant_access(self, labels: list[int], tenant: int):
        assert self.root is not None, "Index is not trained yet"
        self._batch_grant_access_at_node(self.root, labels, tenant)

        for label in labels:
            self.vec2tnts[label].add(tenant)

    def _batch_grant_access_at_node(
        self, node: CuratorNode, labels: list[int], tenant: int
    ):
        if not node.children or len(labels) <= self.param.max_sl_size:
            node.shortlists[tenant] = labels
        else:
            child_to_labels = defaultdict(list)
            for vec in labels:
                child = self._get_closest_child(node, vec)
                child_to_labels[child].append(vec)

            for child, vecs in child_to_labels.items():
                self._batch_grant_access_at_node(child, vecs, tenant)

        node.bloom_filter.add(tenant)


class BoolExprFilter:
    def __init__(self, expr: str):
        self.expr = expr

    @property
    def labels(self) -> set[int]:
        labels = set(self.expr.split()) - {"AND", "OR", "NOT"}
        return set(map(int, labels))

    def evaluate(self, access_list: list[int]) -> bool:
        tokens = self.expr.split()
        stack = []

        def eval_token(token):
            try:
                return int(token) in access_list
            except ValueError:
                raise ValueError(f"Malformed formula: {self.expr}")

        for token in reversed(tokens):
            if token in ("AND", "OR", "NOT"):
                if token == "AND":
                    a = stack.pop()
                    b = stack.pop()
                    stack.append(a and b)
                elif token == "OR":
                    a = stack.pop()
                    b = stack.pop()
                    stack.append(a or b)
                elif token == "NOT":
                    a = stack.pop()
                    stack.append(not a)
            else:
                stack.append(eval_token(token))

        return stack.pop()


class SubsetType(enum.Enum):
    NONE = enum.auto()
    SOME = enum.auto()
    MOST = enum.auto()
    ALL = enum.auto()
    UNK = enum.auto()


class Subset:
    def __init__(self, type: SubsetType, data: set[int] | None = None):
        self.type = type
        self.data = data

    def __str__(self) -> str:
        if self.data is None:
            return f"[{self.type}]"
        else:
            return f"[{self.type} {self.data}]"


class SetValuedBoolExprFilter:
    def __init__(self, expr: str):
        self.expr = expr

    @property
    def labels(self) -> set[int]:
        labels = set(self.expr.split()) - {"AND", "OR", "NOT"}
        return set(map(int, labels))

    def evaluate(self, value_map: dict[int, Subset]) -> Subset:
        tokens = self.expr.split()
        stack = []

        def eval_token(token):
            try:
                return value_map[int(token)]
            except ValueError:
                raise ValueError(f"Malformed formula: {self.expr}")

        for token in reversed(tokens):
            if token in ("AND", "OR", "NOT"):
                if token == "AND":
                    a = stack.pop()
                    b = stack.pop()
                    stack.append(self.eval_and(a, b))
                elif token == "OR":
                    a = stack.pop()
                    b = stack.pop()
                    stack.append(self.eval_or(a, b))
                elif token == "NOT":
                    a = stack.pop()
                    stack.append(self.eval_not(a))
            else:
                stack.append(eval_token(token))

        return stack.pop()

    def eval_and(self, a: Subset, b: Subset) -> Subset:
        if a.type == SubsetType.NONE or b.type == SubsetType.NONE:
            return Subset(SubsetType.NONE)
        elif a.type == SubsetType.ALL:
            return b
        elif b.type == SubsetType.ALL:
            return a
        elif a.type == SubsetType.SOME and b.type == SubsetType.SOME:
            assert a.data is not None and b.data is not None
            return Subset(SubsetType.SOME, a.data & b.data)
        elif a.type == SubsetType.MOST and b.type == SubsetType.MOST:
            assert a.data is not None and b.data is not None
            return Subset(SubsetType.MOST, a.data | b.data)
        elif a.type == SubsetType.SOME and b.type == SubsetType.MOST:
            assert a.data is not None and b.data is not None
            return Subset(SubsetType.SOME, a.data - b.data)
        elif a.type == SubsetType.MOST and b.type == SubsetType.SOME:
            assert a.data is not None and b.data is not None
            return Subset(SubsetType.SOME, b.data - a.data)
        else:
            return Subset(SubsetType.UNK)

    def eval_or(self, a: Subset, b: Subset) -> Subset:
        if a.type == SubsetType.ALL or b.type == SubsetType.ALL:
            return Subset(SubsetType.ALL)
        elif a.type == SubsetType.NONE:
            return b
        elif b.type == SubsetType.NONE:
            return a
        elif a.type == SubsetType.SOME and b.type == SubsetType.SOME:
            assert a.data is not None and b.data is not None
            return Subset(SubsetType.SOME, a.data | b.data)
        elif a.type == SubsetType.MOST and b.type == SubsetType.MOST:
            assert a.data is not None and b.data is not None
            return Subset(SubsetType.MOST, a.data & b.data)
        elif a.type == SubsetType.SOME and b.type == SubsetType.MOST:
            assert a.data is not None and b.data is not None
            return Subset(SubsetType.MOST, b.data - a.data)
        elif a.type == SubsetType.MOST and b.type == SubsetType.SOME:
            assert a.data is not None and b.data is not None
            return Subset(SubsetType.MOST, a.data - b.data)
        else:
            return Subset(SubsetType.UNK)

    def eval_not(self, a: Subset) -> Subset:
        if a.type == SubsetType.ALL:
            return Subset(SubsetType.NONE)
        elif a.type == SubsetType.NONE:
            return Subset(SubsetType.ALL)
        elif a.type == SubsetType.SOME:
            assert a.data is not None
            return Subset(SubsetType.MOST, a.data)
        elif a.type == SubsetType.MOST:
            assert a.data is not None
            return Subset(SubsetType.SOME, a.data)
        else:
            return Subset(SubsetType.UNK)
