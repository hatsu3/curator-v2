import json
import pickle as pkl
import subprocess
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from queue import Queue
from typing import Any

import numpy as np
from tqdm import tqdm

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.utils import get_dataset_config, get_memory_usage, recall
from dataset import get_dataset, get_metadata
from dataset.utils import compute_ground_truth, load_sampled_metadata
from indexes.base import Index


class Dataset:
    def __init__(
        self,
        train_vecs: np.ndarray,
        train_mds: list[list[int]],
        test_vecs: np.ndarray,
        test_mds: list[list[int]],
        ground_truth: np.ndarray,
        all_labels: set[int],
    ):
        self.train_vecs = train_vecs
        self.train_mds = train_mds
        self.test_vecs = test_vecs
        self.test_mds = test_mds
        self.ground_truth = ground_truth
        self.all_labels = all_labels

    @property
    def dim(self):
        return self.train_vecs.shape[1]

    @property
    def num_labels(self):
        return len(self.all_labels)

    @property
    def label_selectivities(self):
        counter = Counter()
        for access_list in self.train_mds:
            counter.update(access_list)
        return {label: count / len(self.train_vecs) for label, count in counter.items()}

    @property
    def largest_cardinality(self):
        return int(max(self.label_selectivities.values()) * len(self.train_vecs))

    def num_vector_label_pairs(self, split: str = "train"):
        if split == "train":
            return sum(len(access_list) for access_list in self.train_mds)
        elif split == "test":
            return sum(len(access_list) for access_list in self.test_mds)
        else:
            raise ValueError(f"Invalid split: {split}")

    def avg_labels_per_vector(self, split: str = "train"):
        if split == "train":
            return self.num_vector_label_pairs(split) / len(self.train_vecs)
        elif split == "test":
            return self.num_vector_label_pairs(split) / len(self.test_vecs)
        else:
            raise ValueError(f"Invalid split: {split}")

    def get_random_split(
        self, split_n_labels: int, seed: int = 42, remap_labels: bool = False
    ) -> "Dataset":
        np.random.seed(seed)
        split_labels = set(
            np.random.choice(
                list(self.all_labels), split_n_labels, replace=False
            ).tolist()
        )

        split_train_mds = [
            [label for label in access_list if label in split_labels]
            for access_list in self.train_mds
        ]
        split_test_mds = [
            [label for label in access_list if label in split_labels]
            for access_list in self.test_mds
        ]

        split_ground_truth = list()
        orig_gt_gen = iter(self.ground_truth)
        for access_list in self.test_mds:
            for label in access_list:
                gt = next(orig_gt_gen)
                if label in split_labels:
                    split_ground_truth.append(gt)
        split_ground_truth = np.array(split_ground_truth)

        if remap_labels:
            label_map = {label: i for i, label in enumerate(sorted(split_labels))}
            split_train_mds = [
                [label_map[label] for label in access_list]
                for access_list in split_train_mds
            ]
            split_test_mds = [
                [label_map[label] for label in access_list]
                for access_list in split_test_mds
            ]

        return Dataset(
            train_vecs=self.train_vecs,
            train_mds=split_train_mds,
            test_vecs=self.test_vecs,
            test_mds=split_test_mds,
            ground_truth=split_ground_truth,
            all_labels=split_labels,
        )

    @classmethod
    def from_dataset_key(
        cls,
        dataset_key: str,
        test_size: float,
        cache_path: Path | None = None,
        k: int = 10,
        verbose: bool = True,
    ):
        dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)
        dataset_config.cache_path = Path(cache_path) if cache_path is not None else None
        return cls.from_config(dataset_config, k=k, verbose=verbose)

    @classmethod
    def from_config(
        cls, dataset_config: DatasetConfig, k: int = 10, verbose: bool = True
    ):
        if dataset_config.cache_path is not None:
            if dataset_config.cache_path.exists():
                print("Loading dataset from cache...")
                train_vecs = np.load(dataset_config.cache_path / "train_vecs.npy")
                test_vecs = np.load(dataset_config.cache_path / "test_vecs.npy")
                ground_truth = np.load(dataset_config.cache_path / "ground_truth.npy")
                all_labels = set(
                    json.load(open(dataset_config.cache_path / "all_labels.json"))
                )
                metadata = pkl.load(
                    open(dataset_config.cache_path / "metadata.pkl", "rb")
                )
                train_mds = pkl.load(
                    open(dataset_config.cache_path / "train_mds.pkl", "rb")
                )
                test_mds = pkl.load(
                    open(dataset_config.cache_path / "test_mds.pkl", "rb")
                )

                return cls(
                    train_vecs, train_mds, test_vecs, test_mds, ground_truth, all_labels
                )
            else:
                print("Cache not found. Generating dataset and saving to cache...")

        else:
            print("No cache path provided. Generating dataset...")

        if verbose:
            print("Loading dataset...")
        train_vecs, test_vecs, metadata = get_dataset(
            dataset_name=dataset_config.dataset_name, **dataset_config.dataset_params
        )

        if verbose:
            print("Loading metadata...")
        train_mds, test_mds = get_metadata(
            synthesized=dataset_config.synthesize_metadata,
            train_vecs=train_vecs,
            test_vecs=test_vecs,
            dataset_name=dataset_config.dataset_name,
            **dataset_config.metadata_params,
        )

        if verbose:
            print("Computing/loading ground truth...")
        ground_truth, all_labels = compute_ground_truth(
            train_vecs,
            train_mds,
            test_vecs,
            test_mds,
            k=k,
            metric=metadata["metric"],
            multi_tenant=True,
        )

        if verbose:
            print("Loading sampled metadata...")
        train_mds, test_mds = load_sampled_metadata(train_mds, test_mds, all_labels)

        if verbose:
            print(f"Dataset statistics:")
            print(f"  # train vectors: {len(train_vecs)}")
            print(f"  # test vectors: {len(test_vecs)}")
            print(f"  # labels: {len(all_labels)}")

        if dataset_config.cache_path is not None:
            assert not dataset_config.cache_path.exists(), "Cache already exists"
            dataset_config.cache_path.mkdir(parents=True, exist_ok=True)

            np.save(dataset_config.cache_path / "train_vecs.npy", train_vecs)
            np.save(dataset_config.cache_path / "test_vecs.npy", test_vecs)
            np.save(dataset_config.cache_path / "ground_truth.npy", ground_truth)

            with open(dataset_config.cache_path / "all_labels.json", "w") as f:
                json.dump([int(label) for label in all_labels], f)

            with open(dataset_config.cache_path / "metadata.pkl", "wb") as f:
                pkl.dump(metadata, f)
            with open(dataset_config.cache_path / "train_mds.pkl", "wb") as f:
                pkl.dump(train_mds, f)
            with open(dataset_config.cache_path / "test_mds.pkl", "wb") as f:
                pkl.dump(test_mds, f)

        return cls(train_vecs, train_mds, test_vecs, test_mds, ground_truth, all_labels)


class IndexProfiler:
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(self.seed)

        self.dataset: Dataset | None = None
        self.index: Index | None = None

    def set_dataset(self, dataset: Dataset):
        self.dataset = dataset
        return self

    def set_index(self, index: Index, track_stats: bool = False):
        self.index = index
        if track_stats:
            self.index.enable_stats_tracking(track_stats)
        return self

    def set_index_search_params(self, search_params: dict[str, Any]):
        assert self.index is not None, "Index not set"
        self.index.search_params = search_params
        return self

    def do_train(self, train_params: dict[str, Any] | None = None):
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"

        train_start = time.time()
        train_params = train_params or {}
        self.index.train(
            self.dataset.train_vecs, self.dataset.train_mds, **train_params
        )

        return {
            "train_latency": time.time() - train_start,
        }

    def do_insert(self, batch_insert: bool = False, with_labels: bool = True):
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"

        if batch_insert:
            print("Batch inserting vectors...")
            if with_labels:
                labels = np.arange(len(self.dataset.train_vecs)).tolist()
            else:
                labels = []

            insert_start = time.time()
            self.index.batch_create(
                self.dataset.train_vecs, labels, self.dataset.train_mds
            )
            results = {
                "batch_insert_latency": time.time() - insert_start,
            }

        else:
            insert_latencies = list()
            access_grant_latencies = list()

            insert_order = np.arange(len(self.dataset.train_vecs))
            np.random.shuffle(insert_order)
            insert_order = insert_order.tolist()

            for label in tqdm(
                insert_order,
                total=len(self.dataset.train_vecs),
                desc="Inserting vectors",
            ):
                vec = self.dataset.train_vecs[label]
                access_list = self.dataset.train_mds[label]

                if len(access_list) == 0:
                    continue

                insert_start = time.time()
                self.index.create(vec, label)
                insert_latencies.append(time.time() - insert_start)

                for tenant in access_list:
                    access_grant_start = time.time()
                    self.index.grant_access(label, int(tenant))
                    access_grant_latencies.append(time.time() - access_grant_start)

            results = {
                "insert_latencies": insert_latencies,
                "access_grant_latencies": access_grant_latencies,
            }

        try:
            self.index.shrink_to_fit()
        except NotImplementedError:
            pass

        return self._compute_metrics(results)

    def do_build(
        self,
        index_config: IndexConfig,
        track_stats: bool = False,
        do_train: bool = True,
        batch_insert: bool = False,
        with_labels: bool = True,
    ):
        mem_before_build = get_memory_usage()

        self.set_index(index_config.index_cls(**index_config.index_params), track_stats)
        train_metrics = self.do_train(index_config.train_params) if do_train else dict()
        insert_metrics = self.do_insert(batch_insert, with_labels)

        return {
            **train_metrics,
            **insert_metrics,
            "index_size_kb": get_memory_usage() - mem_before_build,
        }

    def do_query(
        self,
        batch_query: bool = False,
        num_threads: int = 1,
        k: int = 10,
        num_runs: int = 1,
        return_stats: bool = False,
        return_verbose: bool = False,
    ):
        if num_runs <= 1:
            return self.do_query_single_run(
                batch_query=batch_query,
                num_threads=num_threads,
                k=k,
                return_stats=return_stats,
                return_verbose=return_verbose,
            )
        else:
            multi_run_results = [
                self.do_query_single_run(
                    batch_query=batch_query,
                    num_threads=num_threads,
                    k=k,
                    return_stats=return_stats,
                    return_verbose=return_verbose,
                )
                for _ in range(num_runs)
            ]
            return self.aggregate_multi_run_results(multi_run_results)

    def do_query_single_run(
        self,
        batch_query: bool = False,
        num_threads: int = 1,
        k: int = 10,
        return_stats: bool = False,
        return_verbose: bool = False,
    ):
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"

        if batch_query:
            print(f"Batch querying index with {num_threads} threads...")
            query_start = time.time()
            query_results = self.index.batch_query(
                self.dataset.test_vecs,
                k,
                self.dataset.test_mds,
                num_threads=num_threads,
            )
            batch_query_latency = time.time() - query_start

            results: dict[str, Any] = {
                "query_results": query_results,
                "batch_query_latency": batch_query_latency,
            }

        else:
            query_results = list()
            query_latencies = list()
            query_stats = list()

            for vec, access_list in tqdm(
                zip(self.dataset.test_vecs, self.dataset.test_mds),
                total=len(self.dataset.test_vecs),
                desc="Querying index",
            ):
                for tenant_id in access_list:
                    query_start = time.time()
                    ids = self.index.query(vec, k=k, tenant_id=int(tenant_id))
                    query_latencies.append(time.time() - query_start)
                    query_results.append(ids)

                    if return_stats:
                        query_stats.append(self.index.get_search_stats())

            results: dict[str, Any] = {
                "query_results": query_results,
                "query_latencies": query_latencies,
            }

            if return_stats:
                results["query_stats"] = query_stats

        results["query_recalls"] = [
            recall([res], [gt])
            for res, gt in zip(query_results, self.dataset.ground_truth)
        ]
        results["recall_at_k"] = np.mean(results["query_recalls"]).item()

        if not return_verbose:
            results.pop("query_results")
            results.pop("query_recalls")

        return self._compute_metrics(results, return_verbose=return_verbose)

    def do_delete(self, delete_pct: float = 0.01):
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"

        delete_latencies = list()
        revoke_access_latencies = list()

        n_vecs = len(self.dataset.train_vecs)
        n_delete = int(n_vecs * delete_pct)
        delete_order = np.random.choice(n_vecs, n_delete, replace=False).tolist()

        for label in tqdm(
            delete_order,
            total=len(delete_order),
            desc="Deleting vectors",
        ):
            access_list = self.dataset.train_mds[label]

            if len(access_list) == 0:
                continue

            for tenant in access_list:
                revoke_access_start = time.time()
                self.index.revoke_access(label, int(tenant))
                revoke_access_latencies.append(time.time() - revoke_access_start)

            delete_start = time.time()
            self.index.delete_vector(label)
            delete_latencies.append(time.time() - delete_start)

        return self._compute_metrics(
            {
                "delete_latencies": delete_latencies,
                "revoke_access_latencies": revoke_access_latencies,
            }
        )

    def _compute_metrics(self, results: dict, return_verbose: bool = False):
        new_metrics = dict()
        for metric_name, metric in results.items():
            if metric_name.endswith("latencies"):
                op_name = metric_name[: -len("_latencies")]
                new_metrics.update(
                    {
                        f"{op_name}_qps": 1 / np.mean(metric),
                        f"{op_name}_lat_avg": np.mean(metric),
                        f"{op_name}_lat_std": np.std(metric),
                        f"{op_name}_lat_p50": np.percentile(metric, 50),
                        f"{op_name}_lat_p99": np.percentile(metric, 99),
                    }
                )
                if return_verbose:
                    new_metrics[metric_name] = metric
            else:
                new_metrics[metric_name] = metric

        return new_metrics

    def compute_per_tenant_metrics(self, query_results: dict):
        assert self.dataset is not None, "Dataset not set"
        assert "query_latencies" in query_results, "Batch query not supported"

        per_tenant_metrics = dict()
        for tenant_id in self.dataset.all_labels:
            tenant_query_lats = list()
            tenant_query_ress = list()
            tenant_selector = list()

            for i, (tid, lat, res) in enumerate(
                zip(
                    chain(*self.dataset.test_mds),
                    query_results["query_latencies"],
                    query_results["query_results"],
                )
            ):
                if tenant_id == tid:
                    tenant_query_lats.append(lat)
                    tenant_query_ress.append(res)
                    tenant_selector.append(i)

            tenant_gt = self.dataset.ground_truth[tenant_selector][
                : len(tenant_query_ress)
            ]
            per_tenant_metrics[tenant_id] = {
                "query_qps": 1 / np.mean(tenant_query_lats),
                "query_lat_avg": np.mean(tenant_query_lats),
                "query_lat_std": np.std(tenant_query_lats),
                "query_lat_p50": np.percentile(tenant_query_lats, 50),
                "query_lat_p99": np.percentile(tenant_query_lats, 99),
                "recall_at_k": recall(tenant_query_ress, tenant_gt),
            }

        return per_tenant_metrics

    def aggregate_multi_run_results(self, results: list[dict]):
        agg_res = dict()

        for metric_name in results[0].keys():
            if metric_name in ["train_latency", "index_size_kb"]:
                agg_res[f"{metric_name}_avg"] = np.mean(
                    [res[metric_name] for res in results]
                )
                agg_res[f"{metric_name}_std"] = np.std(
                    [res[metric_name] for res in results]
                )
                agg_res[f"{metric_name}_p50"] = np.percentile(
                    [res[metric_name] for res in results], 50
                )

            elif metric_name in ["recall_at_k"]:
                if len(set([res[metric_name] for res in results])) > 1:
                    print("Warning: recall_at_k is not consistent across runs")
                agg_res[metric_name] = results[-1][metric_name]

            elif metric_name.endswith("qps"):
                agg_res[f"{metric_name}_avg"] = np.mean(
                    [res[metric_name] for res in results]
                )
                agg_res[f"{metric_name}_std"] = np.std(
                    [res[metric_name] for res in results]
                )

            else:
                agg_res[metric_name] = results[-1][metric_name]

        return agg_res


class BatchProfiler:
    def __init__(
        self,
        cpu_groups: list[str],
        show_progress: bool = True,
        log_dir: str | Path | None = None,
        retry_timeout: int = 10,
    ):
        self.cpu_groups = cpu_groups
        self.show_progress = show_progress
        self.retry_timeout = retry_timeout

        if log_dir is not None:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.log_dir = None

        self.lock = threading.Lock()
        self.avail_cpus = set(self.cpu_groups)
        self.avail_cv = threading.Condition(self.lock)

        self.tasks = Queue()
        self.progress_bar: tqdm | None = None
        self.executor = ThreadPoolExecutor(max_workers=len(cpu_groups))

    def _scheduler(self):
        while not self.tasks.empty():
            cpus = self._get_available_cpus()
            task_name, cmd = self.tasks.get()
            self.executor.submit(self._run_task, cpus, cmd, task_name)

    def _get_available_cpus(self) -> str:
        with self.avail_cv:
            while not self.avail_cpus:
                self.avail_cv.wait()
            return self.avail_cpus.pop()

    def _run_task(self, cpus: str, cmd: str, task_name: str):
        if self.log_dir is not None:
            log_file = self.log_dir / f"{task_name}.log"
        else:
            log_file = Path("/dev/null")

        with log_file.open("w") as f:
            process = subprocess.Popen(
                f"taskset -c {cpus} \\\n" + cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                shell=True,
            )
            process.wait()

        with self.avail_cv:
            self.avail_cpus.add(cpus)
            self.avail_cv.notify()

        self.tasks.task_done()

        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def build_command(self, module: str, func: str | None = None, **kwargs):
        cmd = f"python -m {module} \\\n"
        if func is not None:
            cmd += f"\t{func} \\\n"
        for i, (k, v) in enumerate(kwargs.items()):
            if isinstance(v, list):
                v = f'"{v}"'
            cmd += f"\t\t--{k}={v}"
            if i < len(kwargs) - 1:
                cmd += " \\\n"

        return cmd

    def submit(self, task_name: str, cmd: str):
        self.tasks.put((task_name, cmd))
        if self.progress_bar is None:
            self.progress_bar = tqdm(
                total=self.tasks.qsize(),
                desc="Running tasks",
                disable=not self.show_progress,
            )
        else:
            self.progress_bar.total = self.tasks.qsize()

    def run(self):
        self._scheduler()

        self.executor.shutdown(wait=True)
        self.tasks.join()
        if self.progress_bar is not None:
            self.progress_bar.close()

        self.avail_cpus = set(self.cpu_groups)
        self.executor = ThreadPoolExecutor(max_workers=len(self.cpu_groups))
