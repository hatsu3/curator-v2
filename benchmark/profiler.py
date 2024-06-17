import logging
import os
import pickle as pkl
from itertools import chain
from multiprocessing import Process, Queue
from tempfile import TemporaryDirectory
from time import time
from typing import IO, Sequence

import numpy as np
from tqdm import tqdm

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.utils import get_memory_usage, recall
from dataset import get_dataset, get_metadata
from dataset.utils import compute_ground_truth, load_sampled_metadata
from indexes.base import Index

# if OMP_NUM_THREADS is not set, set it to 1
if "OMP_NUM_THREADS" not in os.environ:
    print("OMP_NUM_THREADS not set. Setting it to 1.")
    os.environ["OMP_NUM_THREADS"] = "1"
else:
    print("OMP_NUM_THREADS:", os.environ["OMP_NUM_THREADS"])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class IndexProfiler:
    def __init__(self):
        self.loaded_dataset = None

    def profile(
        self,
        index_config: IndexConfig,
        dataset_config: DatasetConfig,
        k=10,
        verbose: bool = True,
        seed: int = 42,
        log_file: IO[str] | None = None,
        timeout: int | None = None,
    ) -> dict:
        logging.info(
            "\n\n"
            "=========================================\n"
            "Index config: %s\n"
            "=========================================\n"
            "Dataset config: %s\n"
            "=========================================\n",
            index_config,
            dataset_config,
        )

        np.random.seed(seed)

        logging.info("Loading dataset...")
        assert (
            self.loaded_dataset is not None and self.loaded_dataset[0] == dataset_config
        ), "Dataset must be loaded before profiling"

        train_vecs, train_mds, test_vecs, test_mds = self.loaded_dataset[1][:4]
        all_tenant_ids = self.loaded_dataset[1][-1]

        logging.info("Initializing index...")
        mem_before_init = get_memory_usage()
        index = self._initialize_index(index_config)

        # train the index (if necessary)
        logging.info("Training index...")
        train_latency = time()
        if index_config.train_params is not None:
            index.train(train_vecs, train_mds, **index_config.train_params)
        train_latency = time() - train_latency

        # insert vectors into the index
        logging.info("Inserting vectors...")
        insert_latencies, access_grant_latencies = self.run_insert(
            index, train_vecs, train_mds, verbose, log_file
        )

        index_size = get_memory_usage() - mem_before_init

        # query the index
        logging.info("Querying index...")
        if os.environ.get("BATCH_QUERY") is None:
            query_results, query_latencies = self.run_query(
                index, k, test_vecs, test_mds, verbose, log_file, timeout
            )
        else:
            query_results, query_latencies = self.run_batch_query(
                index,
                k,
                test_vecs,
                test_mds,
                all_tenant_ids,
                int(os.environ["OMP_NUM_THREADS"]),
                verbose,
                log_file,
                timeout,
            )

        logging.info("Deleting vectors...")
        delete_latencies, revoke_access_latencies = self.run_delete(
            index, train_vecs, train_mds, verbose, log_file
        )

        res = {
            "train_latency": train_latency,
            "index_size_kb": index_size,
            "query_results": query_results,
            "insert_latencies": insert_latencies,
            "access_grant_latencies": access_grant_latencies,
            "query_latencies": query_latencies,
            "delete_latencies": delete_latencies,
            "revoke_access_latencies": revoke_access_latencies,
        }
        return res

    def run_insert(self, index, train_vecs, train_mds, verbose, log_file):
        insert_latencies = list()
        access_grant_latencies = list()

        # shuffle the order of insertion
        insert_order = np.arange(len(train_vecs))
        np.random.shuffle(insert_order)

        for label in tqdm(
            insert_order, total=len(train_vecs), disable=not verbose, file=log_file
        ):
            label = int(label)
            vec = train_vecs[label]
            access_list = train_mds[label]

            if len(access_list) == 0:
                continue

            insert_start = time()
            index.create(vec, label)
            duration = time() - insert_start
            insert_latencies.append(duration)

            for tenant in access_list:
                access_grant_start = time()
                index.grant_access(label, int(tenant))  # type: ignore
                duration = time() - access_grant_start
                access_grant_latencies.append(duration)

        try:
            index.shrink_to_fit()
        except NotImplementedError:
            pass

        return insert_latencies, access_grant_latencies

    def run_query(self, index, k, test_vecs, test_mds, verbose, log_file, timeout):
        query_results = list()
        query_latencies = list()

        pbar = tqdm(
            enumerate(zip(test_vecs, test_mds)),
            total=len(test_vecs),
            disable=not verbose,
            desc="Querying index",
            file=log_file,
        )
        for i, (vec, tenant_ids) in pbar:
            elapsed = pbar.format_dict["elapsed"]

            if i % 100 == 0 and timeout is not None and elapsed > timeout:
                logging.info("Querying is taking too long. Stopping...")
                break

            for tenant_id in tenant_ids:
                query_start = time()
                ids = index.query(vec, k=k, tenant_id=int(tenant_id))
                query_latencies.append(time() - query_start)
                query_results.append(ids)

        return query_results, query_latencies

    def run_batch_query(
        self,
        index,
        k,
        test_vecs,
        test_mds,
        all_tenant_ids,
        num_threads,
        verbose,
        log_file,
        timeout,
    ):
        query_latencies = list()

        pbar = tqdm(
            all_tenant_ids,
            desc="Batch querying index",
            disable=not verbose,
            file=log_file,
        )
        for tenant_id in pbar:
            if timeout is not None and pbar.format_dict["elapsed"] > timeout:
                logging.info("Querying is taking too long. Stopping...")
                break

            tenant_vecs_mask = np.array(
                [tenant_id in access_list for access_list in test_mds]
            )
            tenant_vecs = test_vecs[tenant_vecs_mask]

            query_start = time()
            query_results = index.batch_query(
                tenant_vecs, k=k, tenant_id=int(tenant_id), num_threads=num_threads
            )
            batch_query_lat = time() - query_start
            query_latencies.extend(
                [batch_query_lat / len(tenant_vecs)] * len(tenant_vecs)
            )

        return query_results, query_latencies

    def run_delete(
        self,
        index,
        train_vecs,
        train_mds,
        verbose,
        log_file,
        delete_pct=0.01,
    ):
        delete_latencies = list()
        revoke_access_latencies = list()

        delete_order = np.arange(len(train_vecs))
        np.random.shuffle(delete_order)
        n_delete = int(len(train_vecs) * delete_pct)
        delete_order = delete_order[:n_delete]

        for label in tqdm(
            delete_order, total=len(delete_order), disable=not verbose, file=log_file
        ):
            label = int(label)
            access_list = train_mds[label]

            if len(access_list) == 0:
                continue

            for tenant in access_list:
                revoke_access_start = time()
                index.revoke_access(label, int(tenant))
                revoke_access_latencies.append(time() - revoke_access_start)

            delete_start = time()
            index.delete_vector(label)
            delete_latencies.append(time() - delete_start)

        return delete_latencies, revoke_access_latencies

    def profile_worker(self, queue, args):
        index_config, dataset_config, k, __, log_file_path, timeout = args

        print("Running experiment for index config:", index_config)

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=log_file_path,
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
        )

        with open(log_file_path, "a") as f:
            results = self.profile(
                index_config,
                dataset_config,
                k=k,
                verbose=True,
                log_file=f,
                timeout=timeout,
            )

        print("Single run results:", self._compute_metrics(results))
        queue.put(results)

    def batch_profile(
        self,
        index_configs: Sequence[IndexConfig],
        dataset_configs: Sequence[DatasetConfig],
        k: int = 10,
        num_runs: int = 1,
        timeout: int | None = 600,
        verbose: bool = True,
    ) -> list[dict]:
        results = list()
        with TemporaryDirectory(prefix="index_profiler_") as tempdir:
            print(f"Logging to {tempdir}...")

            for dataset_config in dataset_configs:
                dataset = self._load_dataset(dataset_config, k=k)
                self.loaded_dataset = (dataset_config, dataset)

                args_list = [
                    (
                        index_config,
                        dataset_config,
                        k,
                        verbose,
                        os.path.join(tempdir, f"log_{i}.txt"),
                        timeout,
                    )
                    for i, index_config in enumerate(index_configs)
                ]

                for args in args_list:
                    res_list = []
                    for _ in range(num_runs):
                        queue = Queue()
                        process = Process(
                            target=self.profile_worker, args=(queue, args)
                        )
                        process.start()
                        res_list.append(queue.get())
                        process.join()

                    self._save_per_tenant_metrics(args, res_list)
                    res_list = [self._compute_metrics(res) for res in res_list]
                    agg_res = self._aggregate_results(res_list)
                    print("Index config:", args[0])
                    print("Aggregated results:", agg_res)
                    results.append(agg_res)

        return results

    def _compute_metrics(self, results: dict):
        new_results = dict()
        for metric_name, metric in results.items():
            if metric_name.endswith("latencies"):
                op_name = metric_name[: -len("_latencies")]
                new_results.update(
                    {
                        f"{op_name}_qps": 1 / np.mean(metric),
                        f"{op_name}_lat_avg": np.mean(metric),
                        f"{op_name}_lat_std": np.std(metric),
                        f"{op_name}_lat_p50": np.percentile(metric, 50),
                        f"{op_name}_lat_p99": np.percentile(metric, 99),
                    }
                )
            elif metric_name == "query_results":
                assert self.loaded_dataset is not None
                ground_truth = self.loaded_dataset[1][-2]
                query_results = metric
                new_results["recall_at_k"] = recall(
                    query_results, ground_truth[: len(query_results)]
                )
            else:
                new_results[metric_name] = metric
        return new_results

    def _compute_per_tenant_metrics(self, results: dict):
        assert self.loaded_dataset is not None
        test_mds = self.loaded_dataset[1][3]
        ground_truth = self.loaded_dataset[1][-2]
        tenant_ids = sorted(self.loaded_dataset[1][-1])

        per_tenant_metrics = dict()
        for tenant_id in tenant_ids:
            tenant_query_lats = list()
            tenant_query_ress = list()
            tenant_selector = list()

            for i, (tid, lat, res) in enumerate(
                zip(
                    chain(*test_mds),
                    results["query_latencies"],
                    results["query_results"],
                )
            ):
                if tenant_id == tid:
                    tenant_query_lats.append(lat)
                    tenant_query_ress.append(res)
                    tenant_selector.append(i)

            tenant_gt = ground_truth[tenant_selector][: len(tenant_query_ress)]
            per_tenant_metrics[tenant_id] = {
                "query_qps": 1 / np.mean(tenant_query_lats),
                "query_lat_avg": np.mean(tenant_query_lats),
                "query_lat_std": np.std(tenant_query_lats),
                "query_lat_p50": np.percentile(tenant_query_lats, 50),
                "query_lat_p99": np.percentile(tenant_query_lats, 99),
                "recall_at_k": recall(tenant_query_ress, tenant_gt),
            }

        return per_tenant_metrics

    def _save_per_tenant_metrics(self, args, res_list):
        per_tenant_metric_path = os.environ.get("SAVE_PER_TENANT_METRICS")
        if per_tenant_metric_path is not None:
            print("Computing per-tenant metrics using results from the first run...")
            per_tenant_metrics = self._compute_per_tenant_metrics(res_list[0])
            with open(per_tenant_metric_path, "wb") as f:
                pkl.dump(
                    {
                        "index_config": str(args[0]),
                        "dataset_config": str(args[1]),
                        "per_tenant_metrics": per_tenant_metrics,
                    },
                    f,
                )

    def _aggregate_results(self, res_list: list[dict]):
        agg_res = dict()

        for metric_name in res_list[0].keys():
            if metric_name in ["train_latency", "index_size_kb"]:
                agg_res[f"{metric_name}_avg"] = np.mean(
                    [res[metric_name] for res in res_list]
                )
                agg_res[f"{metric_name}_std"] = np.std(
                    [res[metric_name] for res in res_list]
                )
                agg_res[f"{metric_name}_p50"] = np.percentile(
                    [res[metric_name] for res in res_list], 50
                )

            elif metric_name in ["recall_at_k"]:
                if len(set([res[metric_name] for res in res_list])) > 1:
                    print(
                        "Warning: recall_at_k is not consistent across runs: "
                        f"{[res[metric_name] for res in res_list]}. "
                        "Using the last value."
                    )
                agg_res[metric_name] = res_list[-1][metric_name]

            elif metric_name.endswith("qps"):
                agg_res[f"{metric_name}_avg"] = np.mean(
                    [res[metric_name] for res in res_list]
                )
                agg_res[f"{metric_name}_std"] = np.std(
                    [res[metric_name] for res in res_list]
                )

            else:
                agg_res[metric_name] = res_list[-1][metric_name]

        return agg_res

    def _load_dataset(self, dataset_config: DatasetConfig, k: int):
        logging.info("Loading dataset...")
        train_vecs, test_vecs, metadata = get_dataset(
            dataset_name=dataset_config.dataset_name, **dataset_config.dataset_params
        )
        logging.info("Loading metadata...")
        train_mds, test_mds = get_metadata(
            synthesized=dataset_config.synthesize_metadata,
            train_vecs=train_vecs,
            test_vecs=test_vecs,
            dataset_name=dataset_config.dataset_name,
            **dataset_config.metadata_params,
        )
        logging.info("Computing ground truth...")
        ground_truth, train_cates = compute_ground_truth(
            train_vecs,
            train_mds,
            test_vecs,
            test_mds,
            k=k,
            metric=metadata["metric"],
            multi_tenant=True,
        )
        logging.info("Loading sampled metadata...")
        train_mds, test_mds = load_sampled_metadata(train_mds, test_mds, train_cates)

        # print some statistics about the dataset
        logging.info("Dataset statistics:")
        logging.info("  # train vectors: %d", len(train_vecs))
        logging.info("  # test vectors: %d", len(test_vecs))
        logging.info("  # unique categories: %d", len(train_cates))

        return train_vecs, train_mds, test_vecs, test_mds, ground_truth, train_cates

    def _initialize_index(self, index_config: IndexConfig) -> Index:
        index = index_config.index_cls(
            **index_config.index_params, **index_config.search_params
        )
        return index
