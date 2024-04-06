import logging
import os
from functools import partial
from itertools import product
from time import time
from typing import IO

import fire
import numpy as np
import pandas as pd

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config, get_memory_usage
from indexes.hybrid_curator import HybridCurator


# TODO: temporarily skip profiling of deletion
def profile_hybrid_curator(
    self,
    index_config: IndexConfig,
    dataset_config: DatasetConfig,
    k=10,
    verbose: bool = True,
    seed: int = 42,
    log_file: IO[str] | None = None,
    timeout: int | None = None,
):
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
        index.train(
            train_vecs,
            train_mds if self.multi_tenant else None,
            **index_config.train_params,
        )
    train_latency = time() - train_latency

    # insert vectors into the index
    logging.info("Inserting vectors...")
    (
        insert_latencies,
        access_grant_latencies,
        insert_grant_latencies,
        vector_creator,
    ) = self.run_insert(index, train_vecs, train_mds, verbose, log_file)

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
    delete_latencies, update_latencies = [0], [0]

    if self.multi_tenant:
        res = {
            "train_latency": train_latency,
            "index_size_kb": index_size,
            "query_results": query_results,
            "insert_latencies": insert_latencies,
            "access_grant_latencies": access_grant_latencies,
            "insert_grant_latencies": insert_grant_latencies,
            "query_latencies": query_latencies,
            "delete_latencies": delete_latencies,
            "update_latencies": update_latencies,
        }
        return res
    else:
        res = {
            "train_latency": train_latency,
            "index_size_kb": index_size,
            "query_results": query_results,
            "insert_latencies": insert_latencies,
            "query_latencies": query_latencies,
            "delete_latencies": delete_latencies,
        }
        return res


def exp_hybrid_curator(
    graph_degree_space=[12, 24, 36, 48],
    branch_factor_space=[4, 8, 16, 32],
    buf_capacity_space=[8, 16, 32, 64],
    alpha_space=[1.0, 1.5, 2.0],
    dataset_key="arxiv-small",
    test_size=0.2,
    num_runs=1,
    timeout=600,
    output_path: str | None = None,
):
    if output_path is None:
        output_path = f"output/hybrid_curator_{dataset_key}.csv"

    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_configs = [
        IndexConfig(
            index_cls=HybridCurator,
            index_params={
                "d": dim,
                "graph_degree": graph_degree,
                "branch_factor": branch_factor,
                "buf_capacity": buf_capacity,
            },
            search_params={
                "alpha": alpha,
            },
            train_params={
                "train_ratio": 1,
                "min_train": 50,
                "random_seed": 42,
            },
        )
        for graph_degree, branch_factor, buf_capacity, alpha in product(
            graph_degree_space, branch_factor_space, buf_capacity_space, alpha_space
        )
    ]

    profiler = IndexProfiler(multi_tenant=True)
    profiler.profile = partial(profile_hybrid_curator, profiler)
    results = profiler.batch_profile(
        index_configs, [dataset_config], num_runs=num_runs, timeout=timeout
    )

    if output_path is not None:
        df = pd.DataFrame(
            [
                {
                    **config.index_params,
                    **config.search_params,
                    **res,
                }
                for res, config in zip(results, index_configs)
            ]
        )
        df.to_csv(output_path, index=False)

    return results


if __name__ == "__main__":
    fire.Fire(exp_hybrid_curator)
