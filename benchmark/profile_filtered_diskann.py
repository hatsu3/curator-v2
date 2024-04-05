import logging
import os
import shutil
import uuid
from functools import partial
from itertools import product
from typing import IO

import fire
import numpy as np
import pandas as pd

from benchmark.config import DatasetConfig, IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config, get_memory_usage
from indexes.filtered_diskann import FilteredDiskANN


# this will be used to replace the profile function in the IndexProfiler class
def profile_filtered_diskann(
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
    index: FilteredDiskANN = self._initialize_index(index_config)

    # training not necessary for filtered diskann
    logging.info("Training index...")
    train_latency = 0.0

    # batch insert vectors into the index
    logging.info("Inserting vectors...")
    index.batch_create(train_vecs, [], train_mds)
    insert_latencies, access_grant_latencies, insert_grant_latencies = [0], [0], [0]
    index_size = get_memory_usage() - mem_before_init

    # query the index
    logging.info("Querying index...")
    if os.environ.get("BATCH_QUERY") is not None:
        logging.warning("DiskANN does not support batch queries. Flag ignored.")

    query_results, query_latencies = self.run_query(
        index, k, test_vecs, test_mds, verbose, log_file, timeout
    )

    # filtered diskann does not support deletion
    logging.info("Deleting vectors...")
    delete_latencies, update_latencies = [0], [0]

    assert self.multi_tenant, "FilteredDiskANN must be multi-tenant"

    return {
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


def exp_filtered_diskann(
    index_dir_prefix="index",
    ef_construct_space=[32, 64, 128],
    graph_degree_space=[16, 32, 64],
    alpha_space=[1.0, 1.5, 2.0],
    filter_ef_construct_space=[32, 64, 128],
    ef_search_space=[32, 64, 128],
    construct_threads=None,
    search_threads=1,
    dataset_key="arxiv-small",
    test_size=0.2,
    num_runs=1,
    timeout=600,
    output_path: str | None = None,
    cache_index=False,
):
    construct_threads = construct_threads or os.environ.get("OMP_NUM_THREADS", 16)

    if output_path is None:
        output_path = f"output/filtered_diskann_{dataset_key}.csv"

    index_dir = f"{index_dir_prefix}_{uuid.uuid4().hex[:8]}"
    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_configs = [
        IndexConfig(
            index_cls=FilteredDiskANN,
            index_params={
                "index_dir": index_dir,
                "d": dim,
                "ef_construct": ef_construct,
                "graph_degree": graph_degree,
                "alpha": alpha,
                "filter_ef_construct": filter_ef_construct,
                "construct_threads": construct_threads,
                "search_threads": search_threads,
            },
            search_params={
                "ef_search": ef_search,
            },
            train_params={
                "train_ratio": 1,
                "min_train": 50,
                "random_seed": 42,
            },
        )
        for ef_construct, graph_degree, alpha, filter_ef_construct, ef_search in product(
            ef_construct_space,
            graph_degree_space,
            alpha_space,
            filter_ef_construct_space,
            ef_search_space,
        )
    ]

    profiler = IndexProfiler(multi_tenant=True)
    profiler.profile = partial(profile_filtered_diskann, profiler)
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

    if not cache_index:
        shutil.rmtree(index_dir, ignore_errors=True)

    return results


if __name__ == "__main__":
    fire.Fire(exp_filtered_diskann)
