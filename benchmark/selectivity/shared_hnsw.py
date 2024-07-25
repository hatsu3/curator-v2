import json
from itertools import product
from pathlib import Path

import fire

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, IndexProfiler
from benchmark.selectivity.dataset import SelectivityDataset
from indexes.hnsw_mt_hnswlib import HNSWMultiTenantHnswlib as SharedHNSW


def exp_shared_hnsw_selectivity(
    output_path: str,
    construction_ef: int = 32,
    m: int = 32,
    search_ef_space: list[int] = [32, 64, 128, 256],
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
):
    profiler = IndexProfiler()

    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)
    profiler.set_dataset(dataset)

    print(f"Building index with construction_ef = {construction_ef}, m = {m} ...")
    index_config = IndexConfig(
        index_cls=SharedHNSW,
        index_params={
            "construction_ef": construction_ef,
            "m": m,
            "max_elements": 1000000,
        },
        search_params={
            "search_ef": search_ef_space[0],
        },
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=False,
        batch_insert=False,
        track_stats=True,
    )

    results = list()
    for search_ef in search_ef_space:
        print(f"Querying index with search_ef = {search_ef} ...")
        profiler.set_index_search_params({"search_ef": search_ef})
        query_res = profiler.do_query(return_verbose=True, return_stats=True)
        query_res.pop("query_results")
        results.append(
            {
                "construction_ef": construction_ef,
                "m": m,
                "search_ef": search_ef,
                **query_res,
                **build_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_shared_hnsw_selectivity_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    construction_ef_space: list[int] = [16, 32, 64, 128],
    m_space: list[int] = [16, 32, 64, 128],
    search_ef_space: list[int] = [32, 64, 128, 256],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    output_dir: str | Path = "output/selectivity/shared_hnsw",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for construction_ef, m in product(construction_ef_space, m_space):
        task_name = f"ef{construction_ef}_m{m}"
        command = batch_profiler.build_command(
            module="benchmark.selectivity.shared_hnsw",
            func="exp_shared_hnsw_selectivity",
            output_path=str(results_dir / f"{task_name}.json"),
            construction_ef=construction_ef,
            m=m,
            search_ef_space=search_ef_space,
            dataset_cache_dir=dataset_cache_dir,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
