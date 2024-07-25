import json
from itertools import product
from pathlib import Path
import tempfile

import fire

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, IndexProfiler
from benchmark.selectivity.dataset import SelectivityDataset
from indexes.filtered_diskann import FilteredDiskANN


def exp_filtered_diskann_selectivity(
    output_path: str,
    graph_degree: int = 32,
    ef_construct: int = 32,
    alpha: float = 1.2,
    ef_search_space: list[int] = [32, 64, 128],
    construct_threads: int = 1,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
):
    profiler = IndexProfiler()

    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)
    profiler.set_dataset(dataset)

    print(
        f"Building index with graph_degree = {graph_degree}, "
        f"ef_construct = {ef_construct}, alpha = {alpha} ..."
    )
    with tempfile.TemporaryDirectory() as index_dir:
        index_config = IndexConfig(
            index_cls=FilteredDiskANN,
            index_params={
                "index_dir": index_dir,
                "d": dataset.dim,
                "ef_construct": ef_construct,
                "graph_degree": graph_degree,
                "alpha": alpha,
                "filter_ef_construct": ef_construct,
                "construct_threads": construct_threads,
                "search_threads": 1,
                "cache_index": True,
            },
            search_params={
                "ef_search": 32,
            },
            train_params={
                "train_ratio": 1,
                "min_train": 50,
                "random_seed": 42,
            },
        )
        build_results = profiler.do_build(
            index_config=index_config,
            do_train=False,
            batch_insert=True,
            track_stats=True,
        )

        results = list()
        for ef_search in ef_search_space:
            print(f"Querying index with ef_search = {ef_search} ...")
            profiler.set_index_search_params({"ef_search": ef_search})
            query_res = profiler.do_query(return_verbose=True, return_stats=True)
            query_res.pop("query_results")  # track per-query recalls and latencies
            results.append(
                {
                    "graph_degree": graph_degree,
                    "ef_construct": ef_construct,
                    "alpha": alpha,
                    "ef_search": ef_search,
                    **query_res,
                    **build_results,
                }
            )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_filtered_diskann_selectivity_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    graph_degree_space: list[int] = [64, 128, 256],
    ef_construct_space: list[int] = [64, 128, 256],
    alpha_space: list[float] = [1.2],
    ef_search_space: list[int] = [64, 128, 256, 512, 1024],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    output_dir: str | Path = "output/selectivity/filtered_diskann",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for graph_degree, ef_construct, alpha in product(
        graph_degree_space, ef_construct_space, alpha_space
    ):
        task_name = f"r{graph_degree}ef{ef_construct}a{alpha}"
        command = batch_profiler.build_command(
            module="benchmark.selectivity.filtered_diskann",
            func="exp_filtered_diskann_selectivity",
            output_path=str(results_dir / f"{task_name}.json"),
            graph_degree=graph_degree,
            ef_construct=ef_construct,
            alpha=alpha,
            ef_search_space=ef_search_space,
            construct_threads=construct_threads,
            dataset_cache_dir=dataset_cache_dir,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
