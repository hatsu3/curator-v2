import json
import tempfile
from itertools import product
from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset, IndexProfiler
from indexes.filtered_diskann import FilteredDiskANN


def exp_filtered_diskann(
    output_path: str,
    graph_degree: int = 32,
    ef_construct: int = 32,
    alpha: float = 1.2,
    ef_search_space: list[int] = [32, 64, 128],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    num_runs: int = 1,
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
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
                "ef_search": ef_search_space[0],
            },
        )
        build_results = profiler.do_build(
            index_config=index_config,
            do_train=False,
            batch_insert=True,
        )

        results = list()
        for ef_search in ef_search_space:
            print(f"Querying index with ef_search = {ef_search} ...")
            profiler.set_index_search_params({"ef_search": ef_search})
            query_results = profiler.do_query(batch_query=False, num_runs=num_runs)
            results.append(
                {
                    "graph_degree": graph_degree,
                    "ef_construct": ef_construct,
                    "alpha": alpha,
                    "ef_search": ef_search,
                    **build_results,
                    **query_results,
                }
            )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def exp_filtered_diskann_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    graph_degree_space: list[int] = [64, 128, 256],
    ef_construct_space: list[int] = [64, 128, 256],
    alpha_space: list[float] = [1.2],
    ef_search_space: list[int] = [64, 128, 256, 512, 1024],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_dir: str | Path = "output/overall_results/filtered_diskann",
):
    params = vars()
    cpu_groups = list(map(str, range(cpu_range[0], cpu_range[1] + 1)))

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
            module="benchmark.overall_results.filtered_diskann",
            func="exp_filtered_diskann",
            output_path=str(results_dir / f"{task_name}.csv"),
            graph_degree=graph_degree,
            ef_construct=ef_construct,
            alpha=alpha,
            ef_search_space=ef_search_space,
            construct_threads=construct_threads,
            dataset_key=dataset_key,
            test_size=test_size,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
