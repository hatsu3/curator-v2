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
    dataset_cache_path: str | Path,
    graph_degree: int = 32,
    ef_construct: int = 32,
    alpha: float = 1.2,
    ef_search_space: list[int] = [32, 64, 128],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    num_runs: int = 1,
    return_verbose: bool = False,
    index_dir: str | Path | None = None,
    skip_build: bool = False,
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    assert Path(
        dataset_cache_path
    ).exists(), f"Dataset cache path {dataset_cache_path} does not exist"
    dataset = Dataset.from_dataset_key(
        dataset_key, test_size=test_size, cache_path=dataset_cache_path
    )
    profiler.set_dataset(dataset)

    # Determine index directory and caching strategy
    if index_dir is not None:
        index_dir = Path(index_dir)
        cache_index = True
        use_temp_dir = False

        # Check if we should skip building
        if skip_build:
            if not index_dir.exists() or not any(index_dir.iterdir()):
                raise ValueError(
                    f"skip_build=True but index directory {index_dir} does not exist or is empty. "
                    "Please build the index first or set skip_build=False."
                )
            print(f"Skipping index construction, loading from {index_dir} ...")
        else:
            # Create index directory if it doesn't exist
            index_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"Building index with graph_degree = {graph_degree}, "
                f"ef_construct = {ef_construct}, alpha = {alpha} ..."
            )
    else:
        # Use temporary directory (original behavior)
        assert not skip_build, "Can't skip build when index_dir is not provided"
        cache_index = False
        use_temp_dir = True
        print(
            f"Building index with graph_degree = {graph_degree}, "
            f"ef_construct = {ef_construct}, alpha = {alpha} ..."
        )

    def run_experiment(experiment_index_dir):
        index_config = IndexConfig(
            index_cls=FilteredDiskANN,
            index_params={
                "index_dir": experiment_index_dir,
                "d": dataset.dim,
                "ef_construct": ef_construct,
                "graph_degree": graph_degree,
                "alpha": alpha,
                "filter_ef_construct": ef_construct,
                "construct_threads": construct_threads,
                "search_threads": 1,
                "cache_index": cache_index,
            },
            search_params={
                "ef_search": ef_search_space[0],
            },
        )

        if skip_build:
            # Only load the index, don't build
            # Load index size from file
            index_size_file = Path(experiment_index_dir) / "preds_index_size.txt"
            if index_size_file.exists():
                with open(index_size_file, "r") as f:
                    index_size_kb = int(f.read().strip())
            else:
                raise FileNotFoundError(f"Index size file {index_size_file} not found")

            # Load build time from file
            build_time_file = Path(experiment_index_dir) / "build_results.json"
            if build_time_file.exists():
                with open(build_time_file, "r") as f:
                    build_data = json.load(f)
                    batch_insert_latency = build_data["build_time"]
            else:
                raise FileNotFoundError(
                    f"Build results file {build_time_file} not found"
                )

            build_results = {
                "index_size_kb": index_size_kb,
                "batch_insert_latency": batch_insert_latency,
            }

            profiler.set_index(
                index_config.index_cls(**index_config.index_params), track_stats=False
            )
        else:
            # Build the index
            build_results = profiler.do_build(
                index_config=index_config,
                do_train=False,
                batch_insert=True,
            )

        results = list()
        for ef_search in ef_search_space:
            print(f"Querying index with ef_search = {ef_search} ...")
            profiler.set_index_search_params({"ef_search": ef_search})
            query_results = profiler.do_query(
                batch_query=False, num_runs=num_runs, return_verbose=return_verbose
            )
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
        return results

    # Execute experiment
    if use_temp_dir:
        with tempfile.TemporaryDirectory() as temp_index_dir:
            results = run_experiment(temp_index_dir)
    else:
        results = run_experiment(str(index_dir))

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
