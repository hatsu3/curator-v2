import json
import tempfile
from itertools import product
from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, IndexProfiler
from benchmark.scalability.nlabels.data.dataset import ScalabilityDataset
from indexes.filtered_diskann import FilteredDiskANN


def exp_filtered_diskann_scalability(
    output_path: str,
    n_labels: int,
    graph_degree: int = 32,
    ef_construct: int = 32,
    alpha: float = 1.2,
    ef_search_space: list[int] = [64, 128, 256, 512, 1024, 2048],
    construct_threads: int = 1,
    dataset_cache_dir: str = "data/scalability/random_yfcc100m",
    seed: int = 42,
):
    profiler = IndexProfiler()

    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = ScalabilityDataset.load(dataset_cache_dir)
    if dataset.n_labels < n_labels:
        raise ValueError(
            f"Dataset has fewer labels ({dataset.n_labels}) than specified ({n_labels})"
        )

    split = dataset.get_random_split(n_labels, seed=seed)
    profiler.set_dataset(split)

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
            query_results = profiler.do_query(batch_query=False)
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


def exp_filtered_diskann_scalability_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    n_labels_n_steps: int = 10,
    n_labels_max: int = 10000,
    graph_degree_space: list[int] = [64, 128, 256, 384, 512],
    ef_construct_space: list[int] = [64, 128, 256, 512, 1024],
    alpha_space: list[float] = [1.2],
    ef_search_space: list[int] = [64, 128, 256, 512, 1024, 2048],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/scalability/random_yfcc100m",
    seed: int = 42,
    output_dir: str | Path = "output/scalability/filtered_diskann",
):
    params = vars()
    cpu_groups = list(map(str, range(cpu_range[0], cpu_range[1] + 1)))

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    n_labels_min = n_labels_max // (2 ** (n_labels_n_steps - 1))
    n_labels_space = [n_labels_min * (2**i) for i in range(n_labels_n_steps)]

    for n_labels, graph_degree, ef_construct, alpha in product(
        n_labels_space, graph_degree_space, ef_construct_space, alpha_space
    ):
        task_name = f"n_labels{n_labels}_r{graph_degree}ef{ef_construct}a{alpha}"
        if (results_dir / f"{task_name}.csv").exists():
            print(f"Skipping finished task {task_name} ...")

        command = batch_profiler.build_command(
            module="benchmark.scalability.filtered_diskann",
            func="exp_filtered_diskann_scalability",
            output_path=str(results_dir / f"{task_name}.csv"),
            n_labels=n_labels,
            graph_degree=graph_degree,
            ef_construct=ef_construct,
            alpha=alpha,
            ef_search_space=ef_search_space,
            construct_threads=construct_threads,
            dataset_cache_dir=dataset_cache_dir,
            seed=seed,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
