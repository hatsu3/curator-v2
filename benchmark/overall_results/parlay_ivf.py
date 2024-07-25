import json
from itertools import product
from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset, IndexProfiler
from indexes.parlay_ivf import ParlayIVF


def write_dataset(
    dataset_dir: str | Path = "data/parlay_ivf/overall_results",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    overwrite: bool = False,
):
    dataset_dir = Path(dataset_dir) / f"{dataset_key}_test{test_size}"
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    ParlayIVF(dataset_dir).write_dataset(
        dataset.train_vecs, dataset.train_mds, overwrite=overwrite
    )


def exp_parlay_ivf(
    output_path: str,
    dataset_dir: str | Path = "data/parlay_ivf/overall_results",
    cutoff: int = 10000,
    ivf_cluster_size: int = 5000,
    graph_degree: int = 16,
    ivf_max_iter: int = 10,
    ivf_search_radius_space: list[int] = [500, 1000, 2000],
    graph_search_L_space: list[int] = [32, 64, 128],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    num_runs: int = 1,
):
    profiler = IndexProfiler()

    dataset_dir = Path(dataset_dir) / f"{dataset_key}_test{test_size}"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory {dataset_dir} not found. "
            f"Please write the dataset first."
        )

    # hacking: ParlayIVF loads dataset from disk
    profiler.set_dataset(
        Dataset(
            train_vecs=None,  # type: ignore
            train_mds=None,  # type: ignore
            test_vecs=None,  # type: ignore
            test_mds=None,  # type: ignore
            ground_truth=None,  # type: ignore
            all_labels=None,  # type: ignore
        )
    )

    print(
        f"Building index with ivf_cluster_size = {ivf_cluster_size}, "
        f"graph_degree = {graph_degree}, ivf_max_iter = {ivf_max_iter} ..."
    )
    index_config = IndexConfig(
        index_cls=ParlayIVF,
        index_params={
            "dataset_dir": str(dataset_dir),
            "index_dir": "",
            "cluster_size": ivf_cluster_size,
            "cutoff": cutoff,
            "max_iter": ivf_max_iter,
            "weight_classes": [100000, 400000],
            "max_degrees": [graph_degree] * 3,
            "bitvector_cutoff": 10000,
            "build_threads": construct_threads,
        },
        search_params={
            "target_points": 1000,
            "tiny_cutoff": 1000,
            "beam_widths": [50] * 3,
            "search_limits": [100000, 400000, 3000000],
            "search_threads": 1,
        },
    )

    build_results = profiler.do_build(
        index_config=index_config,
        do_train=False,
        batch_insert=True,
        with_labels=False,
    )

    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    profiler.set_dataset(dataset)

    results = list()
    for ivf_search_radius, graph_search_L in product(
        ivf_search_radius_space, graph_search_L_space
    ):
        print(
            f"Querying index with ivf_search_radius = {ivf_search_radius}, "
            f"graph_search_L = {graph_search_L} ..."
        )
        profiler.set_index_search_params(
            {
                "target_points": ivf_search_radius,
                "beam_widths": [graph_search_L] * 3,
            }
        )
        query_results = profiler.do_query(batch_query=False, num_runs=num_runs)
        results.append(
            {
                "cutoff": cutoff,
                "ivf_cluster_size": ivf_cluster_size,
                "graph_degree": graph_degree,
                "ivf_max_iter": ivf_max_iter,
                "ivf_search_radius": ivf_search_radius,
                "graph_search_L": graph_search_L,
                **build_results,
                **query_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def exp_parlay_ivf_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    cutoff_space: list[int] = [1000, 5000, 10000],
    ivf_cluster_size_space: list[int] = [100, 500, 1000],
    graph_degree_space: list[int] = [8, 12, 16],
    ivf_max_iter_space: list[int] = [10],
    ivf_search_radius_space: list[int] = [500, 1000, 2000],
    graph_search_L_space: list[int] = [32, 64, 128],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_dir: str | Path = "output/overall_results/parlay_ivf",
):
    params = vars()
    cpu_groups = list(map(str, range(cpu_range[0], cpu_range[1] + 1)))

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for cutoff, ivf_cluster_size, graph_degree, ivf_max_iter in product(
        cutoff_space, ivf_cluster_size_space, graph_degree_space, ivf_max_iter_space
    ):
        task_name = f"cutoff{cutoff}c{ivf_cluster_size}r{graph_degree}i{ivf_max_iter}"
        command = batch_profiler.build_command(
            module="benchmark.overall_results.parlay_ivf",
            func="exp_parlay_ivf",
            output_path=str(results_dir / f"{task_name}.csv"),
            cutoff=cutoff,
            ivf_cluster_size=ivf_cluster_size,
            graph_degree=graph_degree,
            ivf_max_iter=ivf_max_iter,
            ivf_search_radius_space=ivf_search_radius_space,
            graph_search_L_space=graph_search_L_space,
            construct_threads=construct_threads,
            dataset_key=dataset_key,
            test_size=test_size,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
