import json
from itertools import product
from pathlib import Path

import fire

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset, IndexProfiler
from benchmark.selectivity.dataset import SelectivityDataset
from indexes.parlay_ivf import ParlayIVF


def write_dataset(
    output_dir: str | Path = "data/parlay_ivf/selectivity",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    overwrite: bool = False,
):
    dataset_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    dataset = SelectivityDataset.load(dataset_cache_dir)
    ParlayIVF(dataset_dir).write_dataset(
        dataset.train_vecs, dataset.train_mds, overwrite=overwrite
    )


def exp_parlay_ivf_selectivity(
    output_path: str,
    dataset_dir: str | Path = "data/parlay_ivf/selectivity",
    ivf_cluster_size: int = 5000,
    graph_degree: int = 16,
    ivf_max_iter: int = 10,
    ivf_search_radius_space: list[int] = [500, 1000, 2000],
    graph_search_L_space: list[int] = [32, 64, 128],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
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
            "cutoff": 10000,
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

    dataset = SelectivityDataset.load(dataset_cache_dir)
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
        query_res = profiler.do_query(return_verbose=True, return_stats=True)
        query_res.pop("query_results")  # track per-query recalls and latencies
        results.append(
            {
                "ivf_cluster_size": ivf_cluster_size,
                "graph_degree": graph_degree,
                "ivf_max_iter": ivf_max_iter,
                "ivf_search_radius": ivf_search_radius,
                "graph_search_L": graph_search_L,
                **query_res,
                **build_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_parlay_ivf_selectivity_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    ivf_cluster_size_space: list[int] = [100, 500, 1000],
    graph_degree_space: list[int] = [8, 12, 16],
    ivf_max_iter_space: list[int] = [10],
    ivf_search_radius_space: list[int] = [500, 1000, 2000],
    graph_search_L_space: list[int] = [32, 64, 128],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    output_dir: str | Path = "output/selectivity/parlay_ivf",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for ivf_cluster_size, graph_degree, ivf_max_iter in product(
        ivf_cluster_size_space, graph_degree_space, ivf_max_iter_space
    ):
        task_name = f"c{ivf_cluster_size}r{graph_degree}i{ivf_max_iter}"
        command = batch_profiler.build_command(
            module="benchmark.selectivity.parlay_ivf",
            func="exp_parlay_ivf_selectivity",
            output_path=str(results_dir / f"{task_name}.json"),
            ivf_cluster_size=ivf_cluster_size,
            graph_degree=graph_degree,
            ivf_max_iter=ivf_max_iter,
            ivf_search_radius_space=ivf_search_radius_space,
            graph_search_L_space=graph_search_L_space,
            construct_threads=construct_threads,
            dataset_key=dataset_key,
            test_size=test_size,
            dataset_cache_dir=dataset_cache_dir,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
