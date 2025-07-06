import json
from itertools import product
from pathlib import Path

import fire

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.profiler import IndexProfilerForComplexPredicate
from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset
from indexes.parlay_ivf import ParlayIVF


def write_dataset(
    dataset_dir: str | Path = "data/parlay_ivf/complex_predicate",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    overwrite: bool = False,
):
    dataset_dir = Path(dataset_dir) / f"{dataset_key}_test{test_size}"
    # the indexed vectors and access lists of complex predicate dataset is the same
    # as the overall results dataset, so we can use the same dataset class
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    ParlayIVF(dataset_dir).write_dataset(
        dataset.train_vecs, dataset.train_mds, overwrite=overwrite
    )


def exp_parlay_ivf_complex_predicate(
    output_path: str,
    dataset_dir: str | Path = "data/parlay_ivf/complex_predicate",
    ivf_cluster_size: int = 500,
    graph_degree: int = 16,
    ivf_max_iter: int = 10,
    ivf_search_radius_space: list[int] = [500, 1000, 2000],
    graph_search_L_space: list[int] = [32, 64, 128],
    construct_threads: int = 4,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
):
    profiler = IndexProfilerForComplexPredicate()

    dataset_dir = Path(dataset_dir) / f"{dataset_key}_test{test_size}"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory {dataset_dir} not found. "
            f"Please write the dataset first."
        )

    print(f"Loading dataset {dataset_key} ...")
    dataset = ComplexPredicateDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        templates=templates,
        n_filters_per_template=n_filters_per_template,
        n_queries_per_filter=n_queries_per_filter,
        gt_cache_dir=gt_cache_dir,
    )
    profiler.set_dataset(dataset)

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
            "cutoff": 1000,
            "max_iter": ivf_max_iter,
            "weight_classes": [100000, 400000],
            "max_degrees": [graph_degree] * 3,
            "bitvector_cutoff": 10000,
            "build_threads": construct_threads,
        },
        search_params={
            "target_points": ivf_search_radius_space[0],
            "tiny_cutoff": 1000,
            "beam_widths": [graph_search_L_space[0]] * 3,
            "search_limits": [100000, 400000, 3000000],
            "search_threads": 1,
        },
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=False,
        batch_insert=True,
    )

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
        per_template_results = profiler.do_query(templates=["AND {0} {1}"])
        results.append(
            {
                "ivf_cluster_size": ivf_cluster_size,
                "graph_degree": graph_degree,
                "ivf_max_iter": ivf_max_iter,
                "ivf_search_radius": ivf_search_radius,
                "graph_search_L": graph_search_L,
                "per_template_results": per_template_results,
                **build_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_parlay_ivf_complex_predicate_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    dataset_dir: str | Path = "data/parlay_ivf/complex_predicate",
    ivf_cluster_size_space: list[int] = [100, 500, 1000],
    graph_degree_space: list[int] = [8, 12, 16],
    ivf_max_iter_space: list[int] = [10],
    ivf_search_radius_space: list[int] = [500, 1000, 2000],
    graph_search_L_space: list[int] = [32, 64, 128],
    construct_threads: int = 4,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
    output_dir: str | Path = "output/complex_predicate/parlay_ivf",
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
        task_name = f"cluster{ivf_cluster_size}_degree{graph_degree}_iter{ivf_max_iter}"
        command = batch_profiler.build_command(
            module="benchmark.complex_predicate.baselines.parlay_ivf",
            func="exp_parlay_ivf_complex_predicate",
            output_path=str(results_dir / f"{task_name}.json"),
            dataset_dir=dataset_dir,
            ivf_cluster_size=ivf_cluster_size,
            graph_degree=graph_degree,
            ivf_max_iter=ivf_max_iter,
            ivf_search_radius_space=ivf_search_radius_space,
            graph_search_L_space=graph_search_L_space,
            construct_threads=construct_threads,
            dataset_key=dataset_key,
            test_size=test_size,
            templates=templates,
            n_filters_per_template=n_filters_per_template,
            n_queries_per_filter=n_queries_per_filter,
            gt_cache_dir=gt_cache_dir,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
