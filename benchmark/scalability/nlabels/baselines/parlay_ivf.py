import json
from itertools import product
from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset, IndexProfiler
from benchmark.scalability.dataset import ScalabilityDataset
from indexes.parlay_ivf import ParlayIVF


def write_dataset(
    output_dir: str | Path = "data/parlay_ivf/scalability",
    n_labels: int = 10000,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/scalability/random_yfcc100m",
    seed: int = 42,
    overwrite: bool = False,
):
    dataset = ScalabilityDataset.load(dataset_cache_dir)
    assert dataset.n_labels >= n_labels
    split = dataset.get_random_split(n_labels, seed=seed, remap_labels=True)

    dataset_dir = (
        Path(output_dir) / f"{dataset_key}_test{test_size}" / f"n{n_labels}_seed{seed}"
    )
    ParlayIVF(dataset_dir).write_dataset(
        split.train_vecs, split.train_mds, overwrite=overwrite
    )


def exp_parlay_ivf_scalability(
    output_path: str,
    dataset_dir: str | Path = "data/parlay_ivf/scalability",
    n_labels: int = 10000,
    cutoff: int = 10000,
    ivf_cluster_size: int = 5000,
    graph_degree: int = 16,
    ivf_max_iter: int = 10,
    ivf_search_radius_space: list[int] = [500, 1000, 2000],
    graph_search_L_space: list[int] = [32, 64, 128],
    construct_threads: int = 8,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/scalability/random_yfcc100m",
    seed: int = 42,
):
    profiler = IndexProfiler()

    dataset_dir = (
        Path(dataset_dir) / f"{dataset_key}_test{test_size}" / f"n{n_labels}_seed{seed}"
    )
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
        f"Building index with cutoff = {cutoff}, cluster_size = {ivf_cluster_size}, "
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
        index_config,
        do_train=False,
        batch_insert=True,
        with_labels=False,
    )

    dataset = ScalabilityDataset.load(dataset_cache_dir)
    assert dataset.n_labels >= n_labels
    split = dataset.get_random_split(n_labels, seed=seed, remap_labels=True)
    profiler.set_dataset(split)

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
        query_results = profiler.do_query(batch_query=True)
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


def exp_parlay_ivf_scalability_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    n_labels_n_steps: int = 10,
    n_labels_max: int = 10000,
    cutoff: int = 10000,
    ivf_cluster_size_space: list[int] = [100, 500, 1000],
    graph_degree_space: list[int] = [8, 12, 16],
    ivf_max_iter_space: list[int] = [10],
    ivf_search_radius_space: list[int] = [500, 1000, 2000],
    graph_search_L_space: list[int] = [32, 64, 128],
    construct_threads: int = 1,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/scalability/random_yfcc100m",
    seed: int = 42,
    output_dir: str | Path = "output/scalability/parlay_ivf",
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

    for n_labels in n_labels_space:
        dataset_dir = (
            Path("data/parlay_ivf/scalability")
            / f"{dataset_key}_test{test_size}"
            / f"n{n_labels}_seed{seed}"
        )
        if not dataset_dir.is_dir():
            print(f"Writing dataset to {dataset_dir} ...")
            write_dataset(
                output_dir="data/parlay_ivf/scalability",
                n_labels=n_labels,
                dataset_key=dataset_key,
                test_size=test_size,
                dataset_cache_dir=dataset_cache_dir,
                seed=seed,
                overwrite=False,
            )
        else:
            print(f"Dataset directory {dataset_dir} already exists. Skipping ...")

    for n_labels, ivf_cluster_size, graph_degree, ivf_max_iter in product(
        n_labels_space, ivf_cluster_size_space, graph_degree_space, ivf_max_iter_space
    ):
        task_name = f"n_labels{n_labels}_nlist{ivf_cluster_size}_r{graph_degree}"
        command = batch_profiler.build_command(
            module="benchmark.scalability.parlay_ivf",
            func="exp_parlay_ivf_scalability",
            output_path=str(results_dir / f"{task_name}.csv"),
            dataset_dir="data/parlay_ivf/scalability",
            n_labels=n_labels,
            cutoff=cutoff,
            ivf_cluster_size=ivf_cluster_size,
            graph_degree=graph_degree,
            ivf_max_iter=ivf_max_iter,
            ivf_search_radius_space=ivf_search_radius_space,
            graph_search_L_space=graph_search_L_space,
            construct_threads=construct_threads,
            dataset_key=dataset_key,
            test_size=test_size,
            dataset_cache_dir=dataset_cache_dir,
            seed=seed,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
