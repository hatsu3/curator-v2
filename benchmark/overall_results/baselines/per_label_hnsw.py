import json
from itertools import product
from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset, IndexProfiler
from indexes.per_label_hnsw import PerLabelHNSW


def exp_per_label_hnsw(
    output_path: str,
    dataset_cache_path: str | Path,
    construction_ef: int = 32,
    m: int = 16,
    search_ef_space: list[int] = [32, 64, 128, 256],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    num_runs: int = 1,
    n_labels: int | None = None,
    seed: int = 42,
    return_verbose: bool = False,
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    assert Path(
        dataset_cache_path
    ).exists(), f"Dataset cache path {dataset_cache_path} does not exist"
    dataset = Dataset.from_dataset_key(
        dataset_key, test_size=test_size, cache_path=dataset_cache_path
    )
    if n_labels is not None:
        dataset = dataset.get_random_split(n_labels, seed=seed)
    profiler.set_dataset(dataset)

    print(f"Building index with construction_ef = {construction_ef}, m = {m} ...")
    index_config = IndexConfig(
        index_cls=PerLabelHNSW,
        index_params={
            "construction_ef": construction_ef,
            "m": m,
            "max_elements": dataset.largest_cardinality + 100,
        },
        search_params={
            "search_ef": search_ef_space[0],
        },
    )

    build_results = profiler.do_build(
        index_config=index_config,
        do_train=False,
        batch_insert=False,
    )

    results = list()
    for search_ef in search_ef_space:
        print(f"Querying index with search_ef = {search_ef} ...")
        profiler.set_index_search_params({"search_ef": search_ef})
        query_results = profiler.do_query(
            num_runs=num_runs, return_verbose=return_verbose
        )
        results.append(
            {
                "construction_ef": construction_ef,
                "m": m,
                "search_ef": search_ef,
                **build_results,
                **query_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def exp_per_label_hnsw_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    construction_ef_space: list[int] = [16, 32, 64, 128],
    m_space: list[int] = [8, 16, 32],
    search_ef_space: list[int] = [32, 64, 128, 256],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    n_labels: int | None = None,
    seed: int = 42,
    output_dir: str | Path = "output/overall_results/per_label_hnsw",
):
    params = vars()
    cpu_groups = list(map(str, range(cpu_range[0], cpu_range[1] + 1)))

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for construction_ef, m in product(construction_ef_space, m_space):
        task_name = f"ef{construction_ef}_m{m}"
        command = batch_profiler.build_command(
            module="benchmark.overall_results.per_label_hnsw",
            func="exp_per_label_hnsw",
            output_path=str(results_dir / f"{task_name}.csv"),
            construction_ef=construction_ef,
            m=m,
            search_ef_space=search_ef_space,
            dataset_key=dataset_key,
            test_size=test_size,
            n_labels=n_labels,
            seed=seed,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
