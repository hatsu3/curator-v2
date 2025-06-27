import json
import shutil
from itertools import product
from pathlib import Path

import fire
import numpy as np

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.profiler import IndexProfilerForComplexPredicate
from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler
from indexes.acorn import ACORN


def load_latencies_from_binary(latencies_path: str | Path) -> list[float]:
    with Path(latencies_path).open("rb") as file:
        num_queries = int.from_bytes(file.read(8), byteorder="little", signed=False)
        latencies = np.fromfile(file, dtype=np.float64, count=num_queries)
    return latencies.tolist()


def exp_acorn_complex_predicate(
    output_path: str,
    dataset_dir: str,
    index_dir: str,
    m: int = 32,
    gamma: int = 1,
    m_beta: int = 64,
    search_ef_space: list[int] = [16, 32, 64, 128, 256, 512],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
):
    profiler = IndexProfilerForComplexPredicate()

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

    print(f"Building index with m = {m}, gamma = {gamma}, m_beta = {m_beta} ...")
    index_config = IndexConfig(
        index_cls=ACORN,
        index_params={
            "dataset_dir": dataset_dir,
            "index_dir": index_dir,
            "m": m,
            "gamma": gamma,
            "m_beta": m_beta,
        },
        search_params={
            "search_ef": search_ef_space[0],
        },
    )

    if not (Path(index_dir) / "index.bin").exists():
        build_results = profiler.do_build(
            index_config=index_config,
            do_train=False,
            batch_insert=True,
        )
    else:
        print(f"Loading index from {index_dir} ...")
        profiler.set_index(index_config.index_cls(**index_config.index_params), False)
        build_results = dict()

    results = list()
    for search_ef in search_ef_space:
        print(f"Querying index with search_ef = {search_ef} ...")
        profiler.set_index_search_params({"search_ef": search_ef})
        per_template_results = profiler.do_batch_query()

        latencies = load_latencies_from_binary(Path(index_dir) / f"search_latency.bin")
        n_query_per_template = n_filters_per_template * n_queries_per_filter
        assert len(latencies) == len(templates) * n_query_per_template

        for i, template in enumerate(sorted(templates)):
            template_latencies = latencies[
                i * n_query_per_template : (i + 1) * n_query_per_template
            ]
            per_template_results[template]["query_lat_avg"] = (
                np.mean(template_latencies) / 1000
            )

        results.append(
            {
                "m": m,
                "gamma": gamma,
                "m_beta": m_beta,
                "search_ef": search_ef,
                "per_template_results": per_template_results,
                **build_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))

    # print(f"Deleting index at {index_dir} ...")
    # shutil.rmtree(index_dir)


def exp_acorn_complex_predicate_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    m_space: list[int] = [16, 32, 64],
    gamma_space: list[float] = [1, 10, 20],
    m_beta_multiplier_space: list[int] = [1, 2, 4],
    search_ef_space: list[int] = [8, 16, 32, 64, 128],
    dataset_dir: str = "data/acorn/complex_predicate/yfcc100m_test0.01",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
    output_dir: str | Path = "output/complex_predicate/acorn",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    index_dir = output_dir / "index"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for m, gamma, m_beta_multiplier in product(
        m_space, gamma_space, m_beta_multiplier_space
    ):
        if m_beta_multiplier > gamma:
            continue

        m_beta = m * m_beta_multiplier

        task_name = f"m{m}_gamma{gamma}_m_beta{m_beta}"
        output_path = results_dir / f"{task_name}.json"
        if output_path.exists():
            print(f"Skipping {output_path} ...")
            continue

        command = batch_profiler.build_command(
            module="benchmark.complex_predicate.acorn",
            func="exp_acorn_complex_predicate",
            output_path=str(output_path),
            dataset_dir=dataset_dir,
            index_dir=str(index_dir / task_name),
            m=m,
            gamma=gamma,
            m_beta=m_beta,
            search_ef_space=search_ef_space,
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
