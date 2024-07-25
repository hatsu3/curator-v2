import json
from itertools import product
from pathlib import Path

import fire

from benchmark.complex_predicate.dataset import (
    ComplexPredicateDataset,
    PerPredicateDataset,
)
from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, IndexProfiler
from indexes.hnsw_sepidx_hnswlib import HNSWMultiTenantSepIndexHnswlib as PerLabelHNSW


def exp_per_predicate_hnsw_complex_predicate(
    output_path: str,
    construction_ef: int = 8,
    m: int = 8,
    search_ef_space: list[int] = [16, 32, 64, 128],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    dataset = ComplexPredicateDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        templates=templates,
        n_filters_per_template=n_filters_per_template,
        n_queries_per_filter=n_queries_per_filter,
        gt_cache_dir=gt_cache_dir,
    )
    dataset = PerPredicateDataset.from_complex_predicate_dataset(dataset)
    template_splits = {
        template: dataset.get_template_split(template) for template in templates
    }

    print(f"Building index with construction_ef = {construction_ef}, m = {m} ...")
    index_config = IndexConfig(
        index_cls=PerLabelHNSW,
        index_params={
            "construction_ef": construction_ef,
            "m": m,
            "max_elements": 1000000,
        },
        search_params={
            "search_ef": search_ef_space[0],
        },
    )

    results = list()
    per_template_results = dict()

    for template in templates:
        profiler.set_dataset(template_splits[template])

        print(
            f"Building index for template {template} with construction_ef = {construction_ef}, m = {m} ..."
        )
        profiler.do_build(
            index_config=index_config,
            do_train=False,
            batch_insert=False,
        )

        for search_ef in search_ef_space:
            print(f"Querying index with search_ef = {search_ef} ...")
            profiler.set_index_search_params({"search_ef": search_ef})
            query_results = profiler.do_query()

            if search_ef not in per_template_results:
                per_template_results[search_ef] = dict()
            per_template_results[search_ef][template] = query_results

    for search_ef in search_ef_space:
        results.append(
            {
                "construction_ef": construction_ef,
                "m": m,
                "search_ef": search_ef,
                "per_template_results": per_template_results[search_ef],
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_per_predicate_hnsw_complex_predicate_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    construction_ef_space: list[int] = [8, 16, 32],
    m_space: list[int] = [8, 16, 32],
    search_ef_space: list[int] = [16, 32, 64, 128],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
    output_dir: str | Path = "output/complex_predicate/per_predicate_hnsw",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for construction_ef, m in product(construction_ef_space, m_space):
        task_name = f"ef{construction_ef}_m{m}"
        command = batch_profiler.build_command(
            module="benchmark.complex_predicate.per_predicate_hnsw",
            func="exp_per_predicate_hnsw_complex_predicate",
            output_path=str(results_dir / f"{task_name}.json"),
            construction_ef=construction_ef,
            m=m,
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
