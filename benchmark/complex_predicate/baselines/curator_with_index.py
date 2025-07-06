import json
import time
from itertools import product
from pathlib import Path

import fire

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.profiler import IndexProfilerForComplexPredicate
from benchmark.complex_predicate.utils import compute_qualified_labels
from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler
from benchmark.utils import get_memory_usage
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss as CuratorIndex


def exp_curator_with_index_complex_predicate(
    output_path: str,
    nlist: int = 16,
    max_sl_size: int = 256,
    search_ef_space: list[int] = [32, 64, 128, 256, 512],
    beam_size_space: list[int] = [1, 2, 4, 8],
    variance_boost_space: list[float] = [0.4],
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

    print(f"Building index with nlist = {nlist}, max_sl_size = {max_sl_size} ...")
    index_config = IndexConfig(
        index_cls=CuratorIndex,
        index_params={
            "d": dataset.dim,
            "nlist": nlist,
            "max_sl_size": max_sl_size,
            "max_leaf_size": max_sl_size,
            "clus_niter": 20,
            "bf_capacity": 1000,
            "bf_error_rate": 0.01,
        },
        search_params={
            "search_ef": search_ef_space[0],
            "beam_size": beam_size_space[0],
            "variance_boost": variance_boost_space[0],
        },
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=False,
    )

    print("Indexing filters ...")
    mem_usage_before = get_memory_usage()
    filter_index_times = []
    total_filters = sum(
        len(filters) for filters in dataset.template_to_filters.values()
    )

    for template, filters in dataset.template_to_filters.items():
        for i, filter_predicate in enumerate(filters):
            assert isinstance(profiler.index, CuratorIndex)

            # Inline index_filter function and measure build_index_for_filter timing
            qualified_labels = compute_qualified_labels(
                filter_predicate, dataset.train_mds
            )

            # Measure only the build_index_for_filter latency
            start_time = time.time()
            profiler.index.index.build_index_for_filter(qualified_labels, filter_predicate)  # type: ignore
            end_time = time.time()

            filter_index_time = end_time - start_time
            filter_index_times.append(filter_index_time)

            # Set filter label mapping
            profiler.index.filter_to_label[filter_predicate] = (
                profiler.index.index.get_filter_label(filter_predicate)
            )

            if (i + 1) % 10 == 0 or i == len(filters) - 1:
                print(f"  {template}: Indexed {i + 1}/{len(filters)} filters")

    mem_usage_after = get_memory_usage()
    build_results["filter_index_size_kb"] = mem_usage_after - mem_usage_before
    build_results["filter_index_times"] = filter_index_times
    build_results["total_filters"] = total_filters

    print(f"Indexed {total_filters} filters total")
    print(
        f"Filter indexing memory overhead: {build_results['filter_index_size_kb']:.2f} KB"
    )
    print(
        f"Average filter index time: {sum(filter_index_times) / len(filter_index_times):.4f} seconds"
    )

    results = list()
    for search_ef, beam_size, variance_boost in product(
        search_ef_space, beam_size_space, variance_boost_space
    ):
        print(
            f"Querying index with search_ef = {search_ef}, beam_size = {beam_size}, "
            f"variance_boost = {variance_boost} ..."
        )
        profiler.set_index_search_params(
            {
                "search_ef": search_ef,
                "beam_size": beam_size,
                "variance_boost": variance_boost,
            }
        )
        per_template_results = profiler.do_query()
        results.append(
            {
                "nlist": nlist,
                "max_sl_size": max_sl_size,
                "search_ef": search_ef,
                "beam_size": beam_size,
                "variance_boost": variance_boost,
                "per_template_results": per_template_results,
                **build_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_curator_with_index_complex_predicate_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    nlist_space: list[int] = [8, 16, 32],
    max_sl_size_space: list[int] = [64, 128, 256],
    search_ef_space: list[int] = [32, 64, 128, 256, 512],
    beam_size_space: list[int] = [1, 2, 4, 8],
    variance_boost_space: list[float] = [0.4],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
    output_dir: str | Path = "output/complex_predicate/curator_with_index",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for nlist, max_sl_size in product(nlist_space, max_sl_size_space):
        task_name = f"nlist{nlist}_sl{max_sl_size}"
        command = batch_profiler.build_command(
            module="benchmark.complex_predicate.baselines.curator_with_index",
            func="exp_curator_with_index_complex_predicate",
            output_path=str(results_dir / f"{task_name}.json"),
            nlist=nlist,
            max_sl_size=max_sl_size,
            search_ef_space=search_ef_space,
            beam_size_space=beam_size_space,
            variance_boost_space=variance_boost_space,
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
