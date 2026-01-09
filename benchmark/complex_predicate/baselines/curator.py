import json
import time
from itertools import product
from pathlib import Path

import fire

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.profiler import IndexProfilerForComplexPredicate
from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler
from indexes.curator import Curator as CuratorIndex


def exp_curator_complex_predicate(
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
    use_optimized_search: bool = True,
):
    """
    Run curator complex predicate benchmark.

    Parameters
    ----------
    use_optimized_search : bool
        If True, enables optimized search mode that skips preprocessing and sorting phases
        when sorted internal vector IDs are available.
    """
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
            "bf_capacity": dataset.num_filters,
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

    # Set optimized search mode
    assert isinstance(profiler.index, CuratorIndex)
    profiler.index.index.set_optimized_search_enabled(use_optimized_search)
    print(f"Optimized search mode: {'enabled' if use_optimized_search else 'disabled'}")

    print("Building bitmaps for filters ...")
    begin_build_bitmap = time.time()
    for __, filters in dataset.template_to_filters.items():
        for filter in filters:
            if use_optimized_search:
                profiler.index.build_filter_sorted_vids(filter, dataset.train_mds)
            else:
                profiler.index.build_filter_bitmap(filter, dataset.train_mds)
    build_results["bitmap_build_time_s"] = time.time() - begin_build_bitmap
    build_results["use_optimized_search"] = use_optimized_search

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


def exp_curator_complex_predicate_param_sweep(
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
    output_dir: str | Path = "output/complex_predicate/curator",
    use_optimized_search: bool = False,
):
    """
    Parameter sweep with optional optimized search mode.

    Parameters
    ----------
    use_optimized_search : bool
        If True, enables optimized search mode for all parameter combinations.
    """
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for nlist, max_sl_size in product(nlist_space, max_sl_size_space):
        task_name = f"nlist{nlist}_sl{max_sl_size}"
        command = batch_profiler.build_command(
            module="benchmark.complex_predicate.baselines.curator",
            func="exp_curator_complex_predicate",
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
            use_optimized_search=use_optimized_search,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
