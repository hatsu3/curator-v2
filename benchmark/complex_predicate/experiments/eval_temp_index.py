import json
import time
from pathlib import Path

import fire

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.profiler import IndexProfilerForComplexPredicate
from benchmark.complex_predicate.utils import compute_qualified_labels
from benchmark.config import IndexConfig
from benchmark.utils import get_memory_usage
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss as CuratorIndex


def exp_eval_temp_index(
    output_path: str,
    nlist: int = 32,
    max_sl_size: int = 256,
    search_ef_space: list[int] = [32, 64, 128, 256, 512],
    beam_size: int = 4,
    variance_boost: float = 0.4,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
):
    """Compare temp index caching vs direct indexing strategies."""

    print(f"Loading dataset {dataset_key} ...")
    dataset = ComplexPredicateDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        templates=templates,
        n_filters_per_template=n_filters_per_template,
        n_queries_per_filter=n_queries_per_filter,
        gt_cache_dir=gt_cache_dir,
    )

    results = list()

    # Test each strategy
    for use_temp_index_caching in [False, True]:
        index_strategy = "temp_caching" if use_temp_index_caching else "direct_indexing"

        print(f"\n=== Testing {index_strategy} strategy ===")

        profiler = IndexProfilerForComplexPredicate()
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
                "use_temp_index_caching": use_temp_index_caching,
            },
            search_params={
                "search_ef": search_ef_space[0],
                "beam_size": beam_size,
                "variance_boost": variance_boost,
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

        # Add temp index memory usage for temp caching strategy
        if use_temp_index_caching:
            if hasattr(profiler.index, "get_cached_temp_index_memory_usage"):
                build_results["temp_index_size_kb"] = (
                    profiler.index.get_cached_temp_index_memory_usage() / 1024  # type: ignore
                )
            else:
                build_results["temp_index_size_kb"] = 0
        else:
            build_results["temp_index_size_kb"] = 0

        print(f"Indexed {total_filters} filters total")
        print(
            f"Filter indexing memory overhead: {build_results['filter_index_size_kb']:.2f} KB"
        )
        print(
            f"Average filter index time: {sum(filter_index_times) / len(filter_index_times):.4f} seconds"
        )
        if use_temp_index_caching:
            temp_index_mb = build_results["temp_index_size_kb"] / 1024
            print(f"Temp index memory usage: {temp_index_mb:.2f} MB")

        # Test all parameter combinations
        for search_ef in search_ef_space:
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
                    "index_strategy": index_strategy,
                    "per_template_results": per_template_results,
                    **build_results,
                }
            )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("✓ Results saved successfully!")


if __name__ == "__main__":
    """
    python -m benchmark.complex_predicate.experiments.eval_temp_index \
        exp_eval_temp_index \
            --output_path eval_temp_index.json \
            --nlist 32 \
            --max_sl_size 256 \
            --search_ef_space '[32, 64, 128, 256, 512]' \
            --beam_size 4 \
            --variance_boost 0.4 \
            --dataset_key yfcc100m \
            --test_size 0.01 \
            --templates '["AND {0} {1}", "OR {0} {1}"]' \
            --n_filters_per_template 10 \
            --n_queries_per_filter 100
    """
    fire.Fire()
