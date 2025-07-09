import json
import time
from itertools import product
from pathlib import Path

import fire
import numpy as np
from tqdm import tqdm

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.utils import compute_qualified_labels
from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import recall
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss as CuratorIndex


class CuratorProfilerForComplexPredicate(IndexProfiler):
    """Profiler that uses search_with_bitmap_filter for detailed profiling"""

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.dataset: ComplexPredicateDataset | None = None

    def set_dataset(self, dataset: ComplexPredicateDataset):
        self.dataset = dataset
        return self

    def do_query_with_profiling(
        self,
        k: int = 10,
        return_verbose: bool = False,
        templates: list[str] | None = None,
    ) -> dict[str, dict]:
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"
        assert isinstance(self.index, CuratorIndex), "Index must be CuratorIndex"

        # Enable profiling on the underlying FAISS index
        self.index.index.set_profiling_enabled(True)

        template_to_results: dict[str, dict] = dict()

        pbar = tqdm(
            total=self.dataset.num_filters,
            desc="Querying index with profiling",
        )
        sorted_templates = sorted(self.dataset.template_to_filters.keys())

        for template in sorted_templates:
            filters = self.dataset.template_to_filters[template]

            if templates is not None and template not in templates:
                print(f"Skipping template {template}")
                pbar.update(len(filters))
                continue

            query_latencies = list()
            query_results = list()
            query_recalls = list()

            # Profiling data collectors
            preproc_times = list()
            sort_times = list()
            build_temp_index_times = list()
            search_times = list()
            qualified_labels_counts = list()
            temp_nodes_counts = list()

            for filter_predicate in filters:
                # Compute qualified labels for this filter
                qualified_labels = compute_qualified_labels(
                    filter_predicate, self.dataset.train_mds
                )

                query_results_filter = list()

                for vec in self.dataset.test_vecs:
                    query_start = time.time()

                    # Use search_with_bitmap_filter directly
                    ids = self.index.search_with_bitmap_filter(vec, k, qualified_labels)

                    query_end = time.time()
                    query_latencies.append(query_end - query_start)
                    query_results_filter.append(ids)

                    # Collect profiling data from the last search
                    profile_data = self.index.get_last_search_profile()
                    preproc_times.append(profile_data["preproc_time_ms"])
                    sort_times.append(profile_data["sort_time_ms"])
                    build_temp_index_times.append(
                        profile_data["build_temp_index_time_ms"]
                    )
                    search_times.append(profile_data["search_time_ms"])
                    qualified_labels_counts.append(
                        profile_data["qualified_labels_count"]
                    )
                    temp_nodes_counts.append(profile_data["temp_nodes_count"])

                ground_truth = self.dataset.filter_to_ground_truth[filter_predicate]
                query_recalls.extend(
                    [
                        recall([res], [gt])
                        for res, gt in zip(query_results_filter, ground_truth)
                    ]
                )

                query_results.extend(query_results_filter)
                pbar.update()

            recall_at_k = np.mean(query_recalls).item()

            template_results = {
                "recall_at_k": recall_at_k,
                "query_latencies": query_latencies,
                # Profiling metrics
                "preproc_times_ms": preproc_times,
                "sort_times_ms": sort_times,
                "build_temp_index_times_ms": build_temp_index_times,
                "search_times_ms": search_times,
                "qualified_labels_counts": qualified_labels_counts,
                "temp_nodes_counts": temp_nodes_counts,
            }

            template_to_results[template] = self._compute_metrics(template_results)

            if return_verbose:
                template_results.update(
                    {
                        "query_recalls": query_recalls,
                        "query_results": query_results,
                    }
                )

        # Disable profiling
        self.index.index.set_profiling_enabled(False)

        return template_to_results


def exp_curator_complex_predicate_profiling(
    output_path: str,
    nlist: int = 32,
    max_sl_size: int = 256,
    search_ef_space: list[int] = [32, 64, 128, 256, 512],
    beam_size_space: list[int] = [4],
    variance_boost_space: list[float] = [0.4],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
    use_optimized_search: bool = False,
):
    """
    Run curator complex predicate profiling.

    Parameters
    ----------
    use_optimized_search : bool
        If True, enables optimized search mode that skips preprocessing and sorting phases.
    """
    profiler = CuratorProfilerForComplexPredicate()

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

    # Note: We don't pre-build bitmaps since we're testing the bitmap filtering directly
    # But if optimized search is enabled, we need to build sorted internal vector IDs
    if use_optimized_search:
        print("Building sorted internal vector IDs for optimized search ...")
        begin_build_optimized = time.time()
        for __, filters in dataset.template_to_filters.items():
            for filter in filters:
                profiler.index.build_filter_optimized_vids(filter, dataset.train_mds)
        build_results["optimized_vids_build_time_s"] = (
            time.time() - begin_build_optimized
        )
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
        per_template_results = profiler.do_query_with_profiling()
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


if __name__ == "__main__":
    """
    python -m benchmark.complex_predicate.baselines.curator_prof \
        exp_curator_complex_predicate_profiling \
            --output_path test_curator_prof.json \
            --nlist 32 \
            --max_sl_size 256 \
            --search_ef_space "[32, 64, 128, 256, 512]" \
            --beam_size_space "[4]" \
            --variance_boost_space "[0.4]" \
            --dataset_key yfcc100m \
            --test_size 0.01 \
            --templates '["AND {0} {1}", "OR {0} {1}"]' \
            --n_filters_per_template 10 \
            --n_queries_per_filter 100 \
            --gt_cache_dir data/ground_truth/complex_predicate
    """
    fire.Fire()
