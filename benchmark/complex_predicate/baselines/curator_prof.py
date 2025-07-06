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
from benchmark.profiler import BatchProfiler, IndexProfiler
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
                    profile_data = self.index.index.get_last_search_profile()
                    preproc_times.append(profile_data.preproc_time_ms)
                    sort_times.append(profile_data.sort_time_ms)
                    build_temp_index_times.append(profile_data.build_temp_index_time_ms)
                    search_times.append(profile_data.search_time_ms)
                    qualified_labels_counts.append(profile_data.qualified_labels_count)
                    temp_nodes_counts.append(profile_data.temp_nodes_count)

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
                # Summary statistics
                "mean_preproc_time_ms": np.mean(preproc_times).item(),
                "mean_sort_time_ms": np.mean(sort_times).item(),
                "mean_build_temp_index_time_ms": np.mean(build_temp_index_times).item(),
                "mean_search_time_ms": np.mean(search_times).item(),
                "mean_qualified_labels_count": np.mean(qualified_labels_counts).item(),
                "mean_temp_nodes_count": np.mean(temp_nodes_counts).item(),
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

    # Note: We don't pre-build bitmaps since we're testing the bitmap filtering directly

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


def exp_curator_complex_predicate_profiling_param_sweep(
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
    output_dir: str | Path = "output/complex_predicate/curator_prof",
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
            module="benchmark.complex_predicate.baselines.curator_prof",
            func="exp_curator_complex_predicate_profiling",
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
    """
    python -m benchmark.complex_predicate.baselines.curator_prof \
        exp_curator_complex_predicate_profiling \
            --output_path test_curator_prof.json \
            --nlist 16 \
            --max_sl_size 256 \
            --search_ef_space "[32, 64, 128, 256, 512]" \
            --beam_size_space "[4]" \
            --variance_boost_space "[0.4]" \
            --dataset_key yfcc100m \
            --test_size 0.01 \
            --templates '["NOT {0}", "AND {0} {1}", "OR {0} {1}"]' \
            --n_filters_per_template 10 \
            --n_queries_per_filter 100 \
            --gt_cache_dir data/ground_truth/complex_predicate
    """
    fire.Fire()
