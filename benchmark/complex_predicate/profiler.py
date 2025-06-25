import time

import numpy as np
from tqdm import tqdm

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.utils import compute_qualified_labels
from benchmark.profiler import IndexProfiler
from benchmark.utils import recall


class IndexProfilerForComplexPredicate(IndexProfiler):
    def __init__(self, seed: int = 42):
        super().__init__(seed)

        self.dataset: ComplexPredicateDataset | None = None

    def set_dataset(self, dataset: ComplexPredicateDataset):
        self.dataset = dataset
        return self

    def do_query(
        self,
        k: int = 10,
        return_stats: bool = False,
        return_verbose: bool = False,
        templates: list[str] | None = None,
        use_filter_labels: bool = False,
    ) -> dict[str, dict]:
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"

        template_to_results: dict[str, dict] = dict()

        pbar = tqdm(
            total=self.dataset.num_filters,
            desc="Querying index",
        )
        for template, filters in self.dataset.template_to_filters.items():
            if templates is not None and template not in templates:
                print(f"Skipping template {template}")
                pbar.update(len(filters))
                continue

            query_latencies = list()
            query_stats = list()
            query_results = list()
            query_recalls = list()

            for filter in filters:
                query_results_filter = list()

                if use_filter_labels:
                    # Use filter label with regular search interface
                    # First get the filter label for this filter
                    filter_label = self.index.get_filter_label(filter)  # type: ignore

                    for vec in self.dataset.test_vecs:
                        query_start = time.time()
                        # Use regular search with filter label as tenant ID
                        ids = self.index.search(vec, k=k, tid=filter_label)  # type: ignore
                        query_latencies.append(time.time() - query_start)
                        query_results_filter.append(ids)
                else:
                    # Use bitmap filter interface (existing behavior)
                    # Compute qualified labels for this filter once
                    print(f"Computing qualified labels for filter {filter} ...")
                    qualified_labels = compute_qualified_labels(
                        filter, self.dataset.train_mds
                    )
                    print(f"Finished computing qualified labels for filter {filter} ...")

                    for vec in self.dataset.test_vecs:
                        query_start = time.time()
                        # Use the new bitmap filter interface instead of complex predicate
                        # Type assertion needed since base Index class doesn't have this method
                        ids = self.index.search_with_bitmap_filter(  # type: ignore
                            vec, k=k, qualified_labels=qualified_labels.tolist()
                        )
                        query_latencies.append(time.time() - query_start)
                        query_results_filter.append(ids)

                if return_stats:
                    query_stats.append(self.index.get_search_stats())

                ground_truth = self.dataset.filter_to_ground_truth[filter]
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
            }

            if return_stats:
                template_results["query_stats"] = query_stats

            template_to_results[template] = self._compute_metrics(template_results)

            if return_verbose:
                template_results.update(
                    {
                        "query_recalls": query_recalls,
                        "query_results": query_results,
                        "query_latencies": query_latencies,
                    }
                )

        return template_to_results
