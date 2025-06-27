import time

import numpy as np
from tqdm import tqdm

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
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
    ) -> dict[str, dict]:
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"

        template_to_results: dict[str, dict] = dict()

        pbar = tqdm(
            total=self.dataset.num_filters,
            desc="Querying index",
        )
        sorted_templates = sorted(self.dataset.template_to_filters.keys())
        for template in sorted_templates:
            filters = self.dataset.template_to_filters[template]

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

                for vec in self.dataset.test_vecs:
                    query_start = time.time()
                    ids = self.index.query_with_complex_predicate(
                        vec, k=k, predicate=filter
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

    def do_batch_query(
        self,
        k: int = 10,
        return_verbose: bool = False,
    ):
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"

        template_to_results: dict[str, dict] = dict()
        sorted_templates = sorted(self.dataset.template_to_filters.keys())

        all_filters = list()
        for template in sorted_templates:
            all_filters.extend(self.dataset.template_to_filters[template])

        all_results: list[list[int]] = self.index.batch_query_with_complex_predicate(  # type: ignore
            self.dataset.test_vecs, k=k, predicates=all_filters
        )

        all_results_offset = 0
        for template in sorted_templates:
            query_latencies = list()
            query_results = list()
            query_recalls = list()

            for filter in self.dataset.template_to_filters[template]:
                ground_truth = self.dataset.filter_to_ground_truth[filter]
                assert len(self.dataset.test_vecs) == len(ground_truth)
                query_results_filter = all_results[
                    all_results_offset : all_results_offset + len(ground_truth)
                ]

                query_recalls.extend(
                    [
                        recall([res], [gt])
                        for res, gt in zip(query_results_filter, ground_truth)
                    ]
                )
                query_results.extend(query_results_filter)
                query_latencies.extend([0] * len(ground_truth))

                all_results_offset += len(ground_truth)

            recall_at_k = np.mean(query_recalls).item()

            template_results = {
                "recall_at_k": recall_at_k,
                "query_latencies": query_latencies,
            }

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
