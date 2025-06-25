import json
from pathlib import Path

import fire
import numpy as np

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.profiler import IndexProfilerForComplexPredicate
from benchmark.complex_predicate.utils import evaluate_predicate
from benchmark.config import IndexConfig
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss as CuratorIndex


class DebugIndexProfilerForComplexPredicate(IndexProfilerForComplexPredicate):
    def __init__(self, seed: int = 42):
        super().__init__(seed)

    def validate_result(
        self, result_ids: list[int], filter_str: str, query_idx: int
    ) -> dict:
        """Validate that result IDs actually match the filter criteria"""
        assert self.dataset is not None, "Dataset not set"
        # Get the ground truth for this filter and query
        ground_truth = self.dataset.filter_to_ground_truth[filter_str][query_idx]

        validation_results = {
            "result_ids": result_ids,
            "ground_truth": ground_truth,
            "n_results": len([id for id in result_ids if id != -1]),
            "n_gt": len([id for id in ground_truth if id != -1]),
            "valid_results": [],
            "invalid_results": [],
            "missing_from_results": [],
        }

        # Check each result ID against train metadata
        for result_id in result_ids:
            if result_id == -1:  # padding
                continue

            # Get metadata for this result
            train_metadata = self.dataset.train_mds[result_id]

            # Check if this result actually satisfies the filter
            satisfies_filter = evaluate_predicate(filter_str, train_metadata)

            if satisfies_filter:
                validation_results["valid_results"].append(result_id)
            else:
                validation_results["invalid_results"].append(
                    {
                        "id": result_id,
                        "metadata": train_metadata,
                    }
                )

        # Check which ground truth IDs are missing from results
        result_set = set(id for id in result_ids if id != -1)
        gt_set = set(id for id in ground_truth if id != -1)
        validation_results["missing_from_results"] = list(gt_set - result_set)

        # Calculate intersection
        validation_results["intersection"] = list(result_set & gt_set)
        validation_results["precision"] = len(
            validation_results["valid_results"]
        ) / max(1, validation_results["n_results"])
        validation_results["filter_precision"] = len(
            validation_results["intersection"]
        ) / max(1, validation_results["n_results"])
        validation_results["recall"] = len(validation_results["intersection"]) / max(
            1, validation_results["n_gt"]
        )

        return validation_results

    def compute_distances(
        self, query_vec: np.ndarray, result_ids: list[int], ground_truth: list[int]
    ) -> dict:
        """Compute L2 distances for results and ground truth"""
        assert self.dataset is not None, "Dataset not set"
        distances = {
            "result_distances": [],
            "gt_distances": [],
        }

        # Compute distances for results
        for result_id in result_ids:
            if result_id == -1:
                continue
            train_vec = self.dataset.train_vecs[result_id]
            dist = float(np.linalg.norm(query_vec - train_vec))
            distances["result_distances"].append((result_id, dist))

        # Compute distances for ground truth
        for gt_id in ground_truth:
            if gt_id == -1:
                continue
            train_vec = self.dataset.train_vecs[gt_id]
            dist = float(np.linalg.norm(query_vec - train_vec))
            distances["gt_distances"].append((gt_id, dist))

        # Sort by distance
        distances["result_distances"].sort(key=lambda x: x[1])
        distances["gt_distances"].sort(key=lambda x: x[1])

        return distances

    def compute_qualified_vectors_python(self, filter_str: str) -> set[int]:
        """Compute qualified vectors using Python reference implementation"""
        assert self.dataset is not None, "Dataset not set"

        qualified_vectors = set()

        # Iterate over all training vectors and check which ones satisfy the filter
        for vec_id in range(len(self.dataset.train_vecs)):
            train_metadata = self.dataset.train_mds[vec_id]
            if evaluate_predicate(filter_str, train_metadata):
                qualified_vectors.add(vec_id)

        return qualified_vectors

    def compare_qualified_vectors(
        self, filter_str: str, cpp_file_path: str = "/tmp/qualified_vecs.txt"
    ) -> dict:
        """Compare Python-computed qualified vectors with C++ output file"""
        # Compute Python reference
        python_qualified = self.compute_qualified_vectors_python(filter_str)

        # Read C++ output file
        cpp_qualified = set()
        try:
            with open(cpp_file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # Skip comments
                        cpp_qualified.add(int(line))
        except FileNotFoundError:
            print(f"    WARNING: C++ output file not found: {cpp_file_path}")
            return {
                "python_count": len(python_qualified),
                "cpp_count": 0,
                "match": False,
                "only_in_python": list(python_qualified),
                "only_in_cpp": [],
                "intersection": [],
            }
        except Exception as e:
            print(f"    WARNING: Error reading C++ output file: {e}")
            return {
                "python_count": len(python_qualified),
                "cpp_count": 0,
                "match": False,
                "only_in_python": list(python_qualified),
                "only_in_cpp": [],
                "intersection": [],
            }

        # Compare the two sets
        only_in_python = python_qualified - cpp_qualified
        only_in_cpp = cpp_qualified - python_qualified
        intersection = python_qualified & cpp_qualified

        comparison = {
            "python_count": len(python_qualified),
            "cpp_count": len(cpp_qualified),
            "match": len(only_in_python) == 0 and len(only_in_cpp) == 0,
            "only_in_python": sorted(list(only_in_python)),
            "only_in_cpp": sorted(list(only_in_cpp)),
            "intersection": sorted(list(intersection)),
        }

        return comparison

    def do_debug_query(
        self, k: int = 10, max_filters_to_debug: int = 3, max_vectors_to_debug: int = 2
    ) -> dict[str, dict]:
        """Debug version of do_query that validates results and prints debugging info"""
        assert self.index is not None, "Index not set"
        assert self.dataset is not None, "Dataset not set"

        template_to_results: dict[str, dict] = dict()

        for template, filters in self.dataset.template_to_filters.items():
            print(f"\n=== Debugging template: {template} ===")

            query_recalls = []
            debug_info = []

            filters_debugged = 0
            for filter_idx, filter_str in enumerate(filters):
                if filters_debugged >= max_filters_to_debug:
                    print(
                        f"Skipping remaining filters for template {template} (already debugged {max_filters_to_debug})"
                    )
                    break

                print(f"\n--- Filter {filter_idx}: {filter_str} ---")
                print(
                    f"Filter selectivity: {self.dataset.filter_to_selectivity[filter_str]:.4f}"
                )

                filter_debug_info = {
                    "filter": filter_str,
                    "selectivity": self.dataset.filter_to_selectivity[filter_str],
                    "queries": [],
                    "qualified_vectors_comparison": None,
                }

                # First, run one query to trigger C++ qualified vectors computation
                dummy_query_vec = self.dataset.test_vecs[0]
                print(f"\n  Running dummy query to generate qualified vectors file...")
                _ = self.index.query_with_complex_predicate(
                    dummy_query_vec, k=k, predicate=filter_str
                )

                # Compare qualified vectors between Python and C++
                print(f"  Comparing qualified vectors: Python vs C++...")
                qualified_comparison = self.compare_qualified_vectors(filter_str)
                filter_debug_info["qualified_vectors_comparison"] = qualified_comparison

                print(
                    f"    Python qualified vectors: {qualified_comparison['python_count']}"
                )
                print(f"    C++ qualified vectors: {qualified_comparison['cpp_count']}")
                print(f"    Match: {qualified_comparison['match']}")

                if not qualified_comparison["match"]:
                    print(f"    ❌ MISMATCH DETECTED!")
                    print(
                        f"    Only in Python ({len(qualified_comparison['only_in_python'])}): {qualified_comparison['only_in_python'][:10]}{'...' if len(qualified_comparison['only_in_python']) > 10 else ''}"
                    )
                    print(
                        f"    Only in C++ ({len(qualified_comparison['only_in_cpp'])}): {qualified_comparison['only_in_cpp'][:10]}{'...' if len(qualified_comparison['only_in_cpp']) > 10 else ''}"
                    )
                else:
                    print(f"    ✅ Qualified vectors match perfectly!")

                for query_idx, query_vec in enumerate(self.dataset.test_vecs):
                    if (
                        query_idx >= max_vectors_to_debug
                    ):  # Debug only first N queries per filter
                        break

                    print(f"\n  Query {query_idx}:")

                    # Get results from index
                    result_ids = self.index.query_with_complex_predicate(
                        query_vec, k=k, predicate=filter_str
                    )

                    # Validate results
                    validation = self.validate_result(result_ids, filter_str, query_idx)

                    # Compute distances
                    ground_truth = self.dataset.filter_to_ground_truth[filter_str][
                        query_idx
                    ]

                    # Validate ground truth
                    gt_validation = self.validate_result(
                        ground_truth, filter_str, query_idx
                    )
                    if gt_validation["invalid_results"]:
                        print(
                            f"    INVALID GT IDs found: {gt_validation['invalid_results']}"
                        )

                    distances = self.compute_distances(
                        query_vec, result_ids, ground_truth
                    )

                    # Print debug info
                    print(f"    Results: {result_ids}")
                    print(f"    Ground truth: {ground_truth}")
                    print(
                        f"    Valid results: {len(validation['valid_results'])}/{validation['n_results']}"
                    )
                    print(f"    Invalid results: {len(validation['invalid_results'])}")
                    print(f"    Precision: {validation['precision']:.4f}")
                    print(f"    Recall: {validation['recall']:.4f}")

                    if validation["invalid_results"]:
                        print(f"    INVALID IDs found: {validation['invalid_results']}")

                    # Show top distances
                    print(f"    Top result distances: {distances['result_distances']}")
                    print(f"    Top GT distances: {distances['gt_distances']}")

                    query_debug = {
                        "query_idx": query_idx,
                        "validation": validation,
                        "distances": distances,
                    }
                    filter_debug_info["queries"].append(query_debug)
                    query_recalls.append(validation["recall"])

                debug_info.append(filter_debug_info)
                filters_debugged += 1

            # Compute average recall for this template
            avg_recall = np.mean(query_recalls) if query_recalls else 0.0

            template_to_results[template] = {
                "recall_at_k": avg_recall,
                "debug_info": debug_info,
                "n_queries_debugged": len(query_recalls),
            }

            print(f"\nTemplate {template} average recall: {avg_recall:.4f}")

        return template_to_results


def test_evaluate_predicate():
    """Test the evaluate_predicate function with simple access lists and OR filters"""
    print("\n=== Testing evaluate_predicate function ===")

    # Test cases: (formula, access_list, expected_result, description)
    test_cases = [
        # Basic single tenant tests
        ("1", [1], True, "Single tenant in access list"),
        ("1", [2], False, "Single tenant not in access list"),
        ("1", [], False, "Single tenant with empty access list"),
        # OR filter tests
        ("OR 1 2", [1], True, "OR filter: first tenant present"),
        ("OR 1 2", [2], True, "OR filter: second tenant present"),
        ("OR 1 2", [1, 2], True, "OR filter: both tenants present"),
        ("OR 1 2", [3], False, "OR filter: neither tenant present"),
        ("OR 1 2", [], False, "OR filter: empty access list"),
        ("OR 1 2", [1, 3, 5], True, "OR filter: first tenant among others"),
        ("OR 1 2", [3, 2, 5], True, "OR filter: second tenant among others"),
        # AND filter tests
        ("AND 1 2", [1, 2], True, "AND filter: both tenants present"),
        ("AND 1 2", [1], False, "AND filter: only first tenant present"),
        ("AND 1 2", [2], False, "AND filter: only second tenant present"),
        ("AND 1 2", [1, 2, 3], True, "AND filter: both tenants plus others"),
        ("AND 1 2", [], False, "AND filter: empty access list"),
        # NOT filter tests
        ("NOT 1", [], True, "NOT filter: tenant not in empty list"),
        ("NOT 1", [2], True, "NOT filter: tenant not in list"),
        ("NOT 1", [1], False, "NOT filter: tenant in list"),
        ("NOT 1", [1, 2], False, "NOT filter: tenant in list with others"),
        # Complex nested filters
        ("OR 1 AND 2 3", [1], True, "Complex: OR with AND - first branch true"),
        ("OR 1 AND 2 3", [2, 3], True, "Complex: OR with AND - second branch true"),
        ("OR 1 AND 2 3", [2], False, "Complex: OR with AND - second branch false"),
        ("OR 1 AND 2 3", [1, 2, 3], True, "Complex: OR with AND - both branches true"),
        ("OR 1 AND 2 3", [], False, "Complex: OR with AND - both branches false"),
        ("AND 1 OR 2 3", [1, 2], True, "Complex: AND with OR - both parts true"),
        ("AND 1 OR 2 3", [1, 3], True, "Complex: AND with OR - both parts true"),
        ("AND 1 OR 2 3", [1], False, "Complex: AND with OR - only first part true"),
        ("AND 1 OR 2 3", [2, 3], False, "Complex: AND with OR - only second part true"),
        # Three-way OR
        ("OR 1 OR 2 3", [1], True, "Three-way OR: first tenant"),
        ("OR 1 OR 2 3", [2], True, "Three-way OR: second tenant"),
        ("OR 1 OR 2 3", [3], True, "Three-way OR: third tenant"),
        ("OR 1 OR 2 3", [4], False, "Three-way OR: none present"),
        ("OR 1 OR 2 3", [1, 2, 3], True, "Three-way OR: all present"),
    ]

    passed = 0
    failed = 0

    print(f"Running {len(test_cases)} test cases...\n")

    for i, (formula, access_list, expected, description) in enumerate(test_cases):
        try:
            result = evaluate_predicate(formula, access_list)
            if result == expected:
                print(f"✅ Test {i+1:2d}: {description}")
                print(
                    f"    Formula: '{formula}', Access: {access_list}, Result: {result}"
                )
                passed += 1
            else:
                print(f"❌ Test {i+1:2d}: {description}")
                print(f"    Formula: '{formula}', Access: {access_list}")
                print(f"    Expected: {expected}, Got: {result}")
                failed += 1
        except Exception as e:
            print(f"💥 Test {i+1:2d}: {description}")
            print(f"    Formula: '{formula}', Access: {access_list}")
            print(f"    ERROR: {e}")
            failed += 1
        print()

    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")

    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {failed} tests failed")

    return failed == 0


def exp_curator_complex_predicate_debug(
    output_path: str,
    nlist: int = 16,
    max_sl_size: int = 256,
    search_ef: int = 20000,  # Use large search_ef for debugging
    beam_size: int = 4,
    variance_boost: float = 0.4,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["OR {0} {1}"],  # Focus on OR filters
    n_filters_per_template: int = 3,  # Fewer filters for detailed debugging
    n_queries_per_filter: int = 10,  # Fewer queries for detailed debugging
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
    max_filters_to_debug: int = 2,  # Maximum filters to debug per template
    max_vectors_to_debug: int = 2,  # Make this configurable
):
    """Debug version of curator experiment with detailed validation and distance checking"""

    profiler = DebugIndexProfilerForComplexPredicate()

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
            "search_ef": search_ef,
            "beam_size": beam_size,
            "variance_boost": variance_boost,
        },
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=False,
    )

    print(
        f"Running debug queries with search_ef = {search_ef}, beam_size = {beam_size}, variance_boost = {variance_boost} ..."
    )
    debug_results = profiler.do_debug_query(
        k=10,
        max_filters_to_debug=max_filters_to_debug,
        max_vectors_to_debug=max_vectors_to_debug,
    )

    results = {
        "nlist": nlist,
        "max_sl_size": max_sl_size,
        "search_ef": search_ef,
        "beam_size": beam_size,
        "variance_boost": variance_boost,
        "debug_results": debug_results,
        **build_results,
    }

    print(f"Saving debug results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    fire.Fire()
