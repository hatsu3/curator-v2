from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import Dataset, IndexProfiler
from indexes.hybrid_curator import HybridCurator


# TODO: per-selectivity results (verbose output), select search_ef, 
# determine sel_threshold based on profiling results

# TODO: save index to a file

def exp_hybrid_curator(
    output_path: str,
    M: int = 32,
    gamma: int = 10,
    M_beta: int = 64,
    n_branches: int = 16,
    leaf_size: int = 128,
    sel_threshold: float = 0.2,
    search_ef_space: list[int] = [16, 32, 64, 128, 256],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    profiler.set_dataset(dataset)

    print(
        f"Building index with M = {M}, gamma = {gamma}, "
        f"M_beta = {M_beta}, n_branches = {n_branches}, leaf_size = {leaf_size} ... "
    )
    index_config = IndexConfig(
        index_cls=HybridCurator,
        index_params={
            "dim": dataset.dim,
            "M": M,
            "gamma": gamma,
            "M_beta": M_beta,
            "n_branches": n_branches,
            "leaf_size": leaf_size,
            "n_uniq_labels": dataset.num_labels,
        },
        search_params={
            "sel_threshold": sel_threshold,
            "curator_search_ef": search_ef_space[0] * 16,
            "acorn_search_ef": search_ef_space[0],
        },
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=True,
    )

    results = list()
    for search_ef in search_ef_space:
        print(f"Querying index with search_ef = {search_ef} ... ")
        profiler.set_index_search_params(
            {
                "sel_threshold": sel_threshold,
                "curator_search_ef": search_ef * 16,
                "acorn_search_ef": search_ef,
            }
        )

        query_results = profiler.do_query(
            batch_query=False,
            num_threads=1,
        )
        results.append(
            {
                "M": M,
                "gamma": gamma,
                "M_beta": M_beta,
                "n_branches": n_branches,
                "leaf_size": leaf_size,
                "sel_threshold": sel_threshold,
                "search_ef": search_ef,
                **build_results,
                **query_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


if __name__ == "__main__":
    """
    python -m benchmark.overall_results.hybrid_curator \
        exp_hybrid_curator \
            --output_path test_hybrid_curator.csv \
            --M 32 \
            --gamma 10 \
            --M_beta 64 \
            --n_branches 16 \
            --leaf_size 128 \
            --sel_threshold 0.2 \
            --search_ef_space "[16, 32, 64, 128, 256]" \
            --dataset_key yfcc100m \
            --test_size 0.01
    """
    fire.Fire()
