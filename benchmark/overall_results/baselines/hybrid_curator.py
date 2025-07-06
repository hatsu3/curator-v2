from pathlib import Path

import faiss
import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import Dataset, IndexProfiler
from indexes.hybrid_curator import HybridCurator


def exp_hybrid_curator(
    output_path: str,
    dataset_cache_path: str | Path,
    M: int = 32,
    gamma: int = 10,
    M_beta: int = 64,
    n_branches: int = 16,
    leaf_size: int = 128,
    use_local_sel: bool = False,
    sel_threshold: float = 0.2,
    search_ef_space: list[int] = [16, 32, 64, 128, 256],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    return_verbose: bool = False,
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    assert Path(
        dataset_cache_path
    ).exists(), f"Dataset cache path {dataset_cache_path} does not exist"
    dataset = Dataset.from_dataset_key(
        dataset_key, test_size=test_size, cache_path=dataset_cache_path
    )
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
            "use_local_sel": use_local_sel,
            "sel_threshold": sel_threshold,
        },
        search_params={
            "curator_search_ef": search_ef_space[0] * 8,
            "acorn_search_ef": search_ef_space[0],
        },
    )

    n_threads = faiss.omp_get_max_threads()
    print(f"Setting # threads to {n_threads} ...")

    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=True,
    )

    print(f"Setting # threads to 1 ...")
    faiss.omp_set_num_threads(1)

    results = list()
    for search_ef in search_ef_space:
        print(f"Querying index with search_ef = {search_ef} ... ")
        profiler.set_index_search_params(
            {
                "curator_search_ef": search_ef * 8,
                "acorn_search_ef": search_ef,
            }
        )
        query_results = profiler.do_query(
            batch_query=False,
            num_threads=1,
            return_verbose=return_verbose,
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
            --dataset_cache_path /path/to/dataset/cache \
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
