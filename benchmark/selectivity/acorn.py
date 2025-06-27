import json
from pathlib import Path
import pickle as pkl

import fire
import pandas as pd
import numpy as np

from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.selectivity.dataset import SelectivityDataset
from indexes.acorn import ACORN, write_dataset


def load_latencies_from_binary(latencies_path: str | Path) -> list[float]:
    with Path(latencies_path).open("rb") as file:
        num_queries = int.from_bytes(file.read(8), byteorder="little", signed=False)
        latencies = np.fromfile(file, dtype=np.float64, count=num_queries)
    return latencies.tolist()


def write_selectivity_dataset(
    dataset_dir: str | Path,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    overwrite: bool = False,
):
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    dataset = SelectivityDataset.load(dataset_cache_dir)

    write_dataset(
        dataset.train_vecs,
        dataset.train_mds,
        dataset_dir,
        "train",
        overwrite=overwrite,
    )
    write_dataset(
        dataset.test_vecs,
        dataset.test_mds,
        dataset_dir,
        "test",
        overwrite=overwrite,
    )


def construct_acorn_index(
    dataset_dir: str | Path,
    index_dir: str | Path,
    m: int,
    gamma: int,
    m_beta: int,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    build_results_path: str | Path | None = None,
):
    profiler = IndexProfiler()

    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)
    profiler.set_dataset(dataset)

    print(f"Building index with m = {m}, gamma = {gamma}, m_beta = {m_beta} ...")
    index_config = IndexConfig(
        index_cls=ACORN,
        index_params={
            "dataset_dir": dataset_dir,
            "index_dir": index_dir,
            "m": m,
            "gamma": gamma,
            "m_beta": m_beta,
        },
        search_params={
            "search_ef": 16,
        },
    )

    build_results = profiler.do_build(
        index_config=index_config,
        do_train=False,
        batch_insert=True,  # acorn only supports batch insert
    )

    memory_usage = json.load(open(Path(index_dir) / "memory_usage.json"))
    build_results["index_size_kb"] = memory_usage["memory_usage_kb"]

    print(f"Index saved to {index_dir} ...")

    construct_params_path = Path(index_dir) / "construct_params.json"
    print(f"Writing construct params to {construct_params_path} ...")
    json.dump(
        {
            "dataset_dir": str(dataset_dir),
            "index_dir": str(index_dir),
            "m": m,
            "gamma": gamma,
            "m_beta": m_beta,
            "dataset_cache_dir": dataset_cache_dir,
        },
        open(construct_params_path, "w"),
    )

    if build_results_path is None:
        build_results_path = Path(index_dir) / "build_results.json"

    print(f"Writing build results to {build_results_path} ...")
    Path(build_results_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(build_results, open(build_results_path, "w"))


def evaluate_acorn_index(
    index_dir: str | Path,
    search_ef_space: list[int],
    output_path: str | Path,
    num_runs: int = 1,
    return_verbose: bool = False,
):
    profiler = IndexProfiler()

    print(f"Loading index from {index_dir} ...")
    construct_params = json.load(open(Path(index_dir) / "construct_params.json"))
    print(f"Construct params: {construct_params}")

    dataset_cache_dir = construct_params["dataset_cache_dir"]
    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)
    profiler.set_dataset(dataset)

    index_config = IndexConfig(
        index_cls=ACORN,
        index_params={
            "dataset_dir": str(construct_params["dataset_dir"]),
            "index_dir": index_dir,
            "m": construct_params["m"],
            "gamma": construct_params["gamma"],
            "m_beta": construct_params["m_beta"],
        },
        search_params={
            "search_ef": search_ef_space[0],
        },
    )
    index = index_config.index_cls(**index_config.index_params)
    profiler.set_index(index)

    results = list()
    for search_ef in search_ef_space:
        print(f"Querying index with search_ef = {search_ef} ...")
        profiler.set_index_search_params({"search_ef": search_ef})
        query_results = profiler.do_query(
            batch_query=True,
            num_runs=num_runs,
            return_verbose=return_verbose,
        )
        if return_verbose:
            query_results["query_latencies"] = load_latencies_from_binary(
                Path(index_dir) / f"search_latency.bin"
            )

        results.append(
            {
                "m": construct_params["m"],
                "gamma": construct_params["gamma"],
                "m_beta": construct_params["m_beta"],
                "search_ef": search_ef,
                **query_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if return_verbose:
        pkl.dump(results, open(output_path, "wb"))
    else:
        pd.DataFrame(results).to_csv(output_path, index=False)


if __name__ == "__main__":
    fire.Fire()
