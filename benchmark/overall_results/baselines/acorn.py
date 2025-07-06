import json
import pickle as pkl
import shutil
from itertools import product
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset, IndexProfiler
from indexes.acorn import ACORN


def load_latencies_from_binary(latencies_path: str | Path) -> list[float]:
    with Path(latencies_path).open("rb") as file:
        num_queries = int.from_bytes(file.read(8), byteorder="little", signed=False)
        latencies = np.fromfile(file, dtype=np.float64, count=num_queries)
    # Convert from milliseconds to seconds
    return (latencies / 1000.0).tolist()


def construct_acorn_index(
    dataset_dir: str | Path,
    dataset_cache_path: str | Path,
    dataset_key: str,
    test_size: float,
    index_dir: str | Path,
    m: int,
    gamma: int,
    m_beta: int,
    build_results_path: str | Path | None = None,
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
            "dataset_key": dataset_key,
            "test_size": test_size,
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
    dataset_cache_path: str | Path,
    search_ef_space: list[int],
    output_path: str | Path,
    num_runs: int = 1,
    return_verbose: bool = False,
):
    profiler = IndexProfiler()

    print(f"Loading index from {index_dir} ...")
    construct_params = json.load(open(Path(index_dir) / "construct_params.json"))
    print(f"Construct params: {construct_params}")

    dataset_key = construct_params["dataset_key"]
    test_size = construct_params["test_size"]
    print(f"Loading dataset {dataset_key} ...")
    assert Path(
        dataset_cache_path
    ).exists(), f"Dataset cache path {dataset_cache_path} does not exist"
    dataset = Dataset.from_dataset_key(
        dataset_key, test_size=test_size, cache_path=dataset_cache_path
    )
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


def exp_acorn(
    output_path: str,
    dataset_cache_path: str | Path,
    dataset_dir: str,
    index_dir: str,
    m: int = 32,
    gamma: int = 1,
    m_beta: int = 64,
    search_ef_space: list[int] = [16, 32, 64, 128, 256, 512],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    num_runs: int = 1,
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
            "search_ef": search_ef_space[0],
        },
    )

    # Skip index build if index already exists
    delete_index = False  # do not delete cached index
    if (Path(index_dir) / "index.bin").exists():
        print(f"Index already exists at {index_dir}/index.bin, skipping build...")
        build_results = json.load(open(Path(index_dir) / "build_results.json"))
        profiler.set_index(index_config.index_cls(**index_config.index_params))
    else:
        delete_index = True
        build_results = profiler.do_build(
            index_config=index_config,
            do_train=False,
            batch_insert=True,  # acorn only supports batch insert
        )
        memory_usage = json.load(open(Path(index_dir) / "memory_usage.json"))
        build_results["index_size_kb"] = memory_usage["memory_usage_kb"]

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
                "m": m,
                "gamma": gamma,
                "m_beta": m_beta,
                "search_ef": search_ef,
                **build_results,
                **query_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)

    if delete_index:
        print(f"Deleting index at {index_dir} ...")
        shutil.rmtree(index_dir)


def exp_acorn_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    m_space: list[int] = [16, 32, 64],
    gamma_space: list[float] = [1, 10, 20],
    m_beta_multiplier_space: list[int] = [1, 2, 4],
    search_ef_space: list[int] = [8, 16, 32, 64, 128],
    dataset_dir: str = "data/acorn/overall_results/yfcc100m_test0.01",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_dir: str | Path = "output/overall_results/acorn",
    return_verbose: bool = False,
):
    params = vars()
    cpu_groups = list(map(str, range(cpu_range[0], cpu_range[1] + 1)))

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    index_dir = output_dir / "index"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for m, gamma, m_beta_multiplier in product(
        m_space, gamma_space, m_beta_multiplier_space
    ):
        if m_beta_multiplier > gamma:
            continue

        m_beta = m * m_beta_multiplier

        task_name = f"m{m}_gamma{gamma}_m_beta{m_beta}"
        output_path = results_dir / f"{task_name}.csv"
        if output_path.exists():
            print(f"Skipping {output_path} ...")
            continue

        command = batch_profiler.build_command(
            module="benchmark.overall_results.acorn",
            func="exp_acorn",
            output_path=str(output_path),
            dataset_dir=dataset_dir,
            index_dir=str(index_dir / task_name),
            m=m,
            gamma=gamma,
            m_beta=m_beta,
            search_ef_space=search_ef_space,
            dataset_key=dataset_key,
            test_size=test_size,
            return_verbose=return_verbose,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
