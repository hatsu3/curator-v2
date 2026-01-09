import json
from itertools import product
from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset, IndexProfiler
from indexes.curator import Curator as CuratorIndex


def exp_curator_opt(
    output_path: str,
    dataset_cache_path: str | Path,
    nlist: int = 16,
    max_sl_size: int = 256,
    search_ef_space: list[int] = [32, 64, 128, 256, 512],
    beam_size_space: list[int] = [1, 2, 4, 8],
    variance_boost_space: list[float] = [0.4],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    num_runs: int = 1,
    return_verbose: bool = False,
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    assert Path(dataset_cache_path).exists(), f"Dataset cache path {dataset_cache_path} does not exist"
    dataset = Dataset.from_dataset_key(
        dataset_key, test_size=test_size, cache_path=dataset_cache_path
    )
    profiler.set_dataset(dataset)

    print(f"Building index with nlist = {nlist}, max_sl_size = {max_sl_size} ... ")
    index_config = IndexConfig(
        index_cls=CuratorIndex,
        index_params={
            "d": dataset.dim,
            "nlist": nlist,
            "max_sl_size": max_sl_size,
            "max_leaf_size": max_sl_size,
            "clus_niter": 20,
            "bf_capacity": dataset.num_labels,
            "bf_error_rate": 0.01,
        },
        search_params={
            "variance_boost": variance_boost_space[0],
            "search_ef": search_ef_space[0],
            "beam_size": beam_size_space[0],
        },
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=False,
    )

    results = list()
    for search_ef, beam_size, variance_boost in product(
        search_ef_space, beam_size_space, variance_boost_space
    ):
        print(
            f"Querying index with search_ef = {search_ef}, beam_size = {beam_size}, "
            f"variance_boost = {variance_boost} ... "
        )
        profiler.set_index_search_params(
            {
                "search_ef": search_ef,
                "beam_size": beam_size,
                "variance_boost": variance_boost,
            }
        )
        query_results = profiler.do_query(
            num_runs=num_runs, return_verbose=return_verbose
        )
        results.append(
            {
                "nlist": nlist,
                "max_sl_size": max_sl_size,
                "search_ef": search_ef,
                "beam_size": beam_size,
                "variance_boost": variance_boost,
                **build_results,
                **query_results,
            }
        )

    delete_results = profiler.do_delete()
    for result in results:
        result.update(delete_results)

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def exp_curator_opt_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    nlist_space: list[int] = [16, 32],
    max_sl_size_space: list[int] = [16, 32, 64, 128, 256, 384],
    search_ef_space: list[int] = [32, 64, 128, 256, 512, 768, 1024],
    beam_size_space: list[int] = [1, 2, 4, 8],
    variance_boost_space: list[float] = [0.4],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_dir: str | Path = "output/overall_results/curator_opt",
):
    params = vars()
    cpu_groups = list(map(str, range(cpu_range[0], cpu_range[1] + 1)))

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for nlist, max_sl_size in product(nlist_space, max_sl_size_space):
        task_name = f"nlist{nlist}_sl{max_sl_size}"

        if max_sl_size < nlist:
            continue

        if (results_dir / f"{task_name}.csv").exists():
            print(f"Skipping finished task {task_name} ...")
            continue

        command = batch_profiler.build_command(
            module="benchmark.overall_results.curator_opt",
            func="exp_curator_opt",
            output_path=str(results_dir / f"{task_name}.csv"),
            nlist=nlist,
            max_sl_size=max_sl_size,
            search_ef_space=search_ef_space,
            beam_size_space=beam_size_space,
            variance_boost_space=variance_boost_space,
            dataset_key=dataset_key,
            test_size=test_size,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    """
    python -m benchmark.overall_results.curator_opt \
        exp_curator_opt \
            --output_path test_curator.csv \
            --nlist 16 \
            --max_sl_size 256 \
            --search_ef_space "[128]" \
            --beam_size_space "[4]" \
            --variance_boost_space "[0.4]" \
            --dataset_key yfcc100m \
            --test_size 0.01
    """
    fire.Fire()
