import json
from itertools import product
from pathlib import Path

import fire

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, IndexProfiler
from benchmark.selectivity.dataset import SelectivityDataset
from indexes.curator import Curator as CuratorIndex


def exp_curator_opt_selectivity(
    output_path: str,
    nlist: int = 16,
    max_sl_size: int = 256,
    search_ef_space: list[int] = [32, 64, 128, 256, 512, 768, 1024],
    beam_size_space: list[int] = [1, 2, 4, 8],
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
):
    profiler = IndexProfiler()

    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)
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
            "bf_capacity": 1000,
            "bf_error_rate": 0.01,
        },
        search_params={
            "variance_boost": 0.4,
            "search_ef": search_ef_space[0],
            "beam_size": beam_size_space[0],
        },
    )

    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=False,
        track_stats=False,
    )

    results = list()
    for search_ef, beam_size in product(search_ef_space, beam_size_space):
        print(
            f"Querying index with search_ef = {search_ef}, beam_size = {beam_size} ... "
        )
        profiler.set_index_search_params(
            {
                "search_ef": search_ef,
                "beam_size": beam_size,
            }
        )
        query_res = profiler.do_query(return_verbose=True, return_stats=False)
        query_res.pop("query_results")  # track per-query recalls and latencies
        results.append(
            {
                "nlist": nlist,
                "max_sl_size": max_sl_size,
                "search_ef": search_ef,
                "beam_size": beam_size,
                **query_res,
                **build_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_curator_opt_selectivity_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    nlist_space: list[int] = [16, 32],
    max_sl_size_space: list[int] = [128, 256, 384],
    search_ef_space: list[int] = [32, 64, 128, 256, 512, 768, 1024],
    beam_size_space: list[int] = [1, 2, 4, 8],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    output_dir: str | Path = "output/selectivity/curator_opt",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for nlist, max_sl_size in product(nlist_space, max_sl_size_space):
        task_name = f"nlist{nlist}_maxsl{max_sl_size}"
        command = batch_profiler.build_command(
            module="benchmark.selectivity.curator_opt",
            func="exp_curator_opt_selectivity",
            output_path=str(results_dir / f"{task_name}.json"),
            nlist=nlist,
            max_sl_size=max_sl_size,
            search_ef_space=search_ef_space,
            beam_size_space=beam_size_space,
            dataset_cache_dir=dataset_cache_dir,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
