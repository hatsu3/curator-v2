import json
from pathlib import Path

import fire

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, IndexProfiler
from benchmark.selectivity.dataset import SelectivityDataset
from indexes.pre_filtering import PreFilteringIndex


def exp_pre_filtering_selectivity(
    output_path: str,
    max_selectivity: float = 0.1,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
):
    profiler = IndexProfiler()

    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = SelectivityDataset.load(dataset_cache_dir)
    profiler.set_dataset(dataset)

    print(f"Building index ...")
    index_config = IndexConfig(
        index_cls=PreFilteringIndex,
        index_params={},
        search_params={},
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=False,
        batch_insert=True,
        track_stats=True,
    )

    # Run queries only for labels with low selectivity
    dataset = dataset.get_selectivity_range_split(max_sel=max_selectivity)
    profiler.set_dataset(dataset)

    results = list()
    print("Querying index ...")
    query_res = profiler.do_query(return_verbose=True, return_stats=True)
    query_res.pop("query_results")
    results.append(
        {
            **query_res,
            **build_results,
        }
    )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_pre_filtering_selectivity_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    max_selectivity: float = 0.1,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    output_dir: str | Path = "output/selectivity/pre_filtering",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    task_name = "no_params"
    command = batch_profiler.build_command(
        module="benchmark.selectivity.pre_filtering",
        func="exp_pre_filtering_selectivity",
        output_path=str(results_dir / f"{task_name}.json"),
        max_selectivity=max_selectivity,
        dataset_cache_dir=dataset_cache_dir,
    )
    batch_profiler.submit(task_name, command)
    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
