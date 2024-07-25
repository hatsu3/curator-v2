import json
from itertools import product
from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, Dataset, IndexProfiler
from indexes.ivf_flat_mt_faiss import IVFFlatMultiTenantFaiss as SharedIVF


# in each experiment we test a single nprobe value because the experiment
# time is dominated by the query time for shared IVF
def exp_shared_ivf(
    output_path: str,
    nlist: int = 100,
    nprobe: int = 8,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    num_runs: int = 1,
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    profiler.set_dataset(dataset)

    print(f"Building index with nlist = {nlist} ...")
    index_config = IndexConfig(
        index_cls=SharedIVF,
        index_params={
            "d": dataset.dim,
            "nlist": nlist,
        },
        search_params={
            "nprobe": nprobe,
        },
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=False,
    )

    print(f"Querying index with nprobe = {nprobe} ...")
    profiler.set_index_search_params({"nprobe": nprobe})
    query_results = profiler.do_query(num_runs=num_runs)
    results = [
        {
            "nlist": nlist,
            "nprobe": nprobe,
            **build_results,
            **query_results,
        }
    ]

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def exp_shared_ivf_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    nlist_space: list[int] = [200, 400, 800, 1600],
    nprobe_space: list[int] = [8, 16, 32, 64],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_dir: str | Path = "output/overall_results/shared_ivf",
):
    params = vars()
    cpu_groups = list(map(str, range(cpu_range[0], cpu_range[1] + 1)))

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for nlist, nprobe in product(nlist_space, nprobe_space):
        if nprobe > nlist:
            continue

        task_name = f"nlist{nlist}_nprobe{nprobe}"
        command = batch_profiler.build_command(
            module="benchmark.overall_results.shared_ivf",
            func="exp_shared_ivf",
            output_path=str(results_dir / f"{task_name}.csv"),
            nlist=nlist,
            nprobe=nprobe,
            dataset_key=dataset_key,
            test_size=test_size,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
