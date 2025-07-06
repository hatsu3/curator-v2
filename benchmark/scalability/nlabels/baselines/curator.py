import json
from itertools import product
from pathlib import Path

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, IndexProfiler
from benchmark.scalability.dataset import ScalabilityDataset
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss as CuratorIndex


def exp_curator_scalability(
    output_path: str,
    n_labels: int,
    nlist: int = 16,
    max_sl_size: int = 256,
    nprobe_space: list[int] = [2000, 3000, 4000],
    prune_thres_space: list[float] = [1.2, 1.6, 2.0],
    dataset_cache_dir: str = "data/scalability/random_yfcc100m",
    seed: int = 42,
):
    profiler = IndexProfiler()

    print(f"Loading dataset from {dataset_cache_dir} ...")
    dataset = ScalabilityDataset.load(dataset_cache_dir)
    if dataset.n_labels < n_labels:
        raise ValueError(
            f"Dataset has fewer labels ({dataset.n_labels}) than specified ({n_labels})"
        )

    split = dataset.get_random_split(n_labels, seed=seed)
    profiler.set_dataset(split)

    print(f"Building index with nlist = {nlist}, max_sl_size = {max_sl_size} ...")
    index_config = IndexConfig(
        index_cls=CuratorIndex,
        index_params={
            "d": dataset.dim,
            "nlist": nlist,
            "max_sl_size": max_sl_size,
            "max_leaf_size": max_sl_size,
            "clus_niter": 20,
            "bf_capacity": n_labels,
            "bf_error_rate": 0.01,
        },
        search_params={
            "nprobe": nprobe_space[0],
            "prune_thres": prune_thres_space[0],
            "variance_boost": 0.4,
        },
    )
    build_results = profiler.do_build(
        index_config=index_config,
        do_train=True,
        batch_insert=False,
    )

    results = list()
    for nprobe, prune_thres in product(nprobe_space, prune_thres_space):
        print(f"Querying index with nprobe = {nprobe}, prune_thres = {prune_thres} ...")
        profiler.set_index_search_params({"nprobe": nprobe, "prune_thres": prune_thres})
        query_results = profiler.do_query()
        results.append(
            {
                "nlist": nlist,
                "max_sl_size": max_sl_size,
                "nprobe": nprobe,
                "prune_thres": prune_thres,
                **query_results,
                **build_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(output_path, index=False)


def exp_curator_scalability_param_sweep(
    cpu_range: tuple[int, int] = (0, 15),
    n_labels_n_steps: int = 10,
    n_labels_max: int = 10000,
    nlist_space: list[int] = [8, 16, 32],
    max_sl_size_space: list[int] = [64, 128, 256],
    nprobe_space: list[int] = [2000, 3000, 4000],
    prune_thres_space: list[float] = [1.2, 1.6, 2.0],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/scalability/random_yfcc100m",
    seed: int = 42,
    output_dir: str | Path = "output/scalability/curator",
):
    params = vars()
    cpu_groups = list(map(str, range(cpu_range[0], cpu_range[1] + 1)))

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    n_labels_min = n_labels_max // (2 ** (n_labels_n_steps - 1))
    n_labels_space = [n_labels_min * (2 ** i) for i in range(n_labels_n_steps)]

    for n_labels, nlist, max_sl_size in product(
        n_labels_space, nlist_space, max_sl_size_space
    ):
        task_name = f"n_labels{n_labels}_nlist{nlist}_sl{max_sl_size}"
        command = batch_profiler.build_command(
            module="benchmark.scalability.curator",
            func="exp_curator_scalability",
            output_path=str(results_dir / f"{task_name}.csv"),
            n_labels=n_labels,
            nlist=nlist,
            max_sl_size=max_sl_size,
            nprobe_space=nprobe_space,
            prune_thres_space=prune_thres_space,
            dataset_cache_dir=dataset_cache_dir,
            seed=seed,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
