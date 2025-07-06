import json
from itertools import product
from pathlib import Path

import fire

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.complex_predicate.profiler import IndexProfilerForComplexPredicate
from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler
from indexes.ivf_flat_mt_faiss import IVFFlatMultiTenantFaiss as SharedIVF


def exp_shared_ivf_complex_predicate(
    output_path: str,
    nlist: int = 200,
    nprobe: int = 8,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
):
    profiler = IndexProfilerForComplexPredicate()

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
    per_template_results = profiler.do_query()
    results = [
        {
            "nlist": nlist,
            "nprobe": nprobe,
            "per_template_results": per_template_results,
            **build_results,
        }
    ]

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_shared_ivf_complex_predicate_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    nlist_space: list[int] = [200, 400, 800, 1600],
    nprobe_space: list[int] = [8, 16, 32, 64],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
    output_dir: str | Path = "output/complex_predicate/shared_ivf",
):
    params = vars()

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
            module="benchmark.complex_predicate.baselines.shared_ivf",
            func="exp_shared_ivf_complex_predicate",
            output_path=str(results_dir / f"{task_name}.json"),
            nlist=nlist,
            nprobe=nprobe,
            dataset_key=dataset_key,
            test_size=test_size,
            templates=templates,
            n_filters_per_template=n_filters_per_template,
            n_queries_per_filter=n_queries_per_filter,
            gt_cache_dir=gt_cache_dir,
        )
        batch_profiler.submit(task_name, command)

    batch_profiler.run()


if __name__ == "__main__":
    fire.Fire()
