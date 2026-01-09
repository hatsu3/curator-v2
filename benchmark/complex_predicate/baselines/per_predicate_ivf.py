import json
from pathlib import Path

import fire

from benchmark.complex_predicate.dataset import (
    ComplexPredicateDataset,
    PerPredicateDataset,
)
from benchmark.config import IndexConfig
from benchmark.profiler import BatchProfiler, IndexProfiler
from indexes.per_label_ivf import PerLabelIVF


def exp_per_predicate_ivf_complex_predicate(
    output_path: str,
    nlist: int = 10,
    nprobe_space: list[int] = [1, 2, 4, 8],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    dataset = ComplexPredicateDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        templates=templates,
        n_filters_per_template=n_filters_per_template,
        n_queries_per_filter=n_queries_per_filter,
        gt_cache_dir=gt_cache_dir,
    )
    dataset = PerPredicateDataset.from_complex_predicate_dataset(dataset)
    template_splits = {
        template: dataset.get_template_split(template) for template in templates
    }

    index_config = IndexConfig(
        index_cls=PerLabelIVF,
        index_params={
            "d": dataset.dim,
            "nlist": nlist,
        },
        search_params={
            "nprobe": nprobe_space[0],
        },
    )

    results = list()
    per_template_results = dict()

    for template in templates:
        profiler.set_dataset(template_splits[template])

        print(f"Building index for template {template} with nlist = {nlist} ...")
        profiler.do_build(
            index_config=index_config,
            do_train=True,
            batch_insert=False,
        )

        for nprobe in nprobe_space:
            print(f"Querying index with nprobe = {nprobe} ...")
            profiler.set_index_search_params({"nprobe": nprobe})
            query_results = profiler.do_query()

            if nprobe not in per_template_results:
                per_template_results[nprobe] = dict()
            per_template_results[nprobe][template] = query_results

    for nprobe in nprobe_space:
        results.append(
            {
                "nlist": nlist,
                "nprobe": nprobe,
                "per_template_results": per_template_results[nprobe],
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(output_path, "w"))


def exp_per_predicate_ivf_complex_predicate_param_sweep(
    cpu_groups: list[str] = ["0-3", "4-7", "8-11", "12-15"],
    nlist_space: list[int] = [10, 20, 30, 40],
    nprobe_space: list[int] = [1, 2, 4, 8],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 10,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
    output_dir: str | Path = "output/complex_predicate/per_predicate_ivf",
):
    params = vars()

    output_dir = Path(output_dir) / f"{dataset_key}_test{test_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    json.dump(params, open(output_dir / "config.json", "w"))
    results_dir, logs_dir = output_dir / "results", output_dir / "logs"
    batch_profiler = BatchProfiler(cpu_groups, show_progress=True, log_dir=logs_dir)

    for nlist in nlist_space:
        task_name = f"nlist{nlist}"
        command = batch_profiler.build_command(
            module="benchmark.complex_predicate.baselines.per_predicate_ivf",
            func="exp_per_predicate_ivf_complex_predicate",
            output_path=str(results_dir / f"{task_name}.json"),
            nlist=nlist,
            nprobe_space=nprobe_space,
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
