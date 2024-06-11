from itertools import product

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss


def exp_ivf_hier_faiss(
    nlist_space=[4, 8, 16, 32],
    max_sl_size_space=[32, 64, 128, 256],
    update_bf_interval_space=[100],
    clus_niter_space=[20],
    max_leaf_size_space=[128],
    nprobe_space=[1200],
    prune_thres_space=[1.6],
    variance_boost_space=[0.4],
    dataset_key="arxiv-small",
    test_size=0.2,
    num_runs=1,
    timeout=600,
    output_path: str | None = None,
):
    if output_path is None:
        output_path = f"output/ivf_hier_faiss_{dataset_key}.csv"

    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_configs = [
        IndexConfig(
            index_cls=IVFFlatMultiTenantBFHierFaiss,
            index_params={
                "d": dim,
                "nlist": nlist,
                "bf_capacity": 1000,
                "bf_error_rate": 0.01,
                "max_sl_size": max_sl_size,
                "update_bf_interval": update_bf_interval,
                "clus_niter": clus_niter,
                "max_leaf_size": max_leaf_size,
            },
            search_params={
                "nprobe": nprobe,
                "prune_thres": prune_thres,
                "variance_boost": variance_boost,
            },
            train_params={
                "train_ratio": 1,
                "min_train": 50,
                "random_seed": 42,
            },
        )
        for nlist, max_sl_size, update_bf_interval, clus_niter, max_leaf_size, nprobe, prune_thres, variance_boost in product(
            nlist_space,
            max_sl_size_space,
            update_bf_interval_space,
            clus_niter_space,
            max_leaf_size_space,
            nprobe_space,
            prune_thres_space,
            variance_boost_space,
        )
    ]

    profiler = IndexProfiler(multi_tenant=True)
    results = profiler.batch_profile(
        index_configs, [dataset_config], num_runs=num_runs, timeout=timeout
    )

    if output_path is not None:
        df = pd.DataFrame(
            [
                {
                    **config.index_params,
                    **config.search_params,
                    **res,
                }
                for res, config in zip(results, index_configs)
            ]
        )
        df.to_csv(output_path, index=False)

    return results


if __name__ == "__main__":
    fire.Fire(exp_ivf_hier_faiss)
