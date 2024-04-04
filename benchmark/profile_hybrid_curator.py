from itertools import product

import fire
import pandas as pd

from benchmark.config import IndexConfig
from benchmark.profiler import IndexProfiler
from benchmark.utils import get_dataset_config
from indexes.hybrid_curator import HybridCurator


def exp_hybrid_curator(
    m_space=[12, 24, 36, 48],
    branch_factor_space=[4, 8, 16, 32],
    buf_capacity_space=[8, 16, 32, 64],
    alpha_space=[1.0, 1.5, 2.0],
    dataset_key="arxiv-small",
    test_size=0.2,
    num_runs=1,
    timeout=600,
    output_path: str | None = None,
):
    if output_path is None:
        output_path = f"output/hybrid_curator_{dataset_key}.csv"

    dataset_config, dim = get_dataset_config(dataset_key, test_size=test_size)

    index_configs = [
        IndexConfig(
            index_cls=HybridCurator,
            index_params={
                "d": dim,
                "M": m,
                "branch_factor": branch_factor,
                "buf_capacity": buf_capacity,
            },
            search_params={
                "alpha": alpha,
            },
            train_params={
                "train_ratio": 1,
                "min_train": 50,
                "random_seed": 42,
            },
        )
        for m, branch_factor, buf_capacity, alpha in product(
            m_space, branch_factor_space, buf_capacity_space, alpha_space
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
    fire.Fire(exp_hybrid_curator)
