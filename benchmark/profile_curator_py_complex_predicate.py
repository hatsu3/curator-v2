import numpy as np
from indexes.curator_py import CuratorPy, BoolExprFilter

from benchmark.utils import get_dataset_config, load_dataset


# TODO: merge complex predicate branch into main
def compute_ground_truth(
    X: np.ndarray,
    access_lists: list[list[int]],
    filter_str: str,
    k: int = 10,
):
    ...
