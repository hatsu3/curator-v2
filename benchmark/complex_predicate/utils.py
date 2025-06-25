import hashlib
import pickle as pkl
from pathlib import Path

import numpy as np
from tqdm import tqdm


def evaluate_predicate(formula: str, mds: list[int]):
    """
    Evaluate a boolean formula in Polish notation.
    Example: 'AND 1 OR NOT 0 2'
    A variable (like '1') is considered True if it is in mds.
    """
    tokens = formula.split()
    stack = []

    def eval_token(token):
        try:
            return int(token) in mds
        except ValueError:
            raise ValueError(f"Malformed formula: {formula}")

    for token in reversed(tokens):
        if token in ("AND", "OR", "NOT"):
            if token == "AND":
                a = stack.pop()
                b = stack.pop()
                stack.append(a and b)
            elif token == "OR":
                a = stack.pop()
                b = stack.pop()
                stack.append(a or b)
            elif token == "NOT":
                a = stack.pop()
                stack.append(not a)
        else:
            stack.append(eval_token(token))

    return stack.pop()


def generate_random_filters(
    templates: list[str],
    num_filters_per_template: int,
    all_labels: list[int] | int,
    seed: int = 42,
):
    np.random.seed(seed)

    if isinstance(all_labels, int):
        all_labels = list(range(all_labels))

    def sample_parameters(n, d):
        return [
            np.random.choice(all_labels, d, replace=False).tolist() for _ in range(n)
        ]

    filters = {}
    for template in templates:
        n_param = template.count("{")
        filters[template] = []
        for param in sample_parameters(num_filters_per_template, n_param):
            filters[template].append(template.format(*param))

    return filters


def compute_ground_truth(
    queries: np.ndarray,
    filters: list[str],
    train_vecs: np.ndarray,
    train_mds: list[list[int]],
    k: int = 10,
    cache_dir: str | None = None,
    batch_size: int = 8,
) -> tuple[dict[str, list[list[int]]], dict[str, float]]:
    try:
        import torch

        print("Using PyTorch for computing distances")
        use_cuda = True
    except:
        print("Using numpy for computing distances")
        use_cuda = False

    def compute_cache_key(queries, filters, k) -> str:
        arr_bytes = queries.tobytes()
        hasher = hashlib.md5()
        hasher.update(arr_bytes)
        for filter in sorted(filters):
            hasher.update(filter.encode())
        hasher.update(str(k).encode())
        return hasher.hexdigest()

    cache_key = compute_cache_key(queries, filters, k)

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_path = Path(cache_dir) / f"{cache_key}.pkl"
        if cache_path.exists():
            data = pkl.load(cache_path.open("rb"))
            return data["ground_truths"], data["selectivities"]
    else:
        cache_path = None

    ground_truths: dict[str, list[list[int]]] = {}
    selectivities: dict[str, float] = {}

    for filter in tqdm(filters, desc="Computing ground truth"):
        filtered_ids = np.array(
            [
                i
                for i, md in enumerate(train_mds)
                if md and evaluate_predicate(filter, md)
            ]
        )

        selectivities[filter] = filtered_ids.size / train_vecs.shape[0]

        if filtered_ids.size == 0:
            ground_truths[filter] = [[-1] * k for _ in range(queries.shape[0])]
            continue

        filtered_vecs = train_vecs[filtered_ids]

        if use_cuda:
            filtered_vecs_pth = torch.from_numpy(filtered_vecs).unsqueeze(0).cuda()
            queries_pth = torch.from_numpy(queries).unsqueeze(1).cuda()

            topk_ids = []
            for i in range(0, len(queries), batch_size):
                queries_pth_batch = queries_pth[i : i + batch_size]
                dists_pth_batch = torch.norm(
                    filtered_vecs_pth - queries_pth_batch, dim=2
                )
                topk_pth_batch = torch.argsort(dists_pth_batch, dim=1)[:, :k]
                topk_ids_batch = filtered_ids[topk_pth_batch.cpu().numpy()]
                topk_ids.extend(topk_ids_batch.tolist())

            topk_ids = np.array(topk_ids)
        else:
            dists = np.linalg.norm(filtered_vecs - queries[:, None], axis=2)
            topk_ids = filtered_ids[np.argsort(dists, axis=1)[:, :k]]

        topk_ids = np.pad(
            topk_ids, ((0, 0), (0, k - topk_ids.shape[1])), constant_values=-1
        )
        ground_truths[filter] = topk_ids.tolist()

    if cache_path is not None:
        pkl.dump(
            {"ground_truths": ground_truths, "selectivities": selectivities},
            cache_path.open("wb"),
        )

    return ground_truths, selectivities


def compute_qualified_labels(filter_str: str, train_mds: list[list[int]]) -> np.ndarray:
    """
    Compute the list of vector IDs that satisfy the given filter.

    Parameters
    ----------
    filter_str : str
        Filter string in Polish notation (e.g., "OR 1 2")
    train_mds : list[list[int]]
        List of access lists for each training vector

    Returns
    -------
    np.ndarray
        Array of vector IDs that satisfy the filter
    """
    qualified_ids = []
    for i, md in enumerate(train_mds):
        if md and evaluate_predicate(filter_str, md):
            qualified_ids.append(i)
    return np.array(qualified_ids, dtype=np.int64)
