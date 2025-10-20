from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import fire
import numpy as np
import torch

from benchmark.utils import get_dataset_config
from dataset import get_dataset, get_metadata


@dataclass
class EvalConfig:
    dataset_key: str = "yfcc100m"
    test_size: float = 0.01
    n_queries: int = 2000
    stratify_by_m: bool = False
    m_bins: str | None = None  # e.g., "1,1e2,1e3,1e4"
    s_candidates: str | int = "auto"  # or an int
    k_tail: int = 20
    batch_q: int = 256
    device: str = "auto"  # auto|cpu|cuda
    shuffle_seed: int = 42
    ci: bool = True
    ci_bootstrap: int = 200
    ci_alpha: float = 0.05
    seed: int = 42


def _build_label_to_train_indices(train_mds: list[list[int]]) -> dict[int, np.ndarray]:
    label_to_indices: dict[int, list[int]] = {}
    for i, labels in enumerate(train_mds):
        for label in labels:
            label_to_indices.setdefault(int(label), []).append(i)
    return {lbl: np.asarray(ix, dtype=np.int64) for lbl, ix in label_to_indices.items()}


def _min_l2_to_set(x: np.ndarray, Y: np.ndarray) -> float:
    # x: (D,), Y: (M, D)
    # returns min ||x - y||_2 over rows in Y
    diff = Y - x[None, :]
    # Use squared norms then sqrt of min to reduce temporary allocations
    d2 = np.einsum("ij,ij->i", diff, diff, dtype=np.float64)
    return float(np.sqrt(d2.min()))


def _synthesize_shuffled_metadata_like_revision(
    train_mds: list[list[int]], test_mds: list[list[int]], seed: int = 42
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Mirror revision's SkewnessDataset synthesis logic:
    - Preserve distribution of labels-per-vector.
    - Preserve label frequency distribution over vector-label pairs.
    - Sample without replacement per vector.
    Returns new (train_mds_shuf, test_mds_shuf).
    """
    rng = np.random.default_rng(seed)

    # distribution of number of labels per vector (from training split)
    n_labels_per_vec = {}
    for mds in train_mds:
        k = len(mds)
        n_labels_per_vec[k] = n_labels_per_vec.get(k, 0) + 1
    nlpv_vals = np.array(list(n_labels_per_vec.keys()))
    nlpv_probs = np.array(list(n_labels_per_vec.values()), dtype=np.float64)
    nlpv_probs /= nlpv_probs.sum()

    # label frequency over vector-label pairs
    label_freq: dict[int, int] = {}
    for mds in train_mds:
        for lbl in mds:
            label_freq[lbl] = label_freq.get(lbl, 0) + 1
    label_vals = np.array(list(label_freq.keys()), dtype=np.int64)
    label_probs = np.array(list(label_freq.values()), dtype=np.float64)
    label_probs /= label_probs.sum()

    def synthesize_for_split(n_vectors: int) -> list[list[int]]:
        out = []
        for _ in range(n_vectors):
            n_labels = int(rng.choice(nlpv_vals, p=nlpv_probs))
            if n_labels == 0:
                out.append([])
                continue
            n_labels = min(n_labels, len(label_vals))
            # sample without replacement following label_probs
            chosen = rng.choice(label_vals, size=n_labels, replace=False, p=label_probs)
            out.append([int(x) for x in chosen])
        return out

    train_mds_shuf = synthesize_for_split(len(train_mds))
    test_mds_shuf = synthesize_for_split(len(test_mds))
    return train_mds_shuf, test_mds_shuf


def _sample_queries(
    test_mds: Sequence[Sequence[int]],
    label_to_train_ix: dict[int, np.ndarray],
    n_queries: int,
    stratify_by_m: bool = False,
    m_bins: list[int] | None = None,
    seed: int = 42,
) -> list[tuple[int, int, int]]:
    """Return a list of (test_idx, label, m_size) sampled uniformly or stratified by m.
    m_size = |X_label| from label_to_train_ix.
    """
    all_pairs: list[tuple[int, int, int]] = []
    for t_idx, labels in enumerate(test_mds):
        for lbl in labels:
            m = int(label_to_train_ix.get(int(lbl), np.array([], dtype=np.int64)).size)
            if m > 0:
                all_pairs.append((t_idx, int(lbl), m))

    rng = np.random.default_rng(seed)
    if not stratify_by_m:
        if len(all_pairs) <= n_queries:
            return all_pairs
        idxs = rng.choice(len(all_pairs), size=n_queries, replace=False)
        return [all_pairs[i] for i in idxs]

    # stratified by m using bins
    if not m_bins:
        # default log-spaced bins if not provided
        m_vals = np.array([m for _, _, m in all_pairs])
        lo, hi = max(1, int(m_vals.min())), int(m_vals.max())
        m_bins = [1]
        step = max(2, int(10 ** (np.log10(hi) / 4)))
        while m_bins[-1] < hi:
            m_bins.append(min(hi, m_bins[-1] * step))
    # assign to bins
    bins: list[list[int]] = [[] for _ in range(len(m_bins))]
    def bin_index(m: int) -> int:
        for i, b in enumerate(m_bins):
            if m <= b:
                return i
        return len(m_bins) - 1
    for i, (_, _, m) in enumerate(all_pairs):
        bins[bin_index(m)].append(i)

    # sample per bin proportional to bin sizes
    per_bin = [max(1, int(n_queries * (len(b) / max(1, len(all_pairs))))) for b in bins]
    chosen: list[int] = []
    for bin_idxs, k in zip(bins, per_bin):
        if not bin_idxs:
            continue
        if len(bin_idxs) <= k:
            chosen.extend(bin_idxs)
        else:
            chosen.extend(rng.choice(bin_idxs, size=k, replace=False).tolist())
    # trim to n_queries
    if len(chosen) > n_queries:
        chosen = rng.choice(chosen, size=n_queries, replace=False).tolist()
    return [all_pairs[i] for i in chosen]


def run(
    dataset_key: str = EvalConfig.dataset_key,
    test_size: float = EvalConfig.test_size,
    n_queries: int = EvalConfig.n_queries,
    stratify_by_m: bool = EvalConfig.stratify_by_m,
    m_bins: str | None = EvalConfig.m_bins,
    s_candidates: str | int = EvalConfig.s_candidates,
    k_tail: int = EvalConfig.k_tail,
    batch_q: int = EvalConfig.batch_q,
    device: str = EvalConfig.device,
    shuffle_seed: int = EvalConfig.shuffle_seed,
    seed: int = EvalConfig.seed,
):
    """
    Compute the Query Correlation metric C(D, Q) for both original and shuffled datasets.

    C(D, Q) = E_{(x_i, p_i) in Q} [ E_{R_i}[ g(x_i, R_i) ] - g(x_i, X_{p_i}) ]

    where g(x, S) = min_{y in S} dist(x, y) and |R_i| = |X_{p_i}| sampled uniformly from X.
    Prints two lines: original and shuffled averages.
    """
    # Resolve dataset config without triggering ground-truth computation
    ds_cfg, _ = get_dataset_config(dataset_key, test_size=test_size)

    # Load vectors
    train_vecs, test_vecs, _ = get_dataset(
        dataset_name=ds_cfg.dataset_name, **ds_cfg.dataset_params
    )

    # Load metadata (label lists per vector)
    train_mds, test_mds = get_metadata(
        synthesized=ds_cfg.synthesize_metadata,
        dataset_name=ds_cfg.dataset_name,
        **ds_cfg.metadata_params,
    )

    # Build inverse index: label -> train vector indices
    label_to_train_ix = _build_label_to_train_indices(train_mds)

    # Prepare shuffled (revision-like) metadata for train split only
    shuffled_train_mds, _ = _synthesize_shuffled_metadata_like_revision(
        train_mds, test_mds, seed=shuffle_seed
    )
    label_to_train_ix_shuf = _build_label_to_train_indices(shuffled_train_mds)

    # Sample queries
    m_bins_list = (
        [int(float(x)) for x in m_bins.split(",")] if (m_bins is not None) else None
    )
    sampled = _sample_queries(
        test_mds,
        label_to_train_ix,
        n_queries,
        stratify_by_m=stratify_by_m,
        m_bins=m_bins_list,
        seed=seed,
    )
    if not sampled:
        print("Average Query Correlation (original): NaN (no valid queries)")
        print("Average Query Correlation (shuffled): NaN (no valid queries)")
        return

    # Torch setup
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    train_pth = torch.from_numpy(train_vecs).to(dev)
    test_pth = torch.from_numpy(test_vecs).to(dev)

    # Determine s for candidate pool
    if isinstance(s_candidates, str) and s_candidates == "auto":
        m_vals = np.array([m for _, _, m in sampled])
        m90 = float(np.quantile(m_vals, 0.9))
        s = int(max(1024, min(8192, np.ceil(k_tail * (m90 + 1)))))
    else:
        s = int(s_candidates)

    rng = np.random.default_rng(seed)
    n_train = train_vecs.shape[0]

    def compute_batch_estimates(batch: list[tuple[int, int, int]]):
        # X batch
        idx = np.array([t for t, _, _ in batch], dtype=np.int64)
        X = test_pth[idx]
        # Candidate pool shared for the batch
        cand_idx = torch.from_numpy(rng.choice(n_train, size=s, replace=False)).to(dev)
        Y = train_pth[cand_idx]
        # Distances to candidate pool
        D = torch.cdist(X, Y)
        D_sorted, _ = torch.sort(D, dim=1)
        # k-th index for quantile estimator per row
        m_arr = np.array([m for _, _, m in batch], dtype=np.int64)
        ks = np.clip(np.ceil(s * (1.0 / (m_arr + 1.0))).astype(np.int64), 1, s)
        ks_t = torch.from_numpy(ks - 1).to(dev)
        rows = torch.arange(D_sorted.shape[0], device=dev)
        g_rand = D_sorted[rows, ks_t]

        # For shuffled: recompute k based on shuffled label cardinality
        m_arr_shuf = np.array(
            [
                int(label_to_train_ix_shuf.get(lbl, np.array([], dtype=np.int64)).size)
                for _, lbl, _ in batch
            ],
            dtype=np.int64,
        )
        m_arr_shuf = np.clip(m_arr_shuf, 1, n_train)
        ks_shuf = np.clip(
            np.ceil(s * (1.0 / (m_arr_shuf + 1.0))).astype(np.int64), 1, s
        )
        ks_shuf_t = torch.from_numpy(ks_shuf - 1).to(dev)
        g_rand_shuf = D_sorted[rows, ks_shuf_t]

        # Exact g(x, X_p) for original and shuffled
        g_true_list = []
        g_shuf_list = []
        for j, (_, lbl, _) in enumerate(batch):
            # original
            t_ix = label_to_train_ix.get(lbl)
            Y_lbl = train_pth[torch.from_numpy(t_ix).to(dev)]
            d = torch.cdist(X[j : j + 1], Y_lbl)
            g_true_list.append(d.min().item())
            # shuffled
            t_ix_s = label_to_train_ix_shuf.get(lbl)
            if t_ix_s is None or t_ix_s.size == 0:
                g_shuf_list.append(float("nan"))
            else:
                Y_lbl_s = train_pth[torch.from_numpy(t_ix_s).to(dev)]
                d_s = torch.cdist(X[j : j + 1], Y_lbl_s)
                g_shuf_list.append(d_s.min().item())

        g_true = np.array(g_true_list, dtype=np.float64)
        g_shuf = np.array(g_shuf_list, dtype=np.float64)
        g_rand_np = g_rand.detach().cpu().numpy().astype(np.float64)
        return g_rand_np - g_true, g_rand_shuf.detach().cpu().numpy() - g_shuf

    deltas = []
    deltas_shuf = []
    for i in range(0, len(sampled), batch_q):
        batch = sampled[i : i + batch_q]
        d1, d2 = compute_batch_estimates(batch)
        deltas.extend(d1.tolist())
        deltas_shuf.extend(d2.tolist())

    # Filter NaNs (may arise if a shuffled label has zero members)
    deltas = np.array([x for x in deltas if np.isfinite(x)], dtype=np.float64)
    deltas_shuf = np.array(
        [x for x in deltas_shuf if np.isfinite(x)], dtype=np.float64
    )
    orig = float(np.mean(deltas)) if deltas.size else float("nan")
    shuf = float(np.mean(deltas_shuf)) if deltas_shuf.size else float("nan")
    if EvalConfig.ci:
        def _bootstrap_ci(x: np.ndarray, B: int, alpha: float) -> tuple[float, float]:
            if x.size == 0:
                return float("nan"), float("nan")
            rng_b = np.random.default_rng(seed + 123)
            means = []
            n = x.size
            for _ in range(B):
                idx = rng_b.integers(0, n, size=n)
                means.append(float(np.mean(x[idx])))
            lo = np.quantile(means, alpha / 2)
            hi = np.quantile(means, 1 - alpha / 2)
            return float(lo), float(hi)

        lo_o, hi_o = _bootstrap_ci(deltas, EvalConfig.ci_bootstrap, EvalConfig.ci_alpha)
        lo_s, hi_s = _bootstrap_ci(
            deltas_shuf, EvalConfig.ci_bootstrap, EvalConfig.ci_alpha
        )
        print(
            f"Average Query Correlation (original): {orig:.6f}  (95% CI: {lo_o:.6f}, {hi_o:.6f})"
        )
        print(
            f"Average Query Correlation (shuffled): {shuf:.6f}  (95% CI: {lo_s:.6f}, {hi_s:.6f})"
        )
    else:
        print(f"Average Query Correlation (original): {orig:.6f}")
        print(f"Average Query Correlation (shuffled): {shuf:.6f}")


if __name__ == "__main__":
    fire.Fire()
