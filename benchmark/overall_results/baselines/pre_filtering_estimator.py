import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import fire
import numpy as np
import pandas as pd

from benchmark.utils import get_dataset_config, load_dataset
from indexes.pre_filtering import PreFilteringIndex


def _set_single_thread_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("BLIS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@dataclass
class LoadedDataset:
    train_vecs: np.ndarray
    test_vecs: np.ndarray
    train_mds: List[List[int]]
    test_mds: List[List[int]]
    label_selectivities: dict


def _load_real_dataset(dataset_key: str, test_size: float) -> LoadedDataset:
    ds_cfg, _ = get_dataset_config(dataset_key=dataset_key, test_size=test_size)
    train_vecs, test_vecs, train_mds, test_mds, _gt, _labels = load_dataset(ds_cfg)

    # Build selectivity map
    label_counts = {}
    for access_list in train_mds:
        for label in access_list:
            label_counts[label] = label_counts.get(label, 0) + 1
    n_train = len(train_vecs)
    label_selectivities = {lbl: cnt / n_train for lbl, cnt in label_counts.items()}

    return LoadedDataset(
        train_vecs=train_vecs,
        test_vecs=test_vecs,
        train_mds=train_mds,
        test_mds=test_mds,
        label_selectivities=label_selectivities,
    )


def _build_prefilter_index(train_vecs: np.ndarray, train_mds: List[List[int]]) -> PreFilteringIndex:
    index = PreFilteringIndex()
    labels = list(range(len(train_vecs)))
    index.batch_create(train_vecs, labels, train_mds)
    return index


def _label_cardinality_train(train_mds: List[List[int]], label: int) -> int:
    cnt = 0
    for access_list in train_mds:
        for lbl in access_list:
            if lbl == label:
                cnt += 1
    return cnt


def _find_test_occurrences(test_mds: List[List[int]], label: int) -> List[int]:
    idxs = []
    for i, labels in enumerate(test_mds):
        if label in labels:
            idxs.append(i)
    return idxs


def _measure_label_latency(
    index: PreFilteringIndex,
    test_vecs: np.ndarray,
    test_mds: List[List[int]],
    label: int,
    k: int = 10,
    max_queries: int = 64,
) -> Tuple[int, float, int]:
    occurrences = _find_test_occurrences(test_mds, label)
    if not occurrences:
        return 0, 0.0, 0

    # Measure per-occurrence latency for up to max_queries
    lats = []
    for i in occurrences[:max_queries]:
        x = test_vecs[i]
        t0 = time.perf_counter()
        _ = index.query(x, k=k, tenant_id=label)
        lats.append(time.perf_counter() - t0)

    mean_lat = float(np.mean(lats)) if lats else 0.0
    return len(lats), mean_lat, len(occurrences)


def _fit_linear_ab(points: Iterable[Tuple[int, float]]) -> Tuple[float, float]:
    # points: (Nqualified, mean_latency)
    xs, ys = zip(*points)
    coeffs = np.polyfit(np.array(xs, dtype=float), np.array(ys, dtype=float), 1)
    a, b = float(coeffs[0]), float(coeffs[1])
    return a, b


def _predict_all_latencies(
    train_mds: List[List[int]],
    test_mds: List[List[int]],
    a: float,
    b: float,
) -> Tuple[List[float], List[float]]:
    # Pre-compute exact train cardinality per label
    label_counts = {}
    for access_list in train_mds:
        for lbl in access_list:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    lats: List[float] = []
    recs: List[float] = []
    for labels in test_mds:
        for lbl in labels:
            n = label_counts.get(lbl, 0)
            lats.append(a * n + b)
            recs.append(1.0)
    return lats, recs


def run(
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    k: int = 10,
    n_calib_labels: int = 8,
    max_queries_per_label: int = 64,
    output_root: str = "output/overall_results2/pre_filtering",
):
    """
    Calibrate pre-filtering latency (intended to be invoked under single-core
    affinity via `taskset -c <core>`) and emit per-occurrence predicted latencies
    aligned with the multi-tenant test ordering used by overall-results.

    Parameters
    ----------
    dataset_key : str
        Dataset key recognized by benchmark.utils.get_dataset_config (e.g.,
        "yfcc100m-10m", "arxiv-large-10").
    test_size : float
        Fraction of data used for test split when loading the dataset config.
    k : int
        Top-k used when invoking PreFilteringIndex.query during calibration.
        (k does not affect predicted latency since we fit latency vs. qualified count.)
    n_calib_labels : int
        Number of labels sampled across the selectivity spectrum for calibration.
        More labels generally improves the robustness of the linear fit (a, b) at
        the cost of longer calibration time.
    max_queries_per_label : int
        Maximum number of query occurrences per calibration label to average over.
        Bound this to reduce calibration time while smoothing noise in per-occurrence
        timing; values between 32-128 are typically sufficient.
    output_root : str
        Root directory where results will be written, following the structure
        "<output_root>/<dataset_key>_test<test_size>/results.csv". For safe testing,
        set this under a scratch path like "curator-v2/tmp/pre_filtering".

    Output
    ------
    Writes a CSV with two list-valued columns `query_latencies` and `query_recalls`
    (recall fixed to 1.0) compatible with overall-results preprocessing.
    """
    _set_single_thread_env()

    print(f"[pre_filtering_estimator] Loading dataset {dataset_key} (test_size={test_size})")
    ds = _load_real_dataset(dataset_key, test_size)

    print("[pre_filtering_estimator] Building PreFilteringIndex ...")
    index = _build_prefilter_index(ds.train_vecs, ds.train_mds)

    # Choose calibration labels across the selectivity spectrum
    print("[pre_filtering_estimator] Sampling labels for calibration ...")
    labels = sorted(ds.label_selectivities.keys(), key=lambda l: ds.label_selectivities[l])
    if not labels:
        raise RuntimeError("No labels found in dataset metadata")

    # Evenly spaced picks across range
    picks: List[int] = []
    for q in np.linspace(0.05, 0.95, n_calib_labels):
        idx = int(q * (len(labels) - 1))
        picks.append(int(labels[idx]))
    picks = sorted(set(picks))

    calib_points: List[Tuple[int, float]] = []
    used = 0
    for lbl in picks:
        n_occ, mean_lat, total_occ = _measure_label_latency(
            index, ds.test_vecs, ds.test_mds, lbl, k=k, max_queries=max_queries_per_label
        )
        n_train = _label_cardinality_train(ds.train_mds, lbl)
        if n_occ > 0 and mean_lat > 0:
            calib_points.append((n_train, mean_lat))
            used += 1
            print(f"  label={lbl} Ntrain={n_train} occ={n_occ}/{total_occ} mean_lat={mean_lat*1e3:.3f} ms")
        else:
            print(f"  skipped label={lbl} due to no test occurrences or zero latency")

    if len(calib_points) < 2:
        raise RuntimeError("Insufficient calibration points; try increasing test_size or n_calib_labels")

    a, b = _fit_linear_ab(calib_points)
    print(f"[pre_filtering_estimator] Fitted latency model: latency ≈ {a:.3e} * N + {b:.3e} (seconds)")

    # Predict full arrays (aligned with test_mds ordering)
    pred_lats, pred_recs = _predict_all_latencies(ds.train_mds, ds.test_mds, a, b)

    # Save results.csv in overall_results2 folder structure
    dataset_subdir = f"{dataset_key}_test{test_size}"
    out_dir = Path(output_root) / dataset_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.csv"

    print(f"[pre_filtering_estimator] Writing results to {out_path}")
    df = pd.DataFrame(
        {
            "query_latencies": [json.dumps(pred_lats)],
            "query_recalls": [json.dumps(pred_recs)],
        }
    )
    df.to_csv(out_path, index=False)

    # Save fitted linear model parameters
    model_json = {
        "a": a,
        "b": b,
        "dataset_key": dataset_key,
        "test_size": test_size,
    }
    (out_dir / "linreg.json").write_text(json.dumps(model_json, indent=2))

    # Produce calibration plot PDF: latency vs cardinality and vs selectivity with fitted line
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_train_total = len(ds.train_vecs)
        # Scatter points
        xN = np.array([p[0] for p in calib_points], dtype=float)
        y_ms = np.array([p[1] for p in calib_points], dtype=float) * 1e3
        xSel = xN / float(n_train_total)

        # Fitted lines
        xN_line = np.linspace(max(1.0, xN.min()), max(1.0, xN.max()), 100)
        y_line_ms = (a * xN_line + b) * 1e3
        xSel_line = xN_line / float(n_train_total)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
        ax1.scatter(xN, y_ms, s=18, c="tab:red", label="measured")
        ax1.plot(xN_line, y_line_ms, c="tab:blue", label="fit")
        ax1.set_xlabel("Qualified Count (N)")
        ax1.set_ylabel("Latency (ms)")
        ax1.grid(True, alpha=0.4)
        ax1.legend(frameon=False)

        ax2.scatter(xSel, y_ms, s=18, c="tab:red", label="measured")
        ax2.plot(xSel_line, y_line_ms, c="tab:blue", label="fit")
        ax2.set_xlabel("Selectivity")
        ax2.set_ylabel("Latency (ms)")
        ax2.grid(True, alpha=0.4)

        fig.tight_layout()
        plot_path = out_dir / "calibration_fit.pdf"
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"[pre_filtering_estimator] Wrote calibration plot to {plot_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[pre_filtering_estimator] Calibration plot failed: {e}")


if __name__ == "__main__":
    fire.Fire({"run": run})
