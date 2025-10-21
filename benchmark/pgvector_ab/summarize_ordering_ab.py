"""Summarize pgvector HNSW ordering A/B outputs (strict vs relaxed).

Reads artifacts produced by the A/B orchestrator under:
  output/pgvector/hnsw_ordering_ab/<dataset_key>_test<test_size>/{strict_order|relaxed_order}/

For single-label (CSV):
  - Extracts recall_at_k, query_lat_p95, query_qps

This tool prints side-by-side metrics only. No recommendation logic.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import fire


@dataclass
class Metrics:
    recall_at_k: Optional[float] = None
    p95: Optional[float] = None
    qps: Optional[float] = None


def _single_metrics(csv_path: Path) -> Metrics:
    if not csv_path.exists():
        return Metrics()
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return Metrics()
    row = rows[0]

    def _get(name: str) -> Optional[float]:
        try:
            return float(row[name])
        except Exception:
            return None

    return Metrics(
        recall_at_k=_get("recall_at_k"),
        p95=_get("query_lat_p95"),
        qps=_get("query_qps"),
    )


def _fmt(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


def _print_table(title: str, strict: Metrics, relaxed: Metrics) -> None:
    print(f"\n[ab][summary] {title}")
    print("mode           recall@k   p95_latency(s)   qps      Δrecall   speedup_p95")
    # deltas and speedups
    d_recall = None
    if strict.recall_at_k is not None and relaxed.recall_at_k is not None:
        d_recall = relaxed.recall_at_k - strict.recall_at_k
    sp95 = None
    if strict.p95 is not None and relaxed.p95 is not None and strict.p95 > 0:
        sp95 = (strict.p95 - relaxed.p95) / strict.p95
    print(
        f"strict_order   {_fmt(strict.recall_at_k)}     {_fmt(strict.p95)}          {_fmt(strict.qps)}      {'-':>7}   {'-':>10}"
    )
    print(
        f"relaxed_order  {_fmt(relaxed.recall_at_k)}     {_fmt(relaxed.p95)}          {_fmt(relaxed.qps)}      {_fmt(d_recall)}   {_fmt(sp95)}"
    )


class Summary:
    def run(
        self,
        *,
        dataset_key: str = "yfcc100m",
        test_size: float = 0.01,
        base_dir: str | None = None,
    ) -> None:
        """Summarize A/B artifacts for a dataset variant (no recommendation)."""
        root = Path(base_dir) if base_dir else Path("output/pgvector/hnsw_ordering_ab")
        dv_dir = root / f"{dataset_key}_test{test_size}"
        strict_dir = dv_dir / "strict_order"
        relaxed_dir = dv_dir / "relaxed_order"

        print(f"[ab][summary] dataset_key={dataset_key}, test_size={test_size}")
        print(f"[ab][summary] base_dir={dv_dir}")

        # Single-label CSVs
        strict_single = _single_metrics(strict_dir / "results.csv")
        relaxed_single = _single_metrics(relaxed_dir / "results.csv")
        _print_table("Single-label (overall results)", strict_single, relaxed_single)

        # No recommendation emitted.


if __name__ == "__main__":
    fire.Fire(Summary)
