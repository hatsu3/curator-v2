"""Summarize pgvector HNSW ordering A/B outputs (strict vs relaxed).

Reads artifacts produced by the A/B orchestrator under:
  output/pgvector/hnsw_ordering_ab/<dataset_variant>/{strict_order|relaxed_order}/

For single-label (CSV):
  - Extracts recall_at_k, query_lat_p95, query_qps

For complex predicates (JSON with per-template results):
  - Aggregates recall_at_k and query_lat_p95 by taking the mean across templates

No recommendation is made; this is a read-only summarizer to aid manual choice.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def _complex_metrics(json_path: Path) -> Metrics:
    if not json_path.exists():
        return Metrics()
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    results = obj.get("results", {})
    rec_vals = []
    p95_vals = []
    for _, m in results.items():
        try:
            rec_vals.append(float(m.get("recall_at_k")))
        except Exception:
            pass
        try:
            p95_vals.append(float(m.get("query_lat_p95")))
        except Exception:
            pass

    def _mean(vals):
        return sum(vals) / len(vals) if vals else None

    return Metrics(
        recall_at_k=_mean(rec_vals),
        p95=_mean(p95_vals),
        qps=None,
    )


def _fmt(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


def _print_table(title: str, strict: Metrics, relaxed: Metrics) -> None:
    print(f"\n[ab][summary] {title}")
    print("mode           recall@k   p95_latency(s)   qps")
    print(
        f"strict_order   {_fmt(strict.recall_at_k)}     {_fmt(strict.p95)}          {_fmt(strict.qps)}"
    )
    print(
        f"relaxed_order  {_fmt(relaxed.recall_at_k)}     {_fmt(relaxed.p95)}          {_fmt(relaxed.qps)}"
    )


class Summary:
    def run(
        self,
        *,
        dataset_variant: str = "yfcc100m_1m",
        base_dir: str | None = None,
    ) -> None:
        """Summarize A/B artifacts for a dataset variant."""
        root = Path(base_dir) if base_dir else Path("output/pgvector/hnsw_ordering_ab")
        dv_dir = root / dataset_variant
        strict_dir = dv_dir / "strict_order"
        relaxed_dir = dv_dir / "relaxed_order"

        print(f"[ab][summary] dataset_variant={dataset_variant}")
        print(f"[ab][summary] base_dir={dv_dir}")
        # Single-label CSVs
        strict_single = _single_metrics(strict_dir / "results.csv")
        relaxed_single = _single_metrics(relaxed_dir / "results.csv")
        _print_table("Single-label (overall results)", strict_single, relaxed_single)

        # Complex JSONs
        strict_complex = _complex_metrics(strict_dir / "results.json")
        relaxed_complex = _complex_metrics(relaxed_dir / "results.json")
        _print_table("Complex predicates (AND/OR)", strict_complex, relaxed_complex)


def main() -> None:
    fire.Fire(Summary)


if __name__ == "__main__":
    main()

