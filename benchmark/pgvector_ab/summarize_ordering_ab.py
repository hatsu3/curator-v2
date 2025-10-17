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
from dataclasses import dataclass, asdict
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
        dataset_variant: str = "yfcc100m_1m",
        base_dir: str | None = None,
        # Recommendation knobs
        recall_tolerance: float = 0.001,
        p95_speedup_min: float = 0.05,
        # Artifacts
        write_json: bool = True,
        json_path: str | None = None,
    ) -> None:
        """Summarize A/B artifacts for a dataset variant."""
        def _coerce_bool(val, default=False):
            if isinstance(val, bool):
                return val
            if val is None:
                return default
            if isinstance(val, str):
                v = val.strip().lower()
                if v in {"1", "true", "yes", "y", "on"}:
                    return True
                if v in {"0", "false", "no", "n", "off"}:
                    return False
            return bool(val)
        write_json = _coerce_bool(write_json, True)
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

        def _rec_delta(a: Metrics, b: Metrics) -> Optional[float]:
            if a.recall_at_k is None or b.recall_at_k is None:
                return None
            return abs(b.recall_at_k - a.recall_at_k)

        def _p95_speedup(a: Metrics, b: Metrics) -> Optional[float]:
            if a.p95 is None or b.p95 is None or a.p95 <= 0:
                return None
            return (a.p95 - b.p95) / a.p95

        single_rec_delta = _rec_delta(strict_single, relaxed_single)
        complex_rec_delta = _rec_delta(strict_complex, relaxed_complex)
        single_speedup = _p95_speedup(strict_single, relaxed_single)
        complex_speedup = _p95_speedup(strict_complex, relaxed_complex)

        # Recommendation
        recommend: str
        reason: str
        have_single = (
            strict_single.recall_at_k is not None and strict_single.p95 is not None
            and relaxed_single.recall_at_k is not None and relaxed_single.p95 is not None
        )
        have_complex = (
            strict_complex.recall_at_k is not None and strict_complex.p95 is not None
            and relaxed_complex.recall_at_k is not None and relaxed_complex.p95 is not None
        )
        if not (have_single and have_complex):
            recommend = "insufficient_data"
            reason = "missing_metrics"
        else:
            rec_ok = (
                single_rec_delta is not None and complex_rec_delta is not None
                and single_rec_delta <= recall_tolerance
                and complex_rec_delta <= recall_tolerance
            )
            sp_ok = (
                single_speedup is not None and complex_speedup is not None
                and min(single_speedup, complex_speedup) >= p95_speedup_min
            )
            if rec_ok and sp_ok:
                recommend = "relaxed_order"
                reason = "recall_within_tolerance_and_p95_faster"
            else:
                recommend = "strict_order"
                if not rec_ok:
                    reason = "recall_gap"
                else:
                    reason = "no_tail_speedup"

        print(
            f"\n[ab][summary] recommendation: {recommend} (reason={reason}, "
            f"recall_tolerance={recall_tolerance}, p95_speedup_min={p95_speedup_min})"
        )

        if write_json:
            out_json = Path(json_path) if json_path else dv_dir / "ab_recommendation.json"
            out_json.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, object] = {
                "dataset_variant": dataset_variant,
                "thresholds": {
                    "recall_tolerance": float(recall_tolerance),
                    "p95_speedup_min": float(p95_speedup_min),
                },
                "single_label": {
                    "strict": asdict(strict_single),
                    "relaxed": asdict(relaxed_single),
                    "delta_recall": single_rec_delta,
                    "speedup_p95": single_speedup,
                },
                "complex": {
                    "strict": asdict(strict_complex),
                    "relaxed": asdict(relaxed_complex),
                    "delta_recall": complex_rec_delta,
                    "speedup_p95": complex_speedup,
                },
                "decision": {
                    "recommend": recommend,
                    "reason": reason,
                },
            }
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(f"[ab][summary] wrote {out_json}")


def main() -> None:
    fire.Fire(Summary)


if __name__ == "__main__":
    main()
