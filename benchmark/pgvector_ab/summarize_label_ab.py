"""Summarize label-modeling A/B results and storage into a single report.

Inputs (per dataset_variant, default: yfcc100m_1m):
- output/pgvector/label_ab/<variant>/int_array/{hnsw|ivf}/results.csv
- output/pgvector/label_ab/<variant>/boolean/{hnsw|ivf}/results.csv
- output/pgvector/label_ab/<variant>/storage.json (optional)

Outputs:
- output/pgvector/label_ab/<variant>/summary.csv
- output/pgvector/label_ab/<variant>/summary.json (includes recommendation)

Recommendation heuristic (simple): pick the schema+strategy with the
lowest p50 latency; include storage bytes if available.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json
import os

import fire
import pandas as pd


def _variant_dir(variant: str) -> Path:
    return Path("output/pgvector/label_ab") / variant


def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[summarize] missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[summarize][WARN] failed to read {path}: {e}")
        return None


class LabelABSummarizer:
    def summarize(
        self,
        *,
        dataset_variant: str = "yfcc100m_1m",
        dry_run: bool = True,
    ) -> None:
        base = _variant_dir(dataset_variant)
        pairs = [
            ("int_array", "hnsw"),
            ("int_array", "ivf"),
            ("boolean", "hnsw"),
            ("boolean", "ivf"),
        ]
        rows: list[Dict[str, Any]] = []
        for schema, strategy in pairs:
            csv_path = base / schema / strategy / "results.csv"
            print("[summarize] scan:", csv_path)
            df = _read_csv_safe(csv_path)
            if df is None or df.empty:
                continue
            rec = df.iloc[0].to_dict()
            rec["schema"] = schema
            rec["strategy"] = strategy
            rows.append(rec)

        if dry_run:
            print("[summarize] preview only. rows found:", len(rows))
            print("[summarize] planned outputs:")
            print("  ", base / "summary.csv")
            print("  ", base / "summary.json")
            return

        base.mkdir(parents=True, exist_ok=True)
        out_csv = base / "summary.csv"
        out_json = base / "summary.json"
        if not rows:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"error": "no result rows found", "dataset_variant": dataset_variant}, f, indent=2)
            print("[summarize] wrote", out_json)
            return

        df_all = pd.DataFrame(rows)
        df_all.to_csv(out_csv, index=False)

        # Recommendation: choose min p50 latency
        pick = df_all.loc[df_all["query_lat_p50"].astype(float).idxmin()]
        reco = {
            "schema": str(pick.get("schema")),
            "strategy": str(pick.get("strategy")),
            "query_lat_p50": float(pick.get("query_lat_p50", 0.0)),
            "query_qps": float(pick.get("query_qps", 0.0)),
            "recall_at_k": float(pick.get("recall_at_k", 0.0)),
        }

        storage_json = base / "storage.json"
        storage: Dict[str, Any] = {}
        if storage_json.exists():
            try:
                storage = json.loads(storage_json.read_text(encoding="utf-8"))
            except Exception as e:  # pragma: no cover
                print(f"[summarize][WARN] failed to load storage.json: {e}")
        summary = {
            "dataset_variant": dataset_variant,
            "rows": rows,
            "recommendation": reco,
            "storage": storage,
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        print("[summarize] wrote", out_csv)
        print("[summarize] wrote", out_json)


def main() -> None:
    fire.Fire(LabelABSummarizer)


if __name__ == "__main__":
    main()

