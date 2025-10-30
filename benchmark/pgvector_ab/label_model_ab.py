"""Label modeling A/B orchestrator (INT[] + GIN vs boolean columns).

This CLI previews (and optionally runs) single-label experiments on YFCC 1M
for two schema variants:
  - int_array: tags INT[] + GIN(tags)
  - boolean: wide-table boolean columns label_<id>

It prints exact baseline commands to run and canonical output paths:
  output/pgvector/label_ab/yfcc100m_1m/{int_array|boolean}/{ivf|hnsw}/results.csv

Optional: in non-dry-run mode it records storage metrics as storage.json using
helpers from scripts.pgvector.admin (GIN size for int_array; table size for
boolean).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import json
import os

import fire

try:
    from scripts.pgvector import admin  # type: ignore
except Exception:
    admin = None  # type: ignore


def _outdir(schema: str, strategy: str, dataset_variant: str) -> Path:
    return (
        Path("output/pgvector/label_ab")
        / dataset_variant
        / schema
        / ("hnsw" if strategy == "hnsw" else "ivf")
    )


@dataclass
class RunArgs:
    # Datasets
    dataset_variant: str
    dataset_key: str
    test_size: float
    k: int
    # HNSW
    m: int
    ef_construction: int
    ef_search: int
    # IVF
    lists: int
    probes: int
    # DSNs
    dsn_int: Optional[str]
    dsn_bool: Optional[str]
    # Control
    dry_run: bool


class LabelModelAB:
    def run(
        self,
        *,
        dataset_variant: str = "yfcc100m_1m",
        dataset_key: str = "yfcc100m",
        test_size: float = 0.01,
        k: int = 10,
        # HNSW params
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
        # IVF params
        lists: int = 200,
        probes: int = 16,
        # DSNs for the two schemas
        dsn_int: str | None = None,
        dsn_bool: str | None = None,
        # Control
        dry_run: bool = True,
    ) -> None:
        """Preview A/B commands and output paths for label modeling comparison.

        Use two databases: one for INT[] + GIN and one for boolean columns.
        """
        args = RunArgs(
            dataset_variant=dataset_variant,
            dataset_key=dataset_key,
            test_size=test_size,
            k=k,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            lists=lists,
            probes=probes,
            dsn_int=dsn_int or os.environ.get("PG_DSN_INT"),
            dsn_bool=dsn_bool or os.environ.get("PG_DSN_BOOL"),
            dry_run=dry_run,
        )

        if not args.dsn_int or not args.dsn_bool:
            print(
                "[label_ab][WARN] Both dsn_int and dsn_bool are recommended. "
                "You can also provide PG_DSN_INT / PG_DSN_BOOL via env."
            )

        # Build commands
        runs = [
            ("int_array", "hnsw", args.dsn_int),
            ("int_array", "ivf", args.dsn_int),
            ("boolean", "hnsw", args.dsn_bool),
            ("boolean", "ivf", args.dsn_bool),
        ]
        planned: Dict[str, Dict[str, Any]] = {}
        for schema, strategy, dsn in runs:
            out_dir = _outdir(schema, strategy, args.dataset_variant)
            out_csv = out_dir / "results.csv"
            base = (
                "python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single "
                f"--strategy {strategy} --iter_search true --schema {schema} "
                f"--dataset_key {args.dataset_key} --test_size {args.test_size} --k {args.k} "
                f"--output_path {out_csv}"
            )
            if strategy == "hnsw":
                base += f" --m {args.m} --ef_construction {args.ef_construction} --ef_search {args.ef_search}"
            else:
                base += f" --lists {args.lists} --probes {args.probes}"
            if dsn:
                base += f" --dsn {dsn}"
            planned[f"{schema}_{strategy}"] = {
                "output": str(out_csv),
                "cmd": base,
            }

        print("[label_ab] Planned outputs:")
        for key, val in planned.items():
            print(f"  [{key}] {val['output']}")
        print("[label_ab] Commands to run:")
        for key, val in planned.items():
            print(f"--- {key} ---\n{val['cmd']}")

        # Optional: collect and write storage metrics in non-dry-run mode
        if not args.dry_run and admin is not None:
            metrics_dir = Path("output/pgvector/label_ab") / args.dataset_variant
            metrics_dir.mkdir(parents=True, exist_ok=True)
            storage: Dict[str, Any] = {
                "dataset_variant": args.dataset_variant,
                "dataset_key": args.dataset_key,
                "yfcc_dim": 192,
            }
            try:
                if args.dsn_int:
                    storage["int_array_gin_bytes"] = admin.get_relation_size(
                        args.dsn_int, "items_tags_gin"
                    )
                if args.dsn_bool:
                    storage["boolean_table_total_bytes"] = admin.get_total_relation_size(
                        args.dsn_bool, "items"
                    )
            except Exception as e:  # pragma: no cover
                print(f"[label_ab][WARN] storage metrics collection failed: {e}")
            with open(metrics_dir / "storage.json", "w", encoding="utf-8") as f:
                json.dump(storage, f, indent=2, sort_keys=True)
            print("[label_ab] Wrote storage metrics to", metrics_dir / "storage.json")


def main() -> None:
    fire.Fire(LabelModelAB)


if __name__ == "__main__":
    main()
