"""pgvector insert A/B sweeper and summarizer.

Runs single-thread insert benchmarks across strategies and durability modes and
writes per-run artifacts under:
  output/pgvector/insert_ab/<dataset_key>/{hnsw|ivf|gin}/{durable|non_durable}/run.{json,csv}

Also aggregates a summary across all runs into:
  output/pgvector/insert_ab/<dataset_key>/summary.{json,csv}

This module invokes the existing insert benchmark implementation from
`scripts.pgvector.load_dataset` and, for IVF, can build the index before runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import os
import csv

import fire


def _ab_dir(dataset_key: str, idx_tag: str, durability: str) -> Path:
    return Path("output/pgvector/insert_ab") / dataset_key / idx_tag / durability


def _run_json_path(dataset_key: str, strategy: str, durability: str) -> Path:
    idx_tag = "gin" if strategy == "prefilter" else strategy
    return _ab_dir(dataset_key, idx_tag, durability) / "run.json"


@dataclass
class SweepArgs:
    dsn: Optional[str]
    dataset: str
    dataset_key: str
    test_size: float
    dim: int
    m: int
    efc: int
    lists: int
    limit: Optional[int]
    build_ivf: bool
    dry_run: bool


class InsertAB:
    def run(
        self,
        *,
        dsn: str | None = None,
        dataset: str = "yfcc100m",
        dataset_key: str = "yfcc100m",
        test_size: float = 0.01,
        dim: int = 192,
        # HNSW
        m: int = 32,
        efc: int = 64,
        # IVF
        lists: int = 200,
        # Control
        limit: int | None = None,
        build_ivf: bool = True,
        dry_run: bool = True,
    ) -> None:
        """Run or preview insert A/B sweeps and summarize outputs.

        Strategies: hnsw, ivf, prefilter  | Durability: durable, non_durable (UNLOGGED)
        """
        args = SweepArgs(
            dsn=dsn,
            dataset=dataset,
            dataset_key=dataset_key,
            test_size=test_size,
            dim=dim,
            m=m,
            efc=efc,
            lists=lists,
            limit=limit,
            build_ivf=build_ivf,
            dry_run=dry_run,
        )

        strategies = ["hnsw", "ivf", "prefilter"]
        durability_modes: List[Tuple[str, Dict[str, Any]]] = [
            ("durable", {"unlogged": False, "sync_commit_off": False}),
            ("non_durable", {"unlogged": True, "sync_commit_off": False}),
        ]

        # Preview commands and planned outputs
        print("[insert_ab] Planned runs:")
        for strat in strategies:
            for label, flags in durability_modes:
                idx_tag = "gin" if strat == "prefilter" else strat
                out_dir = _ab_dir(args.dataset_key, idx_tag, label)
                cmd = (
                    "python -m scripts.pgvector.load_dataset insert_bench "
                    f"--dsn {args.dsn or '<DSN>'} --dataset {args.dataset} --dataset_key {args.dataset_key} "
                    f"--dim {args.dim} --test_size {args.test_size} --strategy {strat}"
                )
                if strat == "hnsw":
                    cmd += f" --m {args.m} --efc {args.efc}"
                if strat == "ivf":
                    cmd += f" --lists {args.lists}"
                if args.limit is not None:
                    cmd += f" --limit {int(args.limit)}"
                if flags.get("unlogged"):
                    cmd += " --unlogged true"
                if flags.get("sync_commit_off"):
                    cmd += " --sync_commit_off true"
                print(f"  - {strat}/{label} -> {out_dir}/run.{{json,csv}}\n    {cmd}")

        if args.dry_run:
            print("[insert_ab] Dry-run: preview only. Not executing.")
            return

        # Real execution path
        from scripts.pgvector.load_dataset import LoadDataset  # type: ignore
        try:
            from scripts.pgvector import admin  # type: ignore
        except Exception:
            admin = None  # type: ignore

        ld = LoadDataset()
        for strat in strategies:
            for label, flags in durability_modes:
                # IVF: build before inserts if requested
                if strat == "ivf" and args.build_ivf and admin is not None:
                    try:
                        admin.create_index(
                            args.dsn or "",
                            index="ivf",
                            dim=args.dim,
                            lists=args.lists,
                            dry_run=False,
                        )
                    except Exception as e:  # pragma: no cover
                        print(f"[insert_ab][WARN] IVF build failed or admin missing: {e}")

                # Execute insert bench
                ld.insert_bench(
                    dsn=args.dsn,
                    dataset=args.dataset,
                    dataset_key=args.dataset_key,
                    dim=args.dim,
                    test_size=args.test_size,
                    strategy=strat,
                    unlogged=bool(flags.get("unlogged", False)),
                    sync_commit_off=bool(flags.get("sync_commit_off", False)),
                    m=args.m if strat == "hnsw" else None,
                    efc=args.efc if strat == "hnsw" else None,
                    lists=args.lists if strat == "ivf" else None,
                    limit=args.limit,
                    dry_run=False,
                )

        # Aggregate summary
        base_dir = Path("output/pgvector/insert_ab") / args.dataset_key
        rows: List[Dict[str, Any]] = []
        for strat in strategies:
            for label, _flags in durability_modes:
                run_path = _run_json_path(args.dataset_key, strat, label)
                if not run_path.exists():
                    print("[insert_ab] missing run:", run_path)
                    continue
                try:
                    with open(run_path, "r", encoding="utf-8") as f:
                        rec = json.load(f)
                    rec["strategy"] = strat
                    rec["durability"] = label
                    rows.append(rec)
                except Exception as e:
                    print(f"[insert_ab][WARN] failed to parse {run_path}: {e}")

        base_dir.mkdir(parents=True, exist_ok=True)
        out_csv = base_dir / "summary.csv"
        out_json = base_dir / "summary.json"
        # Write CSV
        if rows:
            fieldnames = sorted({k for r in rows for k in r.keys()})
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        # Write JSON
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "dataset_key": args.dataset_key,
                "test_size": args.test_size,
                "runs": rows,
            }, f, indent=2, sort_keys=True)
        print("[insert_ab] wrote", out_csv)
        print("[insert_ab] wrote", out_json)


def main() -> None:
    fire.Fire(InsertAB)


if __name__ == "__main__":
    main()

