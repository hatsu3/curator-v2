"""A/B orchestrator for pgvector HNSW ordering modes (scaffold).

This scaffold provides a thin CLI to preview the commands and output paths
for running A/B comparisons between `strict_order` and `relaxed_order` (with
strict post-sort) for:
- Single-label search (overall results baseline)
- AND/OR complex predicates (two-term)

Commit 1 is preview-only: no DB execution. Use the printed commands with the
existing baselines to perform the runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fire


def _outdir(dataset_variant: str, mode: str) -> Path:
    return Path("output/pgvector/hnsw_ordering_ab") / dataset_variant / mode


@dataclass
class SingleArgs:
    dataset_variant: str
    dataset_key: str
    test_size: float
    k: int
    m: int
    ef_construction: int
    ef_search: int
    dsn: Optional[str]
    dry_run: bool


@dataclass
class ComplexArgs:
    dataset_variant: str
    dataset_key: str
    test_size: float
    k: int
    m: int
    ef_construction: int
    ef_search: int
    n_filters_per_template: int
    n_queries_per_filter: int
    dsn: Optional[str]
    dry_run: bool


class HnswOrderingAB:
    def ab_single(
        self,
        *,
        dataset_variant: str = "yfcc100m_1m",
        dataset_key: str = "yfcc100m",
        test_size: float = 0.01,
        k: int = 10,
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
        dataset_cache_path: str | None = None,
        dsn: str | None = None,
        max_queries: int | None = None,
        dry_run: bool = False,
    ) -> None:
        """Preview A/B commands and output paths for single-label runs."""
        args = SingleArgs(
            dataset_variant=dataset_variant,
            dataset_key=dataset_key,
            test_size=test_size,
            k=k,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            dsn=dsn,
            dry_run=dry_run,
        )

        strict_dir = _outdir(args.dataset_variant, "strict_order")
        relaxed_dir = _outdir(args.dataset_variant, "relaxed_order")
        strict_csv = strict_dir / "results.csv"
        relaxed_csv = relaxed_dir / "results.csv"

        strict_cmd = (
            "python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single "
            f"--strategy hnsw --m {args.m} --ef_construction {args.ef_construction} "
            f"--ef_search {args.ef_search} --iter_mode strict_order "
            f"--dataset_key {args.dataset_key} --test_size {args.test_size} --k {args.k} "
            f"--output_path {strict_csv}"
            + (f" --dsn {args.dsn}" if args.dsn else "")
        )
        if dataset_cache_path:
            strict_cmd += f" --dataset_cache_path {dataset_cache_path}"
        if max_queries is not None:
            strict_cmd += f" --max_queries {int(max_queries)}"

        relaxed_cmd = (
            "python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single "
            f"--strategy hnsw --m {args.m} --ef_construction {args.ef_construction} "
            f"--ef_search {args.ef_search} --iter_mode relaxed_order "
            f"--dataset_key {args.dataset_key} --test_size {args.test_size} --k {args.k} "
            f"--output_path {relaxed_csv}"
            + (f" --dsn {args.dsn}" if args.dsn else "")
        )
        if dataset_cache_path:
            relaxed_cmd += f" --dataset_cache_path {dataset_cache_path}"
        if max_queries is not None:
            relaxed_cmd += f" --max_queries {int(max_queries)}"

        print("[ab][single] Planned outputs:")
        print("  ", strict_csv)
        print("  ", relaxed_csv)
        print("[ab][single] Commands to run:")
        print("--- strict_order ---\n", strict_cmd)
        print("--- relaxed_order (+ strict post-sort in baseline) ---\n", relaxed_cmd)
        if not args.dry_run:
            print("[ab][single] Executing strict_order baseline...")
            from benchmark.overall_results.baselines.pgvector import (
                exp_pgvector_single,
            )
            strict_dir.mkdir(parents=True, exist_ok=True)
            relaxed_dir.mkdir(parents=True, exist_ok=True)
            exp_pgvector_single(
                dsn=dsn,
                strategy="hnsw",
                iter_mode="strict_order",
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                dataset_key=dataset_key,
                test_size=test_size,
                k=k,
                output_path=str(strict_csv),
                dataset_cache_path=dataset_cache_path,
                max_queries=max_queries,
            )
            print("[ab][single] Executing relaxed_order baseline (post-sort enforced by baseline)...")
            exp_pgvector_single(
                dsn=dsn,
                strategy="hnsw",
                iter_mode="relaxed_order",
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                dataset_key=dataset_key,
                test_size=test_size,
                k=k,
                output_path=str(relaxed_csv),
                dataset_cache_path=dataset_cache_path,
                max_queries=max_queries,
            )

    def ab_complex(
        self,
        *,
        dataset_variant: str = "yfcc100m_1m",
        dataset_key: str = "yfcc100m",
        test_size: float = 0.01,
        k: int = 10,
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
        n_filters_per_template: int = 50,
        n_queries_per_filter: int = 100,
        allow_gt_compute: bool = True,
        dsn: str | None = None,
        dry_run: bool = True,
    ) -> None:
        """Preview A/B commands and output paths for AND/OR complex predicates."""
        args = ComplexArgs(
            dataset_variant=dataset_variant,
            dataset_key=dataset_key,
            test_size=test_size,
            k=k,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search,
            n_filters_per_template=n_filters_per_template,
            n_queries_per_filter=n_queries_per_filter,
            dsn=dsn,
            dry_run=dry_run,
        )

        strict_dir = _outdir(args.dataset_variant, "strict_order")
        relaxed_dir = _outdir(args.dataset_variant, "relaxed_order")
        strict_json = strict_dir / "results.json"
        relaxed_json = relaxed_dir / "results.json"

        base = (
            "python -m benchmark.complex_predicate.baselines.pgvector exp_pgvector_complex "
            f"--strategy hnsw --m {args.m} --ef_construction {args.ef_construction} "
            f"--ef_search {args.ef_search} --dataset_key {args.dataset_key} "
            f"--test_size {args.test_size} --k {args.k} "
            f"--n_filters_per_template {args.n_filters_per_template} "
            f"--n_queries_per_filter {args.n_queries_per_filter}"
        )

        strict_cmd = (
            base
            + f" --iter_mode strict_order --output_path {strict_json}"
            + (f" --dsn {args.dsn}" if args.dsn else "")
        )
        if not allow_gt_compute:
            strict_cmd += " --allow_gt_compute false"

        relaxed_cmd = (
            base
            + f" --iter_mode relaxed_order --output_path {relaxed_json}"
            + (f" --dsn {args.dsn}" if args.dsn else "")
        )
        if not allow_gt_compute:
            relaxed_cmd += " --allow_gt_compute false"

        print("[ab][complex] Planned outputs:")
        print("  ", strict_json)
        print("  ", relaxed_json)
        print("[ab][complex] Commands to run:")
        print("--- strict_order ---\n", strict_cmd)
        print("--- relaxed_order (+ strict post-sort in baseline) ---\n", relaxed_cmd)
        if not args.dry_run:
            print("[ab][complex] Executing strict_order baseline...")
            from benchmark.complex_predicate.baselines.pgvector import (
                exp_pgvector_complex,
            )
            strict_dir.mkdir(parents=True, exist_ok=True)
            relaxed_dir.mkdir(parents=True, exist_ok=True)
            exp_pgvector_complex(
                dsn=dsn,
                strategy="hnsw",
                iter_mode="strict_order",
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                dataset_key=dataset_key,
                test_size=test_size,
                k=k,
                n_filters_per_template=n_filters_per_template,
                n_queries_per_filter=n_queries_per_filter,
                output_path=str(strict_json),
                allow_gt_compute=allow_gt_compute,
            )
            print("[ab][complex] Executing relaxed_order baseline (post-sort enforced by baseline)...")
            exp_pgvector_complex(
                dsn=dsn,
                strategy="hnsw",
                iter_mode="relaxed_order",
                m=m,
                ef_construction=ef_construction,
                ef_search=ef_search,
                dataset_key=dataset_key,
                test_size=test_size,
                k=k,
                n_filters_per_template=n_filters_per_template,
                n_queries_per_filter=n_queries_per_filter,
                output_path=str(relaxed_json),
                allow_gt_compute=allow_gt_compute,
            )


def main() -> None:
    fire.Fire(HnswOrderingAB)


if __name__ == "__main__":
    main()
