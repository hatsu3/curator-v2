from pathlib import Path

import fire

from benchmark.overall_results.baselines.pgvector import exp_pgvector_single


def _outdir(dataset_variant: str, mode: str) -> Path:
    return Path("output/pgvector/hnsw_ordering_ab") / dataset_variant / mode


class HnswOrderingAB:
    def ab_single(
        self,
        *,
        dataset_key: str = "yfcc100m",
        test_size: float = 0.01,
        k: int = 10,
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 64,
        dataset_cache_path: str | None = None,
        dsn: str | None = None,  # DSN for the database
        max_queries: int | None = None,  # Maximum number of queries to evaluate
        dry_run: bool = False,  # Preview commands only
    ) -> None:
        """Preview A/B commands and output paths for single-label runs."""
        strict_dir = _outdir(f"{dataset_key}_test{test_size}", "strict_order")
        relaxed_dir = _outdir(f"{dataset_key}_test{test_size}", "relaxed_order")
        strict_csv = strict_dir / "results.csv"
        relaxed_csv = relaxed_dir / "results.csv"

        strict_cmd = (
            "python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single "
            f"--strategy hnsw --m {m} --ef_construction {ef_construction} "
            f"--ef_search {ef_search} --iter_mode strict_order "
            f"--dataset_key {dataset_key} --test_size {test_size} --k {k} "
            f"--output_path {strict_csv}" + (f" --dsn {dsn}" if dsn else "")
        )
        if dataset_cache_path:
            strict_cmd += f" --dataset_cache_path {dataset_cache_path}"
        if max_queries is not None:
            strict_cmd += f" --max_queries {int(max_queries)}"

        relaxed_cmd = (
            "python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single "
            f"--strategy hnsw --m {m} --ef_construction {ef_construction} "
            f"--ef_search {ef_search} --iter_mode relaxed_order "
            f"--dataset_key {dataset_key} --test_size {test_size} --k {k} "
            f"--output_path {relaxed_csv}" + (f" --dsn {dsn}" if dsn else "")
        )
        if dataset_cache_path:
            relaxed_cmd += f" --dataset_cache_path {dataset_cache_path}"
        if max_queries is not None:
            relaxed_cmd += f" --max_queries {int(max_queries)}"

        print("[ab][single] Output paths:")
        print("  ", strict_csv)
        print("  ", relaxed_csv)

        print("[ab][single] Commands to run:")
        print("--- strict_order ---\n", strict_cmd)
        print("--- relaxed_order (+ strict post-sort in baseline) ---\n", relaxed_cmd)

        print("[ab][single] Executing strict_order baseline...")
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
            dry_run=dry_run,
        )

        print(
            "[ab][single] Executing relaxed_order baseline (post-sort enforced by baseline)..."
        )
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
            dry_run=dry_run,
        )


if __name__ == "__main__":
    fire.Fire(HnswOrderingAB)
