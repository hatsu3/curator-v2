"""CLI for loading datasets and running insert benchmarks (skeleton).

This initial skeleton provides:
- Fire CLI with two subcommands: `bulk` and `insert_bench`
- Strict copy format selection: binary (default) or csv
- Canonical output path resolution for pgvector_* baselines
- Dry-run previews of planned actions and output artifact locations

Future commits will implement actual DB execution, COPY streams, and
metrics emission. Use this to validate flags, output pathing, and flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Dict, Any

import os
import sys
import io
import time
import json
import csv
from datetime import datetime

import fire

from dataset import get_dataset, get_metadata
from scripts.pgvector import admin

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/curator_bench"


def _warn(msg: str) -> None:
    print(f"[pgvector][WARN] {msg}", file=sys.stderr)


def _resolve_dsn(dsn: Optional[str]) -> str:
    return dsn or os.environ.get("PG_DSN", DEFAULT_DSN)


def _baseline_dir_for_strategy(strategy: str) -> str:
    s = strategy.lower()
    if s == "prefilter":
        return "pgvector_prefilter"
    if s == "hnsw":
        return "pgvector_hnsw"
    if s == "ivf":
        return "pgvector_ivf"
    raise ValueError(f"unsupported strategy: {strategy}")


def _baseline_dir_for_index(index: str) -> str:
    i = index.lower()
    if i == "gin":
        return "pgvector_prefilter"
    if i == "hnsw":
        return "pgvector_hnsw"
    if i == "ivf":
        return "pgvector_ivf"
    raise ValueError(f"unsupported index: {index}")


def _canonical_output_dir(baseline: str, dataset_key: str, test_size: float) -> str:
    return os.path.join(
        "output",
        "overall_results2",
        baseline,
        f"{dataset_key}_test{test_size}",
    )


def _durability_label(unlogged: bool, sync_commit_off: bool) -> str:
    return "non_durable" if (unlogged or sync_commit_off) else "durable"


def _validate_copy_format(copy_format: str) -> str:
    cf = (copy_format or "").lower()
    if cf not in {"binary", "csv"}:
        raise ValueError("copy_format must be one of: binary, csv")
    return cf


@dataclass
class BulkArgs:
    dsn: Optional[str]
    dataset: str
    dataset_key: str
    dim: int
    test_size: float
    copy_format: str
    unlogged: bool
    sync_commit_off: bool
    build_index: Optional[str]
    m: Optional[int]
    efc: Optional[int]
    lists: Optional[int]
    limit: Optional[int]
    dry_run: bool


@dataclass
class InsertArgs:
    dsn: Optional[str]
    dataset: str
    dataset_key: str
    dim: int
    test_size: float
    strategy: str
    unlogged: bool
    sync_commit_off: bool
    m: Optional[int]
    efc: Optional[int]
    ef_search: Optional[int]
    lists: Optional[int]
    limit: Optional[int]
    dry_run: bool


class LoadDataset:
    def bulk(
        self,
        *,
        dsn: str | None = None,
        dataset: str = "yfcc100m",  # yfcc100m | arxiv
        dataset_key: str = "yfcc100m-10m",
        dim: int = 192,
        test_size: float = 0.001,
        copy_format: str = "binary",  # binary | csv
        unlogged: bool = False,
        sync_commit_off: bool = False,
        # Optional post-load build
        build_index: str | None = None,  # gin | hnsw | ivf
        m: int | None = None,
        efc: int | None = None,
        lists: int | None = None,
        limit: int | None = None,
        dry_run: bool = True,
    ) -> None:
        """Bulk load vectors (skeleton): prints planned actions and outputs.

        Notes:
        - copy_format is strict; default binary; no auto-fallback in skeleton.
        - build_index controls post-load index timing (gin|hnsw|ivf).
        - No DB actions occur when dry_run=True (default in skeleton).
        """
        args = BulkArgs(
            dsn=dsn,
            dataset=dataset,
            dataset_key=dataset_key,
            dim=dim,
            test_size=test_size,
            copy_format=_validate_copy_format(copy_format),
            unlogged=unlogged,
            sync_commit_off=sync_commit_off,
            build_index=build_index,
            m=m,
            efc=efc,
            lists=lists,
            limit=limit,
            dry_run=dry_run,
        )

        dsn_resolved = _resolve_dsn(args.dsn)
        print(f"[pgvector] Using DSN: {dsn_resolved}")
        print(
            "[pgvector] Bulk load (skeleton):",
            {
                "dataset": args.dataset,
                "dataset_key": args.dataset_key,
                "dim": args.dim,
                "test_size": args.test_size,
                "copy_format": args.copy_format,
                "unlogged": args.unlogged,
                "sync_commit_off": args.sync_commit_off,
                "build_index": args.build_index,
                "m": args.m,
                "efc": args.efc,
                "lists": args.lists,
            },
        )

        if args.build_index:
            baseline = _baseline_dir_for_index(args.build_index)
            out_dir = _canonical_output_dir(baseline, args.dataset_key, args.test_size)
            build_json = os.path.join(out_dir, "build.json")
            build_csv = os.path.join(out_dir, "build.csv")
            print("[pgvector] Planned build artifacts:")
            print("  ", build_json)
            print("  ", build_csv)

        if args.dry_run:
            print("[pgvector] Dry-run: preview only. COPY and CREATE INDEX not executed.")
            return

        # Real execution path: create schema (optionally without GIN), COPY rows, ANALYZE, and build index if requested.
        # Defer GIN creation to post-load if we are timing GIN build.
        admin.create_schema(
            _resolve_dsn(args.dsn),
            dim=args.dim,
            create_gin=False if args.build_index == "gin" else True,
            unlogged=args.unlogged,
            dry_run=False,
        )

        # Load train split vectors and metadata
        train_vecs, _test_vecs, _meta = get_dataset(args.dataset, test_size=args.test_size)
        train_mds, _test_mds = get_metadata(
            synthesized=False, dataset_name=args.dataset, test_size=args.test_size
        )
        if train_vecs.shape[1] != args.dim:
            _warn(
                f"dim mismatch: dataset has {train_vecs.shape[1]}, CLI --dim {args.dim}. Proceeding with dataset dim."
            )

        n_rows = train_vecs.shape[0]
        if args.limit is not None:
            n_rows = min(n_rows, int(args.limit))

        # Session settings
        import psycopg2  # type: ignore

        dsn_resolved = _resolve_dsn(args.dsn)
        conn = psycopg2.connect(dsn_resolved)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                if args.sync_commit_off:
                    cur.execute("SET synchronous_commit = off;")

            if args.copy_format == "binary":
                raise RuntimeError(
                    "binary COPY not implemented yet; rerun with --copy_format csv as per policy"
                )

            # Prepare tab-delimited text to avoid quoting commas in CSV
            def row_iter() -> Iterable[str]:
                for i in range(n_rows):
                    # Assign 1-based IDs to rows in the order provided
                    rid = i + 1
                    vec = train_vecs[i]
                    tags = train_mds[i]
                    emb_txt = "[" + ",".join(str(float(x)) for x in vec) + "]"
                    tags_txt = "{" + ",".join(str(int(t)) for t in tags) + "}"
                    yield f"{rid}\t{emb_txt}\t{tags_txt}\n"

            buf = io.StringIO()
            written = 0
            with conn.cursor() as cur:
                for line in row_iter():
                    buf.write(line)
                    written += 1
                    # flush periodically to avoid large memory
                    if written % 100000 == 0:
                        buf.seek(0)
                        cur.copy_from(buf, "items", columns=("id", "embedding", "tags"), sep="\t")
                        buf.close()
                        buf = io.StringIO()
                # flush remainder
                buf.seek(0)
                if written > 0:
                    cur.copy_from(buf, "items", columns=("id", "embedding", "tags"), sep="\t")
            buf.close()

            # Analyze table for realistic planner stats
            with conn.cursor() as cur:
                cur.execute("ANALYZE items;")

        finally:
            conn.close()

        # Optional post-load index build and metrics
        if args.build_index:
            baseline = _baseline_dir_for_index(args.build_index)
            out_dir = _canonical_output_dir(baseline, args.dataset_key, args.test_size)
            build_json = os.path.join(out_dir, "build.json")
            build_csv = os.path.join(out_dir, "build.csv")
            admin.create_index(
                _resolve_dsn(args.dsn),
                index=args.build_index,
                dim=args.dim,
                m=args.m,
                efc=args.efc,
                lists=args.lists,
                dry_run=False,
                output_json=build_json,
                output_csv=build_csv,
            )

    def insert_bench(
        self,
        *,
        dsn: str | None = None,
        dataset: str = "yfcc100m",
        dataset_key: str = "yfcc100m-10m",
        dim: int = 192,
        test_size: float = 0.001,
        strategy: str = "hnsw",  # prefilter | hnsw | ivf
        unlogged: bool = False,
        sync_commit_off: bool = False,
        # Optional knobs (not used in skeleton execution)
        m: int | None = None,
        efc: int | None = None,
        ef_search: int | None = None,
        lists: int | None = None,
        limit: int | None = None,
        # IVF training seed controls
        ivf_seed_frac: float = 0.1,
        seed_copy_format: str = "binary",  # binary | csv (binary not implemented yet)
        dry_run: bool = True,
    ) -> None:
        """Single-thread incremental insert benchmark (skeleton).

        Strategy controls the baseline bucket for outputs.
        """
        args = InsertArgs(
            dsn=dsn,
            dataset=dataset,
            dataset_key=dataset_key,
            dim=dim,
            test_size=test_size,
            strategy=strategy,
            unlogged=unlogged,
            sync_commit_off=sync_commit_off,
            m=m,
            efc=efc,
            ef_search=ef_search,
            lists=lists,
            limit=limit,
            dry_run=dry_run,
        )

        dsn_resolved = _resolve_dsn(args.dsn)
        print(f"[pgvector] Using DSN: {dsn_resolved}")
        print(
            "[pgvector] Insert bench (skeleton):",
            {
                "dataset": args.dataset,
                "dataset_key": args.dataset_key,
                "dim": args.dim,
                "test_size": args.test_size,
                "strategy": args.strategy,
                "unlogged": args.unlogged,
                "sync_commit_off": args.sync_commit_off,
                "m": args.m,
                "ef_search": args.ef_search,
                "lists": args.lists,
            },
        )

        baseline = _baseline_dir_for_strategy(args.strategy)
        out_dir = _canonical_output_dir(baseline, args.dataset_key, args.test_size)
        label = _durability_label(args.unlogged, args.sync_commit_off)
        insert_json = os.path.join(out_dir, f"insert_{label}.json")
        insert_csv = os.path.join(out_dir, f"insert_{label}.csv")
        print("[pgvector] Planned insert artifacts:")
        print("  ", insert_json)
        print("  ", insert_csv)

        # If IVF, also preview build artifacts (training build)
        if args.strategy.lower() == "ivf":
            baseline_b = _baseline_dir_for_index("ivf")
            out_b = _canonical_output_dir(baseline_b, args.dataset_key, args.test_size)
            print("[pgvector] Planned ivf build artifacts:")
            print("  ", os.path.join(out_b, "build.json"))
            print("  ", os.path.join(out_b, "build.csv"))

        if args.dry_run:
            print("[pgvector] Dry-run: preview only. No inserts executed.")
            return

        s = args.strategy.lower()
        if s not in {"hnsw", "ivf"}:
            _warn("insert benchmark real execution currently implemented for strategy=hnsw or ivf only")
            return

        # Ensure schema; keep GIN installed for realism
        admin.create_schema(
            _resolve_dsn(args.dsn),
            dim=args.dim,
            create_gin=True,
            unlogged=args.unlogged,
            dry_run=False,
        )

        # Load training data
        train_vecs, _test_vecs, _meta = get_dataset(args.dataset, test_size=args.test_size)
        train_mds, _test_mds = get_metadata(
            synthesized=False, dataset_name=args.dataset, test_size=args.test_size
        )
        n_rows = train_vecs.shape[0]
        if args.limit is not None:
            n_rows = min(n_rows, int(args.limit))

        import psycopg2  # type: ignore
        dsn_resolved = _resolve_dsn(args.dsn)
        conn = psycopg2.connect(dsn_resolved)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Start from empty table
                cur.execute("TRUNCATE items;")
                # Apply durability toggle
                if args.sync_commit_off:
                    cur.execute("SET synchronous_commit = off;")

            if s == "hnsw":
                # Build HNSW index on empty table (params required)
                if args.m is None or args.efc is None:
                    raise RuntimeError("hnsw insertion benchmark requires --m and --efc parameters")
                admin.create_index(
                    dsn_resolved,
                    index="hnsw",
                    dim=args.dim,
                    m=args.m,
                    efc=args.efc,
                    dry_run=False,
                )
            elif s == "ivf":
                # Seed a fraction for IVFFlat training, then build, then delete seeds
                if args.lists is None:
                    raise RuntimeError("ivf insertion benchmark requires --lists parameter")
                seed_frac = float(ivf_seed_frac)
                seed_n = max(1, int(train_vecs.shape[0] * seed_frac))
                # Copy seed rows via CSV path (binary not implemented yet)
                scf = _validate_copy_format(seed_copy_format)
                if scf == "binary":
                    raise RuntimeError("binary COPY for seeding not implemented; use --seed_copy_format csv")
                # Write seed rows in chunks
                def seed_iter() -> Iterable[str]:
                    import numpy as _np  # local
                    _np.random.seed(42)
                    idxs = _np.random.choice(train_vecs.shape[0], seed_n, replace=False)
                    for rid in idxs:
                        vec = train_vecs[rid]
                        tags = train_mds[rid]
                        emb_txt = "[" + ",".join(str(float(x)) for x in vec) + "]"
                        tags_txt = "{" + ",".join(str(int(t)) for t in tags) + "}"
                        yield f"{rid+1}\t{emb_txt}\t{tags_txt}\n"

                buf = io.StringIO()
                written = 0
                with conn.cursor() as cur:
                    for line in seed_iter():
                        buf.write(line)
                        written += 1
                        if written % 100000 == 0:
                            buf.seek(0)
                            cur.copy_from(buf, "items", columns=("id", "embedding", "tags"), sep="\t")
                            buf.close()
                            buf = io.StringIO()
                    buf.seek(0)
                    if written > 0:
                        cur.copy_from(buf, "items", columns=("id", "embedding", "tags"), sep="\t")
                buf.close()

                # Build IVF and record metrics
                baseline_b = _baseline_dir_for_index("ivf")
                out_b = _canonical_output_dir(baseline_b, args.dataset_key, args.test_size)
                os.makedirs(out_b, exist_ok=True)
                admin.create_index(
                    dsn_resolved,
                    index="ivf",
                    dim=args.dim,
                    lists=args.lists,
                    dry_run=False,
                    output_json=os.path.join(out_b, "build.json"),
                    output_csv=os.path.join(out_b, "build.csv"),
                )
                # Delete seeds and vacuum
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE items;")
                    cur.execute("VACUUM ANALYZE items;")

            # Insert rows one by one; measure per-row latency
            latencies: list[float] = []
            total_start = time.perf_counter()
            with conn.cursor() as cur:
                for i in range(n_rows):
                    rid = i + 1
                    vec = train_vecs[i]
                    tags = train_mds[i]
                    emb_txt = "[" + ",".join(str(float(x)) for x in vec) + "]"
                    t0 = time.perf_counter()
                    cur.execute(
                        "INSERT INTO items (id, embedding, tags) VALUES (%s, %s::vector, %s);",
                        (rid, emb_txt, tags),
                    )
                    latencies.append(time.perf_counter() - t0)
            total_dur = time.perf_counter() - total_start

        finally:
            conn.close()

        # Aggregate metrics
        import numpy as np  # type: ignore
        lat = np.array(latencies, dtype=np.float64)
        # Query index size after insertion
        idx_name = "items_emb_hnsw" if s == "hnsw" else ("items_emb_ivf" if s == "ivf" else None)
        index_size_bytes = None
        if idx_name is not None:
            import psycopg2  # type: ignore
            with psycopg2.connect(_resolve_dsn(args.dsn)) as _c:
                _c.autocommit = True
                with _c.cursor() as _cur:
                    _cur.execute("SELECT pg_relation_size(%s);", (idx_name,))
                    index_size_bytes = int(_cur.fetchone()[0])
        metrics: Dict[str, Any] = {
            "status": "ok",
            "strategy": s,
            "dataset_key": args.dataset_key,
            "test_size": args.test_size,
            "dim": args.dim,
            "n_rows": int(n_rows),
            "durability": _durability_label(args.unlogged, args.sync_commit_off),
            "insert_qps": float(n_rows / max(total_dur, 1e-9)),
            "insert_lat_avg": float(lat.mean()) if lat.size else None,
            "insert_lat_std": float(lat.std()) if lat.size else None,
            "insert_lat_p50": float(np.percentile(lat, 50)) if lat.size else None,
            "insert_lat_p99": float(np.percentile(lat, 99)) if lat.size else None,
            "index_size_bytes": index_size_bytes,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Emit artifacts
        baseline = _baseline_dir_for_strategy(args.strategy)
        out_dir = _canonical_output_dir(baseline, args.dataset_key, args.test_size)
        os.makedirs(out_dir, exist_ok=True)
        label = _durability_label(args.unlogged, args.sync_commit_off)
        insert_json = os.path.join(out_dir, f"insert_{label}.json")
        insert_csv = os.path.join(out_dir, f"insert_{label}.csv")
        with open(insert_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        # CSV append
        flat = metrics.copy()
        write_header = not os.path.exists(insert_csv)
        with open(insert_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat)
        print(f"[pgvector] Insert metrics written to {insert_json} and {insert_csv}")


def main() -> None:
    fire.Fire(LoadDataset)


if __name__ == "__main__":
    main()
