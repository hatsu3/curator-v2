"""pgvector baselines for single-label search (YFCC-focused).

This initial commit provides:
- DSN handling with default and explicit flag
- Session GUC helpers
- Strict-order SQL wrapper (for iterative relaxed modes)
- Index presence checks (GIN(tags), HNSW, IVFFlat)

Implementation is scaffold-only: supports `--dry_run` to preview SQL and
validate DB connectivity/index presence without running full benchmarks.
Subsequent commits will add actual query logic and metrics emission.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any

import fire
import json
import time
import numpy as np
import pandas as pd

from benchmark.profiler import Dataset
from benchmark.utils import recall
"""
Note on dataset caching and ground truth:
- This runner assumes a precomputed dataset cache (see
  benchmark/overall_results/preproc_dataset.py) and WILL NOT recompute
  ground truth. Pass --dataset_cache_path pointing to the cache dir.
- If cache is missing, this runner raises an error to avoid accidental
  heavy recomputation.
"""


# Defaults per project description
DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/curator_bench"


def resolve_dsn(dsn: Optional[str]) -> str:
    """Resolve DSN from explicit flag, env var PG_DSN, or default."""
    return dsn or os.environ.get("PG_DSN", DEFAULT_DSN)


def set_session_gucs(conn, gucs: Dict[str, object]) -> None:
    """Apply session GUCs like ivfflat.probes, iterative_scan, etc."""
    if not gucs:
        return
    with conn.cursor() as cur:
        for k, v in gucs.items():
            cur.execute(f"SET {k} = %s;", (v,))


def strict_order_wrapper(core_sql: str) -> str:
    """Wrap an iterative relaxed query to enforce strict ordering.

    Expects `core_sql` to select columns `(id, distance)`.
    Returns SQL that materializes and re-orders by `distance + 0`.
    """
    return (
        "WITH relaxed_results AS MATERIALIZED (\n"
        f"{core_sql}\n"
        ") SELECT id FROM relaxed_results ORDER BY distance + 0;"
    )


def _index_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = %s;",
            (name,),
        )
        return cur.fetchone() is not None


def _has_vector_index(conn, method: str) -> bool:
    q = (
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'items' "
        "AND indexdef ILIKE %s LIMIT 1;"
    )
    like = f"%USING {method}%"
    with conn.cursor() as cur:
        cur.execute(q, (like,))
        return cur.fetchone() is not None


def _has_column(conn, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = %s LIMIT 1;",
            (table, column),
        )
        return cur.fetchone() is not None


def ensure_indexes_for_strategy(conn, strategy: str, schema_mode: str) -> None:
    """Validate required indexes per strategy.

    - prefilter: requires GIN(tags)
    - ivf: requires an ivfflat index on embedding
    - hnsw: requires an hnsw index on embedding
    Raises AssertionError with actionable message if missing.
    """
    # Require GIN(tags) only for INT[] path; boolean path skips GIN requirement
    if schema_mode == "int_array":
        if not _index_exists(conn, "items_tags_gin"):
            raise AssertionError(
                "Missing GIN(tags) index 'items_tags_gin'. Run scripts.pgvector.setup_db create_schema."
            )
    s = strategy.lower()
    if s == "prefilter":
        return
    if s == "ivf":
        if not _has_vector_index(conn, "ivfflat"):
            raise AssertionError(
                "Missing IVFFlat index on items.embedding. Create 'items_emb_ivf' via setup_db.create_index --index ivf."
            )
        return
    if s == "hnsw":
        if not _has_vector_index(conn, "hnsw"):
            raise AssertionError(
                "Missing HNSW index on items.embedding. Create 'items_emb_hnsw' via setup_db.create_index --index hnsw."
            )
        return
    raise AssertionError(f"Unsupported strategy: {strategy}")


def get_embedding_dim_from_db(conn) -> Optional[int]:
    """Return the dimension of vectors stored in items.embedding if any.

    If the table is empty or the dimension can't be determined, returns None.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT vector_dims(embedding) FROM items WHERE embedding IS NOT NULL LIMIT 1;"
        )
        row = cur.fetchone()
        if row and row[0] is not None:
            try:
                return int(row[0])
            except Exception:
                return None
    return None


@dataclass
class SingleLabelParams:
    # Common
    strategy: str = "hnsw"  # prefilter|ivf|hnsw
    iter_mode: str = "relaxed_order"  # relaxed_order|strict_order
    k: int = 10
    # IVF
    lists: Optional[int] = None
    probes: Optional[int] = None
    # HNSW
    m: Optional[int] = None
    ef_construction: Optional[int] = None
    ef_search: Optional[int] = None


def exp_pgvector_single(
    *,
    dsn: str | None = None,
    strategy: str = "hnsw",
    iter_mode: str = "relaxed_order",
    schema: str = "int_array",  # int_array | boolean
    # IVF params
    lists: int | None = None,
    probes: int | None = None,
    # HNSW params
    m: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
    # Dataset/outputs
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_path: str | Path | None = None,
    k: int = 10,
    output_path: str | None = None,
    # Control
    dry_run: bool = False,
):
    """Single-label baseline runner.

    - dry_run=True: preview SQL and GUCs without DB or dataset
    - dry_run=False: execute queries over Postgres and emit CSV/JSON
    """
    dsn_resolved = resolve_dsn(dsn)
    print(f"[pgvector] Using DSN: {dsn_resolved}")
    # Session GUCs
    gucs: Dict[str, object] = {}
    if strategy == "ivf":
        # If probes not provided, default to nlist (lists) for full coverage
        if probes is not None:
            gucs["ivfflat.probes"] = int(probes)
        elif lists is not None:
            gucs["ivfflat.probes"] = int(lists)
        gucs["ivfflat.iterative_scan"] = iter_mode
    elif strategy == "hnsw":
        if ef_search is not None:
            gucs["hnsw.ef_search"] = ef_search
        gucs["hnsw.iterative_scan"] = iter_mode
        # Remove tuple scan cap to avoid early cutoffs when debugging recall
        gucs["hnsw.max_scan_tuples"] = 1000000000

    if dry_run:
        if gucs:
            print("[pgvector] Session GUCs (preview):", gucs)
        # Example SQL preview without DB connection
        if schema == "int_array":
            core_sql = (
                "SELECT id, embedding <-> $1::vector AS distance "
                "FROM items WHERE tags @> ARRAY[$2] ORDER BY distance LIMIT $3"
            )
        elif schema == "boolean":
            # Show a representative boolean column (substitute a label id)
            core_sql = (
                "SELECT id, embedding <-> $1::vector AS distance "
                "FROM items WHERE label_<L> ORDER BY distance LIMIT $2"
            )
        else:
            raise AssertionError("schema must be one of: int_array, boolean")
        sql = strict_order_wrapper(core_sql) if iter_mode == "relaxed_order" else core_sql
        print("[pgvector] Example query SQL (preview):\n", sql)
        print("[pgvector] Dry-run: skipping DB connection and dataset work.")
        return

    # Real path: connect, set GUCs, check indexes
    try:
        import psycopg2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"psycopg2 not available: {e}")
    conn = psycopg2.connect(dsn_resolved)
    conn.autocommit = True
    try:
        set_session_gucs(conn, gucs)
        print("[pgvector] Session GUCs applied.")
        ensure_indexes_for_strategy(conn, strategy, schema)
        print(f"[pgvector] Index presence OK for strategy '{strategy}'.")
        # Load dataset; if cache_path is provided and exists, Dataset will use it.
        # Otherwise, it will use raw sources and load ground truth from data/ground_truth
        # if present (compute only if missing).
        cache_path = Path(dataset_cache_path) if dataset_cache_path else None
        ds = Dataset.from_dataset_key(
            dataset_key, test_size=test_size, cache_path=cache_path, k=k, verbose=True
        )

        # Enforce dataset dim for known datasets (e.g., YFCC == 192)
        if dataset_key.startswith("yfcc"):
            assert (
                ds.dim == 192
            ), f"YFCC dataset must have dim=192, got {ds.dim}. Check your cache and data."

        # Detect DB embedding dimension; enforce equality when available
        db_dim = get_embedding_dim_from_db(conn)
        if db_dim is not None:
            assert (
                int(db_dim) == ds.dim
            ), (
                f"DB embedding dimension ({db_dim}) does not match dataset dim ({ds.dim}). "
                "Ensure table uses vector(192) for YFCC and data is loaded with 192-D embeddings."
            )

        # Prepare SQL builder per schema
        def build_sql_for_label(lab: int) -> str:
            if schema == "int_array":
                core = (
                    "SELECT id, embedding <-> %s::vector AS distance "
                    "FROM items WHERE tags @> ARRAY[%s] ORDER BY distance LIMIT %s"
                )
            elif schema == "boolean":
                col = f"label_{int(lab)}"
                # Optionally validate column exists before querying
                if not _has_column(conn, "items", col):
                    raise AssertionError(
                        f"Missing boolean label column '{col}'. Create via setup_db.create_schema --schema boolean --label_ids ..."
                    )
                core = (
                    f"SELECT id, embedding <-> %s::vector AS distance "
                    f"FROM items WHERE {col} ORDER BY distance LIMIT %s"
                )
            else:
                raise AssertionError("schema must be one of: int_array, boolean")
            return strict_order_wrapper(core) if iter_mode == "relaxed_order" else core

        def vec_to_literal(vec: np.ndarray) -> str:
            return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

        query_results: List[List[int]] = []
        query_latencies: List[float] = []

        with conn.cursor() as cur:
            for vec, access_list in zip(ds.test_vecs, ds.test_mds):
                for label in access_list:
                    if label not in ds.all_labels:
                        continue
                    vlit = vec_to_literal(vec)
                    sql = build_sql_for_label(int(label))
                    start = time.perf_counter()
                    if schema == "int_array":
                        cur.execute(sql, (vlit, int(label), int(k)))
                    else:
                        # boolean mode: only vector and k are parameters
                        cur.execute(sql, (vlit, int(k)))
                    rows = cur.fetchall()
                    elapsed = time.perf_counter() - start
                    query_latencies.append(elapsed)
                    # Map DB ids (1-based) to 0-based train indices for recall comparison
                    ids = [int(r[0]) - 1 for r in rows]
                    query_results.append(ids)

        # Compute recall (use cached ground truth; do not recompute)
        rec = recall(query_results, ds.ground_truth)

        # Summarize metrics
        lat = np.array(query_latencies, dtype=float)
        results_row: Dict[str, Any] = {
            "strategy": strategy,
            "iter_mode": iter_mode,
            "k": int(k),
            "recall_at_k": float(rec),
            "query_qps": float(1.0 / lat.mean()),
            "query_lat_avg": float(lat.mean()),
            "query_lat_std": float(lat.std()),
            "query_lat_p50": float(np.percentile(lat, 50)),
            "query_lat_p95": float(np.percentile(lat, 95)),
            "query_lat_p99": float(np.percentile(lat, 99)),
            "dataset_key": dataset_key,
            "test_size": float(test_size),
        }

        # Emit outputs
        assert output_path is not None, "output_path is required for execution"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([results_row]).to_csv(out_path, index=False)
        print(f"[pgvector] Wrote results CSV to {out_path}")

        # parameters.json sidecar
        params_json = {
            "dsn": dsn_resolved,
            "strategy": strategy,
            "iter_mode": iter_mode,
            "k": int(k),
            "ivf": {"lists": lists, "probes": probes},
            "hnsw": {"m": m, "ef_construction": ef_construction, "ef_search": ef_search},
            "gucs": gucs,
            "dataset_key": dataset_key,
            "test_size": float(test_size),
            "dim": int(ds.dim),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        params_path = out_path.parent / "parameters.json"
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params_json, f, indent=2, sort_keys=True)
        print(f"[pgvector] Wrote parameters to {params_path}")
    finally:
        conn.close()


def main() -> None:
    fire.Fire()


if __name__ == "__main__":
    main()
