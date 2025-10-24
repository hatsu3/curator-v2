"""pgvector baselines for single-label search.

This initial commit provides:
- DSN handling with default and explicit flag
- Session GUC helpers
- Strict-order SQL wrapper (for iterative relaxed modes)
- Index presence checks (GIN(tags), HNSW, IVFFlat)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import fire
import numpy as np
import pandas as pd
import psycopg2
from tqdm import tqdm

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
            return int(row[0])
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
    # Limit number of label-filtered queries (optional)
    max_queries: int | None = None,
    # IVF params
    lists: int | None = None,
    probes: int | None = None,
    ivf_max_probes: int | None = None,
    # HNSW params (m, ef_construction only required for logging)
    m: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
    hnsw_max_scan_tuples: int | None = None,
    # Dataset/outputs
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_path: str | Path | None = None,
    k: int = 10,
    output_path: str | None = None,
):
    """Single-label baseline runner."""
    dsn_resolved = resolve_dsn(dsn)
    print(f"[pgvector] Using DSN: {dsn_resolved}")

    # Session GUCs
    gucs: Dict[str, object] = {}
    if strategy == "ivf":
        assert (
            lists is not None and probes is not None
        ), "lists and probes are required for ivf"
        gucs["ivfflat.probes"] = int(probes)
        gucs["ivfflat.iterative_scan"] = iter_mode
        # Optional cap for iterative IVF total probes
        if ivf_max_probes is not None:
            gucs["ivfflat.max_probes"] = int(ivf_max_probes)
    elif strategy == "hnsw":
        assert (
            ef_search is not None and m is not None and ef_construction is not None
        ), "ef_search, m, and ef_construction are required for hnsw"
        gucs["hnsw.ef_search"] = ef_search
        gucs["hnsw.iterative_scan"] = iter_mode
        if hnsw_max_scan_tuples is not None:
            gucs["hnsw.max_scan_tuples"] = int(hnsw_max_scan_tuples)

    # Connect, set GUCs, check indexes
    conn = psycopg2.connect(dsn_resolved)
    conn.autocommit = True
    try:
        set_session_gucs(conn, gucs)
        print("[pgvector] Session GUCs applied.")
        ensure_indexes_for_strategy(conn, strategy, schema)
        print(f"[pgvector] Index presence OK for strategy '{strategy}'.")

        # Load dataset
        cache_path = Path(dataset_cache_path) if dataset_cache_path else None
        ds = Dataset.from_dataset_key(
            dataset_key, test_size=test_size, cache_path=cache_path, k=k, verbose=True
        )

        # Check DB embedding dimension
        db_dim = get_embedding_dim_from_db(conn)
        assert db_dim is not None and int(db_dim) == ds.dim, (
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
        processed = 0

        with conn.cursor() as cur:
            for vec, access_list in tqdm(zip(ds.test_vecs, ds.test_mds), total=len(ds.test_vecs), desc="Querying database"):
                for label in access_list:
                    if max_queries is not None and processed >= int(max_queries):
                        break

                    vlit = vec_to_literal(vec)
                    sql = build_sql_for_label(int(label))

                    start = time.perf_counter()
                    if schema == "int_array":
                        cur.execute(sql, (vlit, int(label), int(k)))
                    else:
                        cur.execute(sql, (vlit, int(k)))  # label embedded in SQL
                    rows = cur.fetchall()
                    elapsed = time.perf_counter() - start

                    query_latencies.append(elapsed)
                    ids = [
                        int(r[0]) - 1 for r in rows
                    ]  # map DB ids (1-based) to 0-based
                    query_results.append(ids)
                    processed += 1

                if max_queries is not None and processed >= int(max_queries):
                    break

        # Compute recall
        rec = recall(query_results, ds.ground_truth[: len(query_results)])

        # Gather storage metrics from DB
        def _relation_size(name: str) -> int:
            with conn.cursor() as _c:
                _c.execute("SELECT pg_relation_size(%s);", (name,))
                row = _c.fetchone()
                return int(row[0]) if row and row[0] is not None else 0

        def _table_size(name: str) -> int:
            with conn.cursor() as _c:
                _c.execute("SELECT pg_table_size(%s);", (name,))
                row = _c.fetchone()
                return int(row[0]) if row and row[0] is not None else 0

        idx_name = {"hnsw": "items_emb_hnsw", "ivf": "items_emb_ivf"}[strategy]
        index_size_bytes = _relation_size(idx_name)
        table_size_bytes = _table_size("items")

        # Estimate raw vector bytes from DB row count and dimension
        n_rows = 0
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM items;")
            n_rows = int(cur.fetchone()[0])
        dim_eff = get_embedding_dim_from_db(conn)
        assert dim_eff is not None, "embedding dimension is not found"
        vector_bytes = int(n_rows) * int(dim_eff) * 4

        # Summarize metrics
        lat = np.array(query_latencies, dtype=float)

        # Per-query recall list for plotting pipelines
        def _recall_at_k_single(pred: List[int], gt: List[int], kk: int) -> float:
            return float(len(set(pred) & set(gt[:kk]))) / kk

        gt_lists = ds.ground_truth[: len(query_results)]
        per_query_recalls = [
            _recall_at_k_single(pred, gt, int(k))
            for pred, gt in zip(query_results, gt_lists)
        ]
        results_row: Dict[str, Any] = {
            "strategy": strategy,
            "iter_mode": iter_mode,
            "k": int(k),
            "recall_at_k": float(rec),
            "query_qps": float(1.0 / lat.mean()) if lat.size else 0.0,
            "query_lat_avg": float(lat.mean()) if lat.size else 0.0,
            "query_lat_std": float(lat.std()) if lat.size else 0.0,
            "query_lat_p90": float(np.percentile(lat, 90)) if lat.size else 0.0,
            "query_lat_p95": float(np.percentile(lat, 95)) if lat.size else 0.0,
            "query_lat_p99": float(np.percentile(lat, 99)) if lat.size else 0.0,
            "dataset_key": dataset_key,
            "test_size": float(test_size),
            "index_size_bytes": index_size_bytes,
            "table_size_bytes": table_size_bytes,
            "vector_bytes": vector_bytes,
            # list-valued columns for plotting preprocessors
            "query_latencies": json.dumps([float(x) for x in query_latencies]),
            "query_recalls": json.dumps([float(x) for x in per_query_recalls]),
        }

        # Save results to CSV
        assert output_path is not None, "output_path is required for execution"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([results_row]).to_csv(out_path, index=False)
        print(f"[pgvector] Wrote results CSV to {out_path}")

        # Dump parameters to JSON
        params_json = {
            "dsn": dsn_resolved,
            "strategy": strategy,
            "iter_mode": iter_mode,
            "k": int(k),
            "ivf": {"lists": lists, "probes": probes},
            "hnsw": {
                "m": m,
                "ef_construction": ef_construction,
                "ef_search": ef_search,
            },
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


if __name__ == "__main__":
    fire.Fire()
