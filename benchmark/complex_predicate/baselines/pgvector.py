"""pgvector baselines for AND/OR complex predicates on YFCC 1M.

Features:
- DSN default and explicit flag override
- Session GUC helpers
- Strict-order SQL wrapper for iterative relaxed modes
- Index presence checks (GIN(tags), HNSW, IVFFlat)
- Executes queries per-template (AND/OR), aggregates recall@k and latency/QPS
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import fire
import json
import time
import numpy as np
from pathlib import Path

from benchmark.utils import recall
from benchmark.complex_predicate.dataset import ComplexPredicateDataset

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/curator_bench"


def resolve_dsn(dsn: Optional[str]) -> str:
    return dsn or os.environ.get("PG_DSN", DEFAULT_DSN)


def set_session_gucs(conn, gucs: Dict[str, object]) -> None:
    if not gucs:
        return
    with conn.cursor() as cur:
        for k, v in gucs.items():
            cur.execute(f"SET {k} = %s;", (v,))


def strict_order_wrapper(core_sql: str) -> str:
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


def ensure_indexes_for_strategy(conn, strategy: str) -> None:
    if not _index_exists(conn, "items_tags_gin"):
        raise AssertionError(
            "Missing GIN(tags) index 'items_tags_gin'. Run scripts/pgvector.setup_db create_schema."
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


def get_embedding_dim_from_db(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT vector_dims(embedding) FROM items WHERE embedding IS NOT NULL LIMIT 1;"
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def parse_polish_to_sql(filter_str: str) -> Tuple[str, List[int]]:
    """Translate Polish predicate to SQL over tags (two-term only)."""
    tokens = filter_str.split()
    if len(tokens) != 3 or tokens[0] not in ("AND", "OR"):
        raise AssertionError(f"Unsupported filter: {filter_str}")
    op, a, b = tokens
    a_i, b_i = int(a), int(b)
    if op == "AND":
        return "tags @> ARRAY[%s] AND tags @> ARRAY[%s]", [a_i, b_i]
    else:
        return "tags @> ARRAY[%s] OR tags @> ARRAY[%s]", [a_i, b_i]


def exp_pgvector_complex(
    *,
    dsn: str | None = None,
    strategy: str = "hnsw",  # prefilter|ivf|hnsw
    iter_mode: str = "relaxed_order",
    templates: List[str] | None = None,
    # Dataset / GT
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    gt_cache_dir: str | Path = "data/ground_truth/complex_predicate",
    n_filters_per_template: int = 50,
    n_queries_per_filter: int = 100,
    k: int = 10,
    # IVF params
    lists: int | None = None,
    probes: int | None = None,
    # HNSW params
    m: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
    # Output
    output_path: str | None = None,
    # Control
    dry_run: bool = False,
    allow_gt_compute: bool = True,
):
    """AND/OR baselines for complex predicates (two-term): metrics per template."""
    dsn_resolved = resolve_dsn(dsn)
    print(f"[pgvector] Using DSN: {dsn_resolved}")
    if templates is None:
        templates = ["AND {0} {1}", "OR {0} {1}"]

    # Session GUCs
    gucs: Dict[str, object] = {}
    if strategy == "ivf":
        if probes is not None:
            gucs["ivfflat.probes"] = probes
        gucs["ivfflat.iterative_scan"] = iter_mode
    elif strategy == "hnsw":
        if ef_search is not None:
            gucs["hnsw.ef_search"] = ef_search
        gucs["hnsw.iterative_scan"] = iter_mode

    if dry_run:
        if gucs:
            print("[pgvector] Session GUCs (preview):", gucs)
        # Example SQL previews without DB connection
        pred_and = "tags @> ARRAY[$2] AND tags @> ARRAY[$3]"
        pred_or = "tags @> ARRAY[$2] OR tags @> ARRAY[$3]"
        core_sql = (
            "SELECT id, embedding <-> $1::vector AS distance FROM items WHERE {pred} "
            "ORDER BY distance LIMIT $4"
        )
        sqls = []
        for t in templates:
            if t.startswith("AND"):
                c = core_sql.format(pred=pred_and)
            elif t.startswith("OR"):
                c = core_sql.format(pred=pred_or)
            else:
                raise AssertionError(f"Unsupported template: {t}")
            sqls.append(strict_order_wrapper(c) if iter_mode == "relaxed_order" else c)
        print("[pgvector] Example query SQLs (preview):")
        for s in sqls:
            print("---\n", s)
        print("[pgvector] Dry-run: skipping DB connection and index checks.")
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
        ensure_indexes_for_strategy(conn, strategy)
        print(f"[pgvector] Index presence OK for strategy '{strategy}'.")

        # Load complex predicate dataset (uses cached GT, computes if allowed)
        cpd = ComplexPredicateDataset.from_dataset_key(
            dataset_key,
            test_size=test_size,
            templates=templates,
            n_filters_per_template=n_filters_per_template,
            n_queries_per_filter=n_queries_per_filter,
            gt_cache_dir=str(gt_cache_dir),
        )
        # Optional guard: ensure GT exists if compute is disallowed
        if not allow_gt_compute:
            # Heuristic check: require at least one GT list present
            if not cpd.filter_to_ground_truth:
                raise AssertionError(
                    "Missing complex predicate GT cache. Pre-generate via benchmark.complex_predicate.dataset.generate_dataset"
                )

        # Prepare SQL templates
        core_sql_tpl = (
            "SELECT id, embedding <-> %s::vector AS distance FROM items WHERE {pred} "
            "ORDER BY distance LIMIT %s"
        )

        def vec_to_literal(vec: np.ndarray) -> str:
            return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

        results: Dict[str, Dict[str, float]] = {}

        for template in sorted(cpd.templates):
            query_latencies: List[float] = []
            query_results: List[List[int]] = []

            for fstr in cpd.template_to_filters[template]:
                pred_sql, pred_params = parse_polish_to_sql(fstr)
                core_sql = core_sql_tpl.format(pred=pred_sql)
                sql = strict_order_wrapper(core_sql) if iter_mode == "relaxed_order" else core_sql
                gt_list = cpd.filter_to_ground_truth[fstr]

                with conn.cursor() as cur:
                    for qi, vec in enumerate(cpd.test_vecs):
                        vlit = vec_to_literal(vec)
                        start = time.perf_counter()
                        cur.execute(sql, (vlit, *pred_params, int(k)))
                        rows = cur.fetchall()
                        elapsed = time.perf_counter() - start
                        query_latencies.append(elapsed)
                        ids = [int(r[0]) for r in rows]
                        query_results.append(ids)

            # Compute metrics per template
            flat_gts = [gt for f in cpd.template_to_filters[template] for gt in cpd.filter_to_ground_truth[f]]
            rec = recall(query_results, flat_gts)
            lat = np.array(query_latencies, dtype=float)
            results[template] = {
                "recall_at_k": float(rec),
                "query_qps": float(1.0 / lat.mean()) if lat.size else 0.0,
                "query_lat_avg": float(lat.mean()) if lat.size else 0.0,
                "query_lat_p90": float(np.percentile(lat, 90)) if lat.size else 0.0,
                "query_lat_p95": float(np.percentile(lat, 95)) if lat.size else 0.0,
                "query_lat_p99": float(np.percentile(lat, 99)) if lat.size else 0.0,
            }

        # Emit JSON
        assert output_path is not None, "output_path is required"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_key": dataset_key,
                    "test_size": float(test_size),
                    "strategy": strategy,
                    "iter_mode": iter_mode,
                    "k": int(k),
                    "templates": sorted(cpd.templates),
                    "results": results,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[pgvector] Wrote results JSON to {out_path}")
    finally:
        conn.close()


def main() -> None:
    fire.Fire()


if __name__ == "__main__":
    main()
