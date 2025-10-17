"""pgvector baselines (skeleton) for AND/OR complex predicates on YFCC 1M.

This initial commit provides:
- DSN handling with default and explicit flag
- Session GUC helpers (reuse logic)
- Strict-order SQL wrapper (for iterative relaxed modes)
- Index presence checks (GIN(tags), HNSW, IVFFlat)

Implementation is scaffold-only: supports `--dry_run` to preview SQL and
validate DB connectivity/index presence without running full benchmarks.
Subsequent commits will add execution and metrics emission.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import fire

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


def exp_pgvector_complex(
    *,
    dsn: str | None = None,
    strategy: str = "hnsw",  # prefilter|ivf|hnsw
    iter_mode: str = "relaxed_order",
    templates: List[str] | None = None,
    # IVF params
    lists: int | None = None,
    probes: int | None = None,
    # HNSW params
    m: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
    # Dataset/outputs (accepted, not used in skeleton execution)
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_path: str | None = None,
    # Control
    dry_run: bool = False,
):
    """AND/OR baselines (skeleton): verify DSN, indexes, and preview SQL.

    Only two-term templates are in scope: AND/OR.
    """
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
        print("[pgvector] Skeleton only: execution will be added in next commits.")
    finally:
        conn.close()


def main() -> None:
    fire.Fire()


if __name__ == "__main__":
    main()
