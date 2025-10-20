"""Admin helpers for pgvector setup and index lifecycle.

This module provides a thin abstraction for:
- Creating schema objects (extension, table, relational index)
- Creating/dropping vector indexes (HNSW, IVFFLAT)
- Adding boolean label columns and backfilling from tags INT[]
- Measuring relation sizes (table and indexes) for storage comparisons
- Emitting profiling artifacts (JSON/CSV)

Use `--dry_run` for previewing SQL without connecting to the database
where supported.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional, Sequence


def _warn(msg: str) -> None:
    print(f"[pgvector][WARN] {msg}", file=sys.stderr)


def deterministic_index_name(index: str) -> str:
    """Return deterministic index name for the `items` table.

    - index: one of {"hnsw", "ivf"}
    """
    if index == "hnsw":
        return "items_emb_hnsw"
    if index == "ivf":
        return "items_emb_ivf"
    raise ValueError(f"unsupported index type: {index}")


@dataclass
class IndexBuildResult:
    status: str  # "ok", "deferred", or "dry_run"
    index_type: str
    index_name: str
    table: str = "items"
    build_time_seconds: Optional[float] = None
    index_size_bytes: Optional[int] = None
    dim: Optional[int] = None
    params: Optional[Dict[str, Any]] = None
    gucs: Optional[Dict[str, Any]] = None
    dataset_key: Optional[str] = None
    schema_option: Optional[str] = None
    timestamp: str = datetime.utcnow().isoformat()


def _emit_json_csv(
    result: IndexBuildResult,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> None:
    """Write JSON and/or CSV artifacts for a build run."""
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2, sort_keys=True)
        print(f"[pgvector] Wrote {output_json}")
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        row = asdict(result)
        # Flatten params/gucs for CSV
        flat_row: Dict[str, Any] = {
            **{k: v for k, v in row.items() if k not in {"params", "gucs"}},
        }
        if row.get("params"):
            for k, v in (row["params"] or {}).items():
                flat_row[f"param_{k}"] = v
        if row.get("gucs"):
            for k, v in (row["gucs"] or {}).items():
                flat_row[f"guc_{k}"] = v
        write_header = not os.path.exists(output_csv)
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(flat_row)
        print(f"[pgvector] Appended to {output_csv}")


def get_relation_size(dsn: str, relname: str) -> int:
    """Return `pg_relation_size(relname)`.

    Raises on connection or query errors.
    """
    try:
        import psycopg2  # type: ignore
    except Exception as e:  # pragma: no cover
        _warn(f"psycopg2 not available: {e}")
        raise
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_relation_size(%s);", (relname,))
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()


def get_total_relation_size(dsn: str, relname: str) -> int:
    """Return `pg_total_relation_size(relname)`.

    Useful for table-wide storage comparisons (includes TOAST, indexes).
    """
    try:
        import psycopg2  # type: ignore
    except Exception as e:  # pragma: no cover
        _warn(f"psycopg2 not available: {e}")
        raise
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_total_relation_size(%s);", (relname,))
            row = cur.fetchone()
            return int(row[0]) if row and row[0] is not None else 0
    finally:
        conn.close()


def _add_boolean_label_columns(conn, label_ids: Sequence[int]) -> None:
    """Add boolean columns label_<id> if not already present."""
    with conn.cursor() as cur:
        for lid in label_ids:
            col = f"label_{int(lid)}"
            cur.execute(
                f"ALTER TABLE items ADD COLUMN IF NOT EXISTS {col} BOOLEAN DEFAULT FALSE;"
            )


def _backfill_boolean_labels_from_tags(conn, label_ids: Sequence[int]) -> None:
    """Backfill boolean label columns from tags INT[] using ANY()."""
    with conn.cursor() as cur:
        for lid in label_ids:
            col = f"label_{int(lid)}"
            cur.execute(
                f"UPDATE items SET {col} = TRUE WHERE {col} = FALSE AND %s = ANY(tags);",
                (int(lid),),
            )


def create_all_boolean_labels(
    dsn: str,
    *,
    dry_run: bool = False,
) -> None:
    """Create boolean columns for all distinct labels in items.tags and backfill.

    This procedure:
      1) Discovers all distinct label IDs from items.tags
      2) Adds columns `label_<id> boolean DEFAULT false` for each ID
      3) Backfills each column to TRUE where the label is present in tags

    The implementation deliberately separates label discovery from DDL/DML to
    avoid ALTER TABLE while the table is being scanned in the same session.
    """
    plan_note = (
        "[pgvector] Plan: add boolean columns for all distinct labels in items.tags, "
        "then backfill from tags"
    )
    print(plan_note)
    if dry_run:
        print("[pgvector] Dry-run: no DB changes executed.")
        return

    try:
        import psycopg2  # type: ignore
    except Exception as e:  # pragma: no cover
        _warn(f"psycopg2 not available: {e}")
        raise

    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        # Phase 1: discover labels and close the query before any DDL
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT unnest(tags) AS lid FROM items ORDER BY lid;")
            lids = [int(row[0]) for row in cur.fetchall()]
        print(f"[pgvector] Distinct labels found: {len(lids)}")

        if not lids:
            print("[pgvector] No labels found; nothing to do.")
            return

        # Phase 2: add columns outside of any active scan
        _add_boolean_label_columns(conn, lids)

        # Phase 3: backfill values, then analyze
        _backfill_boolean_labels_from_tags(conn, lids)
        with conn.cursor() as cur:
            cur.execute("ANALYZE items;")
    finally:
        conn.close()
    print("[pgvector] Boolean columns created and backfilled for all labels.")


def create_schema(
    dsn: str,
    *,
    dim: int,
    schema: str = "option_a",
    create_gin: bool = True,
    unlogged: bool = False,
    dry_run: bool = False,
    label_ids: Optional[Iterable[int]] = None,
) -> None:
    """Create extension, `items` table, and relational index.

    Option A (default):
      - items(id BIGINT PRIMARY KEY, tags INT[], embedding vector(D))
      - CREATE INDEX items_tags_gin ON items USING GIN (tags)

    Boolean schema variant (schema='boolean'):
      - Same base table (includes tags INT[]) to enable backfill
      - Adds boolean columns label_<id> for provided label_ids and backfills
      - GIN creation is typically skipped for fair A/B storage comparison
    """
    if schema not in {"option_a", "boolean"}:
        _warn("unsupported schema; expected one of: option_a, boolean")
    table_kw = "UNLOGGED " if unlogged else ""
    sql = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        f"CREATE {table_kw}TABLE IF NOT EXISTS items (id BIGINT PRIMARY KEY, tags INT[], embedding vector({dim}));",
    ]
    if create_gin:
        sql.append("CREATE INDEX IF NOT EXISTS items_tags_gin ON items USING GIN (tags);")

    print("[pgvector] Planned schema SQL:")
    for stmt in sql:
        print("  ", stmt)

    if dry_run:
        return

    # Real execution path
    try:
        import psycopg2  # type: ignore
    except Exception as e:  # pragma: no cover
        _warn(f"psycopg2 not available: {e}")
        raise

    def _exec_all(dsn_: str, statements: Iterable[str]) -> None:
        conn = psycopg2.connect(dsn_)
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                for s in statements:
                    cur.execute(s)
        finally:
            conn.close()

    _exec_all(dsn, sql)
    print("[pgvector] Schema created or already exists (idempotent).")

    # Boolean labels (optional)
    if schema == "boolean" and label_ids:
        lids: Sequence[int] = [int(x) for x in label_ids]
        conn = psycopg2.connect(dsn)
        try:
            conn.autocommit = True
            _add_boolean_label_columns(conn, lids)
            _backfill_boolean_labels_from_tags(conn, lids)
        finally:
            conn.close()
        print("[pgvector] Boolean label columns added and backfilled from tags.")


def create_index(
    dsn: str,
    *,
    index: str,  # "hnsw" or "ivf" or "gin"
    dim: int,
    m: Optional[int] = None,
    efc: Optional[int] = None,
    lists: Optional[int] = None,
    opclass: str = "vector_l2_ops",
    force: bool = False,
    dry_run: bool = False,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> None:
    """Create a vector index and optionally emit profiling artifacts.

    In this scaffold, we print SQL and emit a placeholder result when
    `dry_run=True`. Full execution and timings land in later commits.
    """
    if index == "gin":
        idx_name = "items_tags_gin"
    else:
        idx_name = deterministic_index_name(index)
    sql: list[str] = []
    if index == "hnsw":
        if m is None or efc is None:
            _warn("hnsw requires parameters: m and efc")
        sql.append(
            f"CREATE INDEX IF NOT EXISTS {idx_name} ON items USING hnsw (embedding {opclass}) WITH (m = {m}, ef_construction = {efc});"
        )
    elif index == "ivf":
        if lists is None:
            _warn("ivf requires parameter: lists")
        sql.append(
            f"CREATE INDEX IF NOT EXISTS {idx_name} ON items USING ivfflat (embedding {opclass}) WITH (lists = {lists});"
        )
    elif index == "gin":
        sql.append(
            "CREATE INDEX IF NOT EXISTS items_tags_gin ON items USING GIN (tags);"
        )
    else:
        raise ValueError("index must be 'hnsw' or 'ivf' or 'gin'")

    print(f"[pgvector] Planned index SQL for {index}:")
    for stmt in sql:
        print("  ", stmt)

    if dry_run:
        result = IndexBuildResult(
            status="dry_run",
            index_type=index,
            index_name=idx_name,
            dim=dim,
            params={"m": m, "efc": efc, "lists": lists, "opclass": opclass},
        )
        _emit_json_csv(result, output_json=output_json, output_csv=output_csv)
        return

    # Real execution path
    try:
        import psycopg2  # type: ignore
    except Exception as e:  # pragma: no cover
        _warn(f"psycopg2 not available: {e}")
        raise

    def _exec(conn, stmt: str) -> None:
        with conn.cursor() as cur:
            cur.execute(stmt)

    def _drop_other_vector_indexes(conn) -> None:
        q = (
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = 'items' "
            "AND (indexdef ILIKE '%USING hnsw%' OR indexdef ILIKE '%USING ivfflat%')"
        )
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
        for (name,) in rows:
            if name != idx_name:
                _exec(conn, f'DROP INDEX IF EXISTS "{name}";')
        # Drop target as well to ensure fresh build
        _exec(conn, f'DROP INDEX IF EXISTS "{idx_name}";')

    conn = psycopg2.connect(dsn)
    try:
        conn.autocommit = True
        # For ivf, training requires existing data. If empty and not forced, defer.
        if index == "ivf":
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT 1 FROM items LIMIT 1);")
                has_rows = bool(cur.fetchone()[0])
            if not has_rows and not force:
                msg = (
                    "[pgvector] Table 'items' is empty; deferring ivfflat index build. "
                    "Load data first or pass --force to override."
                )
                print(msg)
                result = IndexBuildResult(
                    status="deferred",
                    index_type=index,
                    index_name=idx_name,
                    dim=dim,
                    params={"m": m, "efc": efc, "lists": lists, "opclass": opclass},
                )
                _emit_json_csv(result, output_json=output_json, output_csv=output_csv)
                return

        if index in {"hnsw", "ivf"}:
            _drop_other_vector_indexes(conn)
        else:
            # For GIN baseline, ensure fresh timing of the GIN index
            _exec(conn, 'DROP INDEX IF EXISTS "items_tags_gin";')

        start = time.monotonic()
        for stmt in sql:
            _exec(conn, stmt)
        build_time = time.monotonic() - start

        # Measure index size
        with conn.cursor() as cur:
            cur.execute("SELECT pg_relation_size(%s);", (idx_name,))
            size_bytes = cur.fetchone()[0]
    finally:
        conn.close()

    result = IndexBuildResult(
        status="ok",
        index_type=index,
        index_name=idx_name,
        dim=dim,
        build_time_seconds=build_time,
        index_size_bytes=size_bytes,
        params={"m": m, "efc": efc, "lists": lists, "opclass": opclass},
    )
    _emit_json_csv(result, output_json=output_json, output_csv=output_csv)
    print(
        f"[pgvector] Built {index} index {idx_name} in {build_time:.3f}s, size {size_bytes} bytes"
    )
