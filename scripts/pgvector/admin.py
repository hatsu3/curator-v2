"""Admin helpers for pgvector setup and index lifecycle.

This module provides a thin abstraction for:
- Creating schema objects (extension, table, relational index)
- Creating/dropping vector indexes (HNSW, IVFFLAT)
- Emitting profiling artifacts (JSON/CSV)

Commit 2 is a scaffold: functions print SQL and stub warnings unless
subsequent commits implement real DB execution. Use `--dry_run` for
previewing SQL without connecting to the database.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import csv
import json
import os
import sys
from datetime import datetime


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
        print(f"[pgvector] wrote {output_json}")
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
        print(f"[pgvector] appended {output_csv}")


def create_schema(
    dsn: str,
    *,
    dim: int,
    schema: str = "option_a",
    dry_run: bool = False,
) -> None:
    """Create extension, `items` table, and relational index.

    Option A (default):
      - items(id BIGINT PRIMARY KEY, tags INT[], embedding vector(D))
      - CREATE INDEX items_tags_gin ON items USING GIN (tags)

    In this scaffold, we print SQL and a stub warning; real execution
    will be implemented in subsequent commits.
    """
    if schema != "option_a":
        _warn("only schema=option_a is supported in the scaffold")
    sql = [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        f"CREATE TABLE IF NOT EXISTS items (id BIGINT PRIMARY KEY, tags INT[], embedding vector({dim}));",
        "CREATE INDEX IF NOT EXISTS items_tags_gin ON items USING GIN (tags);",
    ]

    print("[pgvector] Planned schema SQL:")
    for stmt in sql:
        print("  ", stmt)

    if not dry_run:
        _warn(
            "create_schema is a scaffold. Real DB execution will be added in a later commit."
        )


def create_index(
    dsn: str,
    *,
    index: str,  # "hnsw" or "ivf"
    dim: int,
    m: Optional[int] = None,
    efc: Optional[int] = None,
    lists: Optional[int] = None,
    opclass: str = "vector_l2_ops",
    dry_run: bool = False,
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
) -> None:
    """Create a vector index and optionally emit profiling artifacts.

    In this scaffold, we print SQL and emit a placeholder result when
    `dry_run=True`. Full execution and timings land in later commits.
    """
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
    else:
        raise ValueError("index must be 'hnsw' or 'ivf'")

    print(f"[pgvector] Planned index SQL for {index}:")
    for stmt in sql:
        print("  ", stmt)

    status = "dry_run" if dry_run else "ok"
    if not dry_run:
        _warn(
            f"create_index({index}) is a scaffold. Real DB execution and profiling will be added in a later commit."
        )
    result = IndexBuildResult(
        status=status,
        index_type=index,
        index_name=idx_name,
        dim=dim,
        params={"m": m, "efc": efc, "lists": lists, "opclass": opclass},
    )
    _emit_json_csv(result, output_json=output_json, output_csv=output_csv)

