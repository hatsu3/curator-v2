"""CLI for pgvector database setup and index management.

Commands are implemented with Python Fire and delegate to `admin`.
This file is a scaffold in commit 2: commands print SQL and warnings
unless `dry_run=True`. Subsequent commits will add real DB execution.
"""

from __future__ import annotations

import fire
from scripts.pgvector import admin


class SetupDB:
    def create_schema(
        self,
        *,
        dsn: str,
        dim: int,
        schema: str = "option_a",
        dry_run: bool = False,
    ) -> None:
        """Create extension, `items` table, and `GIN(tags)`.

        Args:
            dsn: Postgres DSN, e.g. `postgresql://user:pass@host:5432/db`.
            dim: Embedding dimension for `vector(dim)`.
            schema: Schema option; only `option_a` supported in scaffold.
            dry_run: If true, print SQL without connecting to DB.
        """
        admin.create_schema(dsn, dim=dim, schema=schema, dry_run=dry_run)

    def create_index(
        self,
        *,
        dsn: str,
        index: str,  # "hnsw" or "ivf"
        dim: int,
        m: int | None = None,
        efc: int | None = None,
        lists: int | None = None,
        opclass: str = "vector_l2_ops",
        dry_run: bool = False,
        output_json: str | None = None,
        output_csv: str | None = None,
    ) -> None:
        """Create a vector index and optionally emit artifacts.

        Args mirror underlying admin.create_index. See `--help` for usage.
        """
        admin.create_index(
            dsn,
            index=index,
            dim=dim,
            m=m,
            efc=efc,
            lists=lists,
            opclass=opclass,
            dry_run=dry_run,
            output_json=output_json,
            output_csv=output_csv,
        )


def main() -> None:
    fire.Fire(SetupDB)


if __name__ == "__main__":
    main()
