"""CLI for pgvector database setup and index management."""

import fire

from scripts.pgvector import admin


class SetupDB:
    def create_schema(
        self,
        *,
        dsn: str,
        dim: int,
        schema: str = "int_array",
        label_ids: object | None = None,
    ) -> None:
        """Create extension, `items` table, and `GIN(tags)`.

        Args:
            dsn: Postgres DSN, e.g. `postgresql://user:pass@host:5432/db`.
            dim: Embedding dimension for `vector(dim)`.
            schema: Schema option; `int_array` or `boolean`.
            label_ids: Comma-separated top label IDs for boolean columns, e.g. "1,2,3".
        """
        lids = None
        if label_ids is not None:
            try:
                if isinstance(label_ids, (list, tuple)):
                    lids = [int(x) for x in label_ids]
                elif isinstance(label_ids, str):
                    lids = [int(x.strip()) for x in label_ids.split(",") if x.strip()]
                else:
                    # Fallback: attempt to coerce single value
                    lids = [int(label_ids)]
            except Exception as e:
                raise ValueError(f"invalid label_ids: {label_ids}") from e

        admin.create_schema(
            dsn,
            dim=dim,
            schema=schema,
            label_ids=lids,
            create_gin=(schema != "boolean"),
        )

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
        force: bool = False,
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
            force=force,
            output_json=output_json,
            output_csv=output_csv,
        )

    def create_all_boolean_labels(
        self,
        *,
        dsn: str,
    ) -> None:
        """Create boolean columns for all distinct labels and backfill.

        Args:
            dsn: Postgres DSN
        """
        admin.create_all_boolean_labels(dsn)


if __name__ == "__main__":
    fire.Fire(SetupDB)
