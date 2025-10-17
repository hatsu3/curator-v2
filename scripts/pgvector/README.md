pgvector Admin & Setup CLIs
=================================

Prerequisites
-------------
- Environment: ann_bench2 with psycopg2-binary installed
- Local Postgres with pgvector for tests:
  - docker run --name pgvector-dev -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:0.8.1-pg18-trixie
  - docker exec -it pgvector-dev psql -U postgres -c "CREATE DATABASE curator_bench;"
  - docker exec -it pgvector-dev psql -U postgres -d curator_bench -c "CREATE EXTENSION IF NOT EXISTS vector;"
- DSN used below: postgresql://postgres:postgres@localhost:5432/curator_bench

Schema Creation
---------------
- Create extension, table, and GIN(tags):
  - python -m scripts.pgvector.setup_db create_schema --dsn postgresql://... --dim 192 --schema option_a
- Dry-run preview without connecting:
  - python -m scripts.pgvector.setup_db create_schema --dsn postgresql://... --dim 192 --schema option_a --dry_run true

Index Build & Profiling
-----------------------
- HNSW (builds regardless of table state):
  - python -m scripts.pgvector.setup_db create_index --dsn postgresql://... --index hnsw --m 32 --efc 64 --dim 192 --output_json output/build/hnsw.json --output_csv output/build/hnsw.csv

- IVFFLAT (requires data; defers on empty by default):
  - python -m scripts.pgvector.setup_db create_index --dsn postgresql://... --index ivf --lists 200 --dim 192 --output_json output/build/ivf.json --output_csv output/build/ivf.csv
  - To override defer behavior: add --force true

Outputs & Conventions
---------------------
- Deterministic index names: items_emb_hnsw, items_emb_ivf
- JSON/CSV fields include: status, build_time_seconds, index_size_bytes, dim, params, timestamp
- Index isolation: non-target vector indexes are dropped before build to ensure clean profiling

Notes
-----
- For larger datasets, consider session GUCs (e.g., work_mem) before building; a helper will be added if needed
- Future: a loader may trigger ivfflat post-load automatically


Dataset Load & Insert Bench (skeleton)
--------------------------------------

- CLI module: scripts.pgvector.load_dataset
- Two subcommands:
  - bulk: prepare COPY stream and (optionally) time post-load index build
  - insert_bench: single-thread incremental inserts with durability toggles

Strict Copy Format
- copy_format: binary (default) or csv. No auto-fallback; if binary is unsupported in your environment, rerun with --copy_format csv.

Examples (dry-run preview)
- Bulk (prefilter; plan build of GIN):
  - python -m scripts.pgvector.load_dataset bulk \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --dataset yfcc100m --dataset_key yfcc100m-10m --dim 192 --test_size 0.001 \
      --copy_format binary --build_index gin --dry_run true

- Insert bench (HNSW):
  - python -m scripts.pgvector.load_dataset insert_bench \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --dataset yfcc100m --dataset_key yfcc100m --dim 192 --test_size 0.01 \
      --strategy hnsw --dry_run true
