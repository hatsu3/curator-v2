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

