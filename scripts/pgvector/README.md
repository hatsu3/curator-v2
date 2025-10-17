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

- Boolean label columns (Option B: wide table with label_<id>):
  - Create base table without GIN and add boolean columns for top labels:
    - python -m scripts.pgvector.setup_db create_schema \
        --dsn postgresql://... --dim 192 --schema boolean --label_ids 1,2,3,4,5
  - Dry-run preview:
    - python -m scripts.pgvector.setup_db create_schema \
        --dsn postgresql://... --dim 192 --schema boolean --label_ids 1,2,3 --dry_run true

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
- For IVF, training requires existing rows; build the index post-load via setup CLI before insert benchmarking


Dataset Load & Insert Bench
--------------------------

- CLI module: scripts.pgvector.load_dataset
- Subcommands:
  - bulk: execute COPY (binary or csv) and optionally time post-load index build
  - insert_bench: single-thread inserts with durability toggles; for IVF, build index separately first

Durability Modes (insert_bench)
- Durable: LOGGED table + synchronous_commit=on (default)
- Non-durable: UNLOGGED table (preferred) or synchronous_commit=off

Copy Format (bulk)
- copy_format: binary (default) or csv. No auto-fallback; if binary is unsupported in your environment, rerun with --copy_format csv.
- create_gin: set to false to suppress creating GIN(tags) during bulk load (useful for boolean schema DB or when timing GIN build separately).

Examples
- Real bulk (prefilter; build GIN timing):
  - python -m scripts.pgvector.load_dataset bulk \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --dataset yfcc100m --dataset_key yfcc100m-10m --dim 192 --test_size 0.001 \
      --copy_format binary --build_index gin

- Real bulk (boolean schema DB; no GIN):
  - python -m scripts.pgvector.load_dataset bulk \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bool \
      --dataset yfcc100m --dataset_key yfcc100m --dim 192 --test_size 0.01 \
      --copy_format csv --create_gin false

- Dry-run bulk preview:
  - python -m scripts.pgvector.load_dataset bulk \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --dataset yfcc100m --dataset_key yfcc100m-10m --dim 192 --test_size 0.001 \
      --copy_format binary --build_index gin --dry_run true

- Insert bench (HNSW; durable):
  - python -m scripts.pgvector.load_dataset insert_bench \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --dataset yfcc100m --dataset_key yfcc100m --dim 192 --test_size 0.01 \
      --strategy hnsw --m 32 --efc 64

- Insert bench (HNSW; non-durable via UNLOGGED):
  - python -m scripts.pgvector.load_dataset insert_bench \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --dataset yfcc100m --dataset_key yfcc100m --dim 192 --test_size 0.01 \
      --strategy hnsw --m 32 --efc 64 --unlogged true

- Insert bench (IVF; durable; build index first):
  - python -m scripts.pgvector.setup_db create_index \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --index ivf --dim 192 --lists 200
  - python -m scripts.pgvector.load_dataset insert_bench \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --dataset yfcc100m --dataset_key yfcc100m --dim 192 --test_size 0.01 \
      --strategy ivf --lists 200

- Insert bench (prefilter/GIN; durable):
  - python -m scripts.pgvector.load_dataset insert_bench \
      --dsn postgresql://postgres:postgres@localhost:5432/curator_bench \
      --dataset yfcc100m --dataset_key yfcc100m --dim 192 --test_size 0.01 \
      --strategy prefilter

Artifacts
- Insert metrics written under canonical paths:
  - output/overall_results2/pgvector_{hnsw|ivf|prefilter}/<dataset_key>_test<test_size>/insert_{durable|non_durable}.{json,csv}
 - A/B artifacts for insert ablations:
   - output/pgvector/insert_ab/<dataset_key>/{hnsw|ivf|gin}/{durable|non_durable}/run.{json,csv}

Label Modeling A/B (INT[] + GIN vs Boolean)
-------------------------------------------
- Orchestrator (preview commands and outputs):
  - eval "$(conda shell.bash hook)" && conda activate ann_bench2
  - python -m benchmark.pgvector_ab.label_model_ab run \
      --dsn_int postgresql://postgres:postgres@localhost:5432/curator_int \
      --dsn_bool postgresql://postgres:postgres@localhost:5432/curator_bool \
      --dataset_key yfcc100m --test_size 0.01 --k 10 \
      --lists 200 --probes 16 --m 32 --ef_construction 64 --ef_search 64 \
      --dry_run true
- Outputs:
  - output/pgvector/label_ab/yfcc100m_1m/{int_array|boolean}/{hnsw|ivf}/results.csv
- Summary (optional, after runs):
  - python -m benchmark.pgvector_ab.summarize_label_ab summarize --dataset_variant yfcc100m_1m --dry_run true
