pgvector Baselines (Skeleton)
----------------------------

This repository adds pgvector baselines for:
- Single-label search (this directory)
- AND/OR complex predicates (see benchmark/complex_predicate/baselines/pgvector.py)

Status: initial skeleton
- DSN handling: default DSN plus explicit --dsn flag
- Session GUC utilities
- Strict-order SQL wrapper (for iterative relaxed modes)
- Index presence checks (GIN(tags), HNSW, IVFFlat)

DSN and Environment
- Default DSN: postgresql://postgres:postgres@localhost:5432/curator_bench
- Override by flag: --dsn postgresql://user:pass@host:5432/db
- Or set env var: PG_DSN

Setup (YFCC 1M)
1) Start pg18 + pgvector:
   docker run --name pgvector-dev -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:0.8.1-pg18-trixie
   docker exec -it pgvector-dev psql -U postgres -c "CREATE DATABASE curator_bench;"
   docker exec -it pgvector-dev psql -U postgres -d curator_bench -c "CREATE EXTENSION IF NOT EXISTS vector;"
2) Create schema and indexes as needed:
   python -m scripts.pgvector.setup_db create_schema --dsn postgresql://postgres:postgres@localhost:5432/curator_bench --dim 192
   # Build one vector index per strategy (example: HNSW)
   python -m scripts.pgvector.setup_db create_index --dsn postgresql://postgres:postgres@localhost:5432/curator_bench --index hnsw --m 32 --efc 64 --dim 192

Usage (skeleton / dry-run)
- Single-label (SQL preview only):
  python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single --strategy hnsw --ef_search 64 --iter_mode relaxed_order --dataset_key yfcc100m --test_size 0.01 --dry_run true

Notes
- This commit does not execute benchmarks or write results; later commits will add full runs and CSV/JSON outputs under output/overall_results2.

