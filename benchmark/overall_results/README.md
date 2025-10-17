pgvector Baselines
------------------

This repository adds pgvector baselines for:
- Single-label search (this directory)
- AND/OR complex predicates (see `benchmark/complex_predicate/baselines/pgvector.py`)

DSN and Environment
- Default DSN: `postgresql://postgres:postgres@localhost:5432/curator_bench`
- Override by flag: `--dsn postgresql://user:pass@host:5432/db`
- Or set env var: `PG_DSN`

Database Setup (YFCC 1M)
1) Start pg18 + pgvector and create DB/extension (if not already running):
   - `docker run --name pgvector-dev -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d pgvector/pgvector:0.8.1-pg18-trixie`
   - `docker exec -it pgvector-dev psql -U postgres -c "CREATE DATABASE curator_bench;"`
   - `docker exec -it pgvector-dev psql -U postgres -d curator_bench -c "CREATE EXTENSION IF NOT EXISTS vector;"`
2) Create schema and indexes:
   - `psql -d postgresql://postgres:postgres@localhost:5432/curator_bench -c "DROP TABLE IF EXISTS items; CREATE TABLE items (id BIGINT PRIMARY KEY, tags INT[], embedding vector(192));"`
   - `psql -d postgresql://postgres:postgres@localhost:5432/curator_bench -c "CREATE INDEX IF NOT EXISTS items_tags_gin ON items USING GIN (tags);"`
   - `psql -d postgresql://postgres:postgres@localhost:5432/curator_bench -c "CREATE INDEX IF NOT EXISTS items_emb_hnsw ON items USING hnsw (embedding vector_l2_ops) WITH (m = 32, ef_construction = 64);"`

Optional Dataset Cache
- Precompute once for reuse: `python -m benchmark.overall_results.preproc_dataset --dataset_key yfcc100m --test_size 0.01 --output_dir data/cache`
- Otherwise, the loaders will use raw sources and load ground truth from `data/ground_truth` if available.

Single-Label Baseline (HNSW)
- Run (INT[] + GIN):
  - `python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single --strategy hnsw --m 32 --ef_construction 64 --ef_search 64 --iter_mode relaxed_order --schema int_array --dataset_key yfcc100m --test_size 0.01 --k 10 --output_path output/overall_results2/pgvector_hnsw/yfcc100m_test0.01/results.csv [--dataset_cache_path data/cache]`
- Outputs:
  - CSV: `output/overall_results2/pgvector_hnsw/yfcc100m_test0.01/results.csv`
  - JSON: `output/overall_results2/pgvector_hnsw/yfcc100m_test0.01/parameters.json`

Single-Label Baseline (IVFFlat)
- Requires non-empty table; build IVF index after load.
- Run (boolean wide-table; requires `label_<id>` columns):
  - `python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single --strategy ivf --lists 200 --probes 16 --iter_mode relaxed_order --schema boolean --dataset_key yfcc100m --test_size 0.01 --k 10 --output_path output/overall_results2/pgvector_ivf/yfcc100m_test0.01/results.csv`

Single-Label Baseline (Prefilter exact)
- Relational-only; no vector index.
- Run (INT[] + GIN only):
  - `python -m benchmark.overall_results.baselines.pgvector exp_pgvector_single --strategy prefilter --iter_mode strict_order --schema int_array --dataset_key yfcc100m --test_size 0.01 --k 10 --output_path output/overall_results2/pgvector_prefilter/yfcc100m_test0.01/results.csv`

Notes
- `GIN(tags)` must exist for the INT[] schema path. The boolean schema path uses `label_<id>` columns and does not require GIN.
- For HNSW/IVF strategies, the corresponding vector index must exist.
- YFCC uses 192-D embeddings end-to-end.

A/B Orchestrator (preview)
- To preview HNSW `strict_order` vs `relaxed_order` runs and output paths:
  - `python -m benchmark.pgvector_ab.pgvector_hnsw_ordering ab_single --dataset_variant yfcc100m_1m --dataset_key yfcc100m --test_size 0.01 --k 10 --m 32 --ef_construction 64 --ef_search 64 --dry_run true`
- The orchestrator prints the exact baseline commands to run and target paths under `output/pgvector/hnsw_ordering_ab/yfcc100m_1m/{strict_order|relaxed_order}/`.
