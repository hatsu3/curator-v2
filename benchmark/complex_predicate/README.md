pgvector Complex Predicates (AND/OR)
-----------------------------------

This module adds the pgvector baseline for two-term AND/OR complex predicates.
It uses `ComplexPredicateDataset` to prepare filters, queries, and cached ground truth.

Prerequisites
- Postgres (pg18) with pgvector extension
- `items(id BIGINT PRIMARY KEY, tags INT[], embedding vector(192))`
- Indexes:
  - `CREATE INDEX IF NOT EXISTS items_tags_gin ON items USING GIN (tags);`
  - For HNSW: `CREATE INDEX IF NOT EXISTS items_emb_hnsw ON items USING hnsw (embedding vector_l2_ops) WITH (m = 32, ef_construction = 64);`
  - For IVF: `CREATE INDEX IF NOT EXISTS items_emb_ivf ON items USING ivfflat (embedding vector_l2_ops) WITH (lists = <L>);` (requires non-empty table)

DSN
- Default: `postgresql://postgres:postgres@localhost:5432/curator_bench`
- Override with `--dsn` or set `PG_DSN` env var

Run (HNSW)
- `python -m benchmark.complex_predicate.baselines.pgvector exp_pgvector_complex \\
    --strategy hnsw --m 32 --ef_construction 64 --ef_search 64 --iter_mode relaxed_order \\
    --dataset_key yfcc100m --test_size 0.01 --k 10 \\
    --n_filters_per_template 50 --n_queries_per_filter 100 \\
    --gt_cache_dir data/ground_truth/complex_predicate \\
    --output_path output/complex_predicate_optimal/pgvector_hnsw/yfcc100m_test0.01/results.json`

Run (IVFFlat)
- `python -m benchmark.complex_predicate.baselines.pgvector exp_pgvector_complex \\
    --strategy ivf --lists 200 --probes 16 --iter_mode relaxed_order \\
    --dataset_key yfcc100m --test_size 0.01 --k 10 \\
    --n_filters_per_template 50 --n_queries_per_filter 100 \\
    --gt_cache_dir data/ground_truth/complex_predicate \\
    --output_path output/complex_predicate_optimal/pgvector_ivf/yfcc100m_test0.01/results.json`

Run (Prefilter exact)
- `python -m benchmark.complex_predicate.baselines.pgvector exp_pgvector_complex \\
    --strategy prefilter --iter_mode strict_order \\
    --dataset_key yfcc100m --test_size 0.01 --k 10 \\
    --n_filters_per_template 50 --n_queries_per_filter 100 \\
    --gt_cache_dir data/ground_truth/complex_predicate \\
    --output_path output/complex_predicate_optimal/pgvector_prefilter/yfcc100m_test0.01/results.json`

Outputs
- JSON at `--output_path` with per-template metrics:
  - `recall_at_k`, `query_qps`, `query_lat_avg/p50/p95/p99`

Notes
- `ComplexPredicateDataset` computes and caches ground truth under `--gt_cache_dir` if missing.
- For reproducibility, keep the dataset and DB schema aligned to 192-D embeddings for YFCC.

A/B Orchestrator (preview)
- To preview HNSW `strict_order` vs `relaxed_order` commands and output paths for complex predicates:
  - `python -m benchmark.pgvector_ab.pgvector_hnsw_ordering ab_complex --dataset_variant yfcc100m_1m --dataset_key yfcc100m --test_size 0.01 --k 10 --m 32 --ef_construction 64 --ef_search 64 --n_filters_per_template 50 --n_queries_per_filter 100 --dry_run true`
- The orchestrator prints baseline commands and target paths under `output/pgvector/hnsw_ordering_ab/yfcc100m_1m/{strict_order|relaxed_order}/`.
