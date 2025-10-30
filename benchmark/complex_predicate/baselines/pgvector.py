"""pgvector AND/OR baselines for YFCC 1M.

- DSN + session GUC helpers
- Index presence checks (GIN, HNSW, IVFFlat)
- Iterative scans: use relaxed order and post-sort wrapper for strict results
- Metrics per template (recall@k, QPS, latency percentiles)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import psycopg2

from benchmark.complex_predicate.dataset import ComplexPredicateDataset
from benchmark.utils import recall

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/curator_bench"


def resolve_dsn(dsn: Optional[str]) -> str:
    return dsn or os.environ.get("PG_DSN", DEFAULT_DSN)


def set_session_gucs(conn, gucs: Dict[str, object]) -> None:
    if not gucs:
        return
    with conn.cursor() as cur:
        for k, v in gucs.items():
            cur.execute(f"SET {k} = %s;", (v,))


def strict_order_wrapper(core_sql: str) -> str:
    """Materialize relaxed results then sort by distance.

    Expects ``core_sql`` to select ``(id, distance)`` and include
    ``ORDER BY distance LIMIT %s``.
    """
    return (
        "WITH relaxed_results AS MATERIALIZED (\n"
        f"{core_sql}\n"
        ") SELECT id FROM relaxed_results ORDER BY distance + 0;"
    )


def _index_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = %s;",
            (name,),
        )
        return cur.fetchone() is not None


def _has_vector_index(conn, method: str) -> bool:
    q = (
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND tablename = 'items' "
        "AND indexdef ILIKE %s LIMIT 1;"
    )
    like = f"%USING {method}%"
    with conn.cursor() as cur:
        cur.execute(q, (like,))
        return cur.fetchone() is not None


def ensure_indexes_for_strategy(conn, strategy: str) -> None:
    if not _index_exists(conn, "items_tags_gin"):
        raise AssertionError(
            "Missing GIN(tags) index 'items_tags_gin'. Run scripts/pgvector.setup_db create_schema."
        )
    s = strategy.lower()
    if s == "prefilter":
        return
    if s == "ivf":
        if not _has_vector_index(conn, "ivfflat"):
            raise AssertionError(
                "Missing IVFFlat index on items.embedding. Create 'items_emb_ivf' via setup_db.create_index --index ivf."
            )
        return
    if s == "hnsw":
        if not _has_vector_index(conn, "hnsw"):
            raise AssertionError(
                "Missing HNSW index on items.embedding. Create 'items_emb_hnsw' via setup_db.create_index --index hnsw."
            )
        return
    raise AssertionError(f"Unsupported strategy: {strategy}")


def get_embedding_dim_from_db(conn) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT vector_dims(embedding) FROM items WHERE embedding IS NOT NULL LIMIT 1;"
        )
        row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def parse_polish_to_sql(filter_str: str) -> Tuple[str, List[int]]:
    """Translate Polish predicate to SQL over tags (two-term only)."""
    tokens = filter_str.split()
    if len(tokens) != 3 or tokens[0] not in ("AND", "OR"):
        raise AssertionError(f"Unsupported filter: {filter_str}")
    op, a, b = tokens
    a_i, b_i = int(a), int(b)
    if op == "AND":
        return "tags @> ARRAY[%s] AND tags @> ARRAY[%s]", [a_i, b_i]
    else:
        return "tags @> ARRAY[%s] OR tags @> ARRAY[%s]", [a_i, b_i]


def exp_pgvector_complex(
    *,
    dsn: str | None = None,
    strategy: str = "hnsw",  # prefilter|ivf|hnsw
    iter_search: bool = False,
    iter_search_mode: str | None = None,  # optional override: 'relaxed_order' | 'strict_order' | 'off'
    templates: List[str] | None = None,
    # Dataset / GT
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    gt_cache_dir: str | Path = "data/ground_truth/complex_predicate",
    n_filters_per_template: int = 50,
    n_queries_per_filter: int = 100,
    k: int = 10,
    # IVF params
    lists: int | None = None,
    probes: int | None = None,
    ivf_max_probes: int | None = None,
    ivf_overscan: int | None = None,
    # HNSW params
    m: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
    hnsw_max_scan_tuples: int | None = None,
    output_path: str | None = None,
    allow_gt_compute: bool = True,
):
    """Run AND/OR (two-term) baselines.

    - Requires iterative search for HNSW
    - If ``iter_search`` is True and strategy is HNSW/IVF, applies
      ``strict_order_wrapper`` for strict final ordering.
    """
    dsn_resolved = resolve_dsn(dsn)
    print(f"[pgvector] Using DSN: {dsn_resolved}")
    if templates is None:
        templates = ["AND {0} {1}", "OR {0} {1}"]

    # Normalize iterative-search controls
    if strategy == "hnsw" and not iter_search:
        raise AssertionError("HNSW requires iterative search; set --iter_search true.")
    if strategy == "ivf" and (not iter_search) and ivf_overscan:
        raise AssertionError(
            "ivf_overscan is only meaningful in iterative mode; set --iter_search true or unset ivf_overscan."
        )

    # Session GUCs
    gucs: Dict[str, object] = {}
    if strategy == "ivf":
        assert lists is not None and probes is not None, "lists and probes are required for ivf"
        gucs["ivfflat.probes"] = int(probes)
        if iter_search_mode is not None:
            gucs["ivfflat.iterative_scan"] = str(iter_search_mode)
        else:
            gucs["ivfflat.iterative_scan"] = "relaxed_order" if iter_search else "off"
        # Optional cap only applies to iterative mode
        if ivf_max_probes is not None and iter_search:
            gucs["ivfflat.max_probes"] = int(ivf_max_probes)
    elif strategy == "hnsw":
        assert ef_search is not None and m is not None and ef_construction is not None, \
            "ef_search, m, and ef_construction are required for hnsw"
        gucs["hnsw.ef_search"] = ef_search
        if iter_search_mode is not None:
            gucs["hnsw.iterative_scan"] = str(iter_search_mode)
        else:
            gucs["hnsw.iterative_scan"] = "relaxed_order" if iter_search else "off"
        if hnsw_max_scan_tuples is not None:
            gucs["hnsw.max_scan_tuples"] = int(hnsw_max_scan_tuples)

    # Connect, set GUCs, check indexes
    conn = psycopg2.connect(dsn_resolved)
    conn.autocommit = True
    try:
        set_session_gucs(conn, gucs)
        print("[pgvector] Session GUCs applied.")
        ensure_indexes_for_strategy(conn, strategy)
        print(f"[pgvector] Index presence OK for strategy '{strategy}'.")

        # Load complex predicate dataset (uses cached GT, computes if allowed)
        cpd = ComplexPredicateDataset.from_dataset_key(
            dataset_key,
            test_size=test_size,
            templates=templates,
            n_filters_per_template=n_filters_per_template,
            n_queries_per_filter=n_queries_per_filter,
            gt_cache_dir=str(gt_cache_dir),
        )
        # Optional guard: ensure GT exists if compute is disallowed
        if not allow_gt_compute:
            # Heuristic check: require at least one GT list present
            if not cpd.filter_to_ground_truth:
                raise AssertionError(
                    "Missing complex predicate GT cache. Pre-generate via benchmark.complex_predicate.dataset.generate_dataset"
                )

        # Prepare SQL templates
        core_sql_tpl = (
            "SELECT id, embedding <-> %s::vector AS distance FROM items WHERE {pred} "
            "ORDER BY distance LIMIT %s"
        )

        def vec_to_literal(vec: np.ndarray) -> str:
            return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"

        results: Dict[str, Dict[str, float]] = {}

        for template in sorted(cpd.templates):
            query_latencies: List[float] = []
            query_results: List[List[int]] = []

            for fstr in cpd.template_to_filters[template]:
                pred_sql, pred_params = parse_polish_to_sql(fstr)
                core_sql = core_sql_tpl.format(pred=pred_sql)
                # Iterative relaxed scan + post-sort for strict order (faster)
                sql = (
                    strict_order_wrapper(core_sql)
                    if (
                        iter_search
                        and strategy in {"hnsw", "ivf"}
                        and (iter_search_mode is None or str(iter_search_mode) == "relaxed_order")
                    )
                    else core_sql
                )

                with conn.cursor() as cur:
                    for qi, vec in enumerate(cpd.test_vecs):
                        vlit = vec_to_literal(vec)
                        inner_k = (
                            int(k) * int(ivf_overscan)
                            if (strategy == "ivf" and ivf_overscan and iter_search)
                            else int(k)
                        )

                        start = time.perf_counter()
                        cur.execute(sql, (vlit, *pred_params, inner_k))
                        rows = cur.fetchall()
                        elapsed = time.perf_counter() - start

                        query_latencies.append(elapsed)

                        if strategy == "ivf" and ivf_overscan and iter_search:
                            rows = rows[: int(k)]
                        ids = [int(r[0]) - 1 for r in rows]  # map DB ids (1-based) to 0-based
                        query_results.append(ids)

            # Compute metrics per template
            flat_gts = [
                gt
                for f in cpd.template_to_filters[template]
                for gt in cpd.filter_to_ground_truth[f]
            ]
            rec = recall(query_results, flat_gts)
            lat = np.array(query_latencies, dtype=float)
            results[template] = {
                "recall_at_k": float(rec),
                "query_qps": float(1.0 / lat.mean()) if lat.size else 0.0,
                "query_lat_avg": float(lat.mean()) if lat.size else 0.0,
                "query_lat_p90": float(np.percentile(lat, 90)) if lat.size else 0.0,
                "query_lat_p95": float(np.percentile(lat, 95)) if lat.size else 0.0,
                "query_lat_p99": float(np.percentile(lat, 99)) if lat.size else 0.0,
            }

        # Emit JSON
        assert output_path is not None, "output_path is required"
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_key": dataset_key,
                    "test_size": float(test_size),
                    "strategy": strategy,
                    "iter_search": bool(iter_search),
                    "k": int(k),
                    "templates": sorted(cpd.templates),
                    "results": results,
                },
                f,
                indent=2,
                sort_keys=True,
            )
        print(f"[pgvector] Wrote results JSON to {out_path}")
    finally:
        conn.close()


if __name__ == "__main__":
    fire.Fire()
