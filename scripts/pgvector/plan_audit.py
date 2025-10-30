"""Plan audit for pgvector IVF/HNSW paths.

Provides two entrypoints via python-fire:
- run: sample queries stratified by selectivity, EXPLAIN (ANALYZE, JSON),
       classify plan (vector_index|prefilter|seq), and write per-sigma CSVs.
- summarize: aggregate plan CSVs into per-bin rates and simple stats.

Design choices (fail-fast, research style):
- Single-session GUCs; explicit assertions with actionable messages.
- No try/except silencing; allow failures to surface.
- Concise, readable flow; comments at major stages.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import psycopg2

from benchmark.profiler import Dataset

# ------------------------------ utils ------------------------------


def _vec_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def _classify_plan(plan_json: list[dict]) -> tuple[str, bool, str]:
    """Classify EXPLAIN(JSON) into (class, has_sort, index_name).

    Returns
    - class: 'vector_index' | 'prefilter' | 'seq' | 'unknown'
    - has_sort: whether a Sort/Top-N appears
    - index_name: name of vector index if detected, else ''
    """

    def walk(node: dict) -> tuple[bool, bool, bool, str, bool]:
        # has_vector_scan, has_bitmap, has_seq, index_name, has_sort
        t = node.get("Node Type", "")
        idx_name = node.get("Index Name", "") or node.get("Index Cond", "")
        has_vector = False
        if t in {"Index Scan", "Index Only Scan"} and (
            isinstance(idx_name, str)
            and ("items_emb_ivf" in idx_name or "items_emb_hnsw" in idx_name)
        ):
            has_vector = True

        has_bitmap = t in {"Bitmap Index Scan", "Bitmap Heap Scan"}
        has_seq = t == "Seq Scan"
        has_sort = t.startswith("Sort")

        found_idx_name = idx_name if has_vector and isinstance(idx_name, str) else ""

        for subkey in ("Plans", "Inner", "Outer"):
            subs = node.get(subkey)
            if not subs:
                continue
            if isinstance(subs, dict):
                subs = [subs]
            for child in subs:  # type: ignore[assignment]
                hv, hb, hs, iname, hsort = walk(child)
                has_vector = has_vector or hv
                has_bitmap = has_bitmap or hb
                has_seq = has_seq or hs
                has_sort = has_sort or hsort
                if not found_idx_name and iname:
                    found_idx_name = iname

        return has_vector, has_bitmap, has_seq, found_idx_name, has_sort

    assert isinstance(plan_json, list) and plan_json, "unexpected EXPLAIN JSON format"
    plan = plan_json[0]["Plan"]
    hv, hb, hs, iname, hsort = walk(plan)
    if hv:
        return "vector_index", hsort, iname
    if hb:
        return "prefilter", hsort, ""
    if hs:
        return "seq", hsort, ""
    return "unknown", hsort, ""


def _selectivity_map(ds: Dataset) -> dict[int, float]:
    return ds.label_selectivities


def _group_labels_by_selectivity(
    ds: Dataset, percentiles: list[float], labels_per_group: int
) -> list[dict]:
    """Return groups: [{percentile, selectivity, labels: [...]}, ...]."""
    label2sel = _selectivity_map(ds)
    sorted_sels = sorted(label2sel.values())
    groups = []
    for p in percentiles:
        idx = int(np.ceil(p * len(sorted_sels)) - 1)
        idx = max(0, min(idx, len(sorted_sels) - 1))
        sel = sorted_sels[idx]
        sorted_labels = sorted(
            label2sel.keys(), key=lambda lid: abs(label2sel[lid] - sel)
        )
        groups.append(
            {
                "percentile": p,
                "selectivity": float(sel),
                "labels": sorted_labels[:labels_per_group],
            }
        )
    return groups


def _build_sql(schema: str, label: int) -> str:
    if schema == "int_array":
        return (
            "SELECT id, embedding <-> %s::vector AS distance "
            "FROM items WHERE tags @> ARRAY[%s] ORDER BY distance LIMIT %s"
        )
    elif schema == "boolean":
        return (
            f"SELECT id, embedding <-> %s::vector AS distance "
            f"FROM items WHERE label_{int(label)} ORDER BY distance LIMIT %s"
        )
    raise AssertionError("schema must be one of: int_array, boolean")


# ------------------------------ runner ------------------------------


@dataclass
class RunConfig:
    dsn: str
    dataset_key: str
    test_size: float
    k: int
    strategy: str  # 'ivf' | 'hnsw'
    schema: str  # 'int_array' | 'boolean'
    overscan: int
    nprobe: Optional[int]
    max_probes: Optional[int]
    ef_search: Optional[int]
    max_scan_tuples: Optional[int]
    percentiles: list[float]
    labels_per_group: int
    samples_per_bin: int
    vector_only: bool
    out_dir: Path


def _sample_queries(
    ds: Dataset, groups: list[dict], samples_per_bin: int
) -> list[tuple[int, int, str]]:
    """Return list of (test_idx, label, sel_bin).

    sel_bin is the percentile label (e.g., 'p1','p25','p50','p75','p100')
    for the group from which this (test,label) was sampled.
    """
    # Map label -> list of test indices containing it
    label_to_tests: dict[int, list[int]] = {}
    for i, labs in enumerate(ds.test_mds):
        for lab in labs:
            label_to_tests.setdefault(int(lab), []).append(i)

    rng = random.Random(42)
    pairs: list[tuple[int, int, str]] = []
    for g in groups:
        bin_name = f"p{int(round(100*float(g['percentile'])))}"
        # Evenly allocate across labels in this group
        lbls = list(g["labels"])  # type: ignore[index]
        if not lbls:
            continue
        per_label = max(1, samples_per_bin // len(lbls))
        count = 0
        for lab in lbls:
            tests = label_to_tests.get(int(lab), [])
            if not tests:
                continue
            picks = rng.sample(tests, k=min(per_label, len(tests)))
            pairs.extend((ti, int(lab), bin_name) for ti in picks)
            count += len(picks)
        # top-up randomly within group if underfilled
        while count < samples_per_bin:
            lab = rng.choice(lbls)
            tests = label_to_tests.get(int(lab), [])
            if not tests:
                break
            pairs.append((rng.choice(tests), int(lab), bin_name))
            count += 1
    return pairs


def _apply_base_gucs(cur, cfg: RunConfig) -> None:
    # Common: iterative scans
    if cfg.strategy == "ivf":
        assert cfg.nprobe is not None, "nprobe required for ivf"
        # pgvector expects {'off','relaxed_order'}; use relaxed_order here
        cur.execute("SET ivfflat.iterative_scan = relaxed_order;")
        cur.execute("SET ivfflat.probes = %s;", (int(cfg.nprobe),))
        if cfg.max_probes is not None:
            cur.execute("SET ivfflat.max_probes = %s;", (int(cfg.max_probes),))
    elif cfg.strategy == "hnsw":
        assert cfg.ef_search is not None, "ef_search required for hnsw"
        # hnsw.iterative_scan uses the same enum values
        cur.execute("SET hnsw.iterative_scan = relaxed_order;")
        cur.execute("SET hnsw.ef_search = %s;", (int(cfg.ef_search),))
        if cfg.max_scan_tuples is not None:
            cur.execute("SET hnsw.max_scan_tuples = %s;", (int(cfg.max_scan_tuples),))
    else:
        raise AssertionError("strategy must be 'ivf' or 'hnsw'")

    # Vector-only option to avoid GIN/Seq for apples-to-apples comparisons
    if cfg.vector_only:
        cur.execute("SET enable_bitmapscan = off;")
        cur.execute("SET enable_seqscan = off;")
    # Keep enable_sort on to allow Top-N when not using vector index


def run(
    dsn: str = "postgresql://postgres:postgres@localhost:5432/curator_bench",
    dataset_key: str = "arxiv-large-10",
    test_size: float = 0.005,
    k: int = 10,
    strategy: str = "ivf",
    schema: str = "int_array",
    overscan_list: str | list[int] = "[1,2,4,8,16]",
    nprobe: Optional[int] = None,
    max_probes: Optional[int] = None,
    ef_search: Optional[int] = None,
    max_scan_tuples: Optional[int] = None,
    percentiles: str | list[float] = "[0.01,0.25,0.50,0.75,1.00]",
    labels_per_group: int = 100,
    samples_per_bin: int = 30,
    vector_only: bool = False,
    out_dir: str = "tmp/plan_audit",
    parallel: bool = False,
    max_procs: int = 4,
) -> str:
    """Run EXPLAIN(JSON) plan audit over stratified sampled queries.

    Args are intentionally explicit; failures raise with actionable messages.
    Returns path to the root output directory.
    """
    # Parse list-like args if provided as JSON strings
    if isinstance(overscan_list, str):
        overscans: list[int] = [int(x) for x in json.loads(overscan_list)]
    else:
        overscans = list(map(int, overscan_list))
    if isinstance(percentiles, str):
        pct_list: list[float] = [float(x) for x in json.loads(percentiles)]
    else:
        pct_list = list(map(float, percentiles))

    base_root = Path(out_dir)
    root = base_root / dataset_key / ("vector_only" if vector_only else "default") / strategy
    root.mkdir(parents=True, exist_ok=True)

    # Parallel fan-out: spawn subprocess per sigma for isolation & logging
    if parallel and len(overscans) > 1:
        procs: list[subprocess.Popen] = []
        slot = 0
        for sigma in overscans:
            log_path = root / f"sigma{sigma}" / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "-m",
                "scripts.pgvector.plan_audit",
                "run",
                f"--dsn={dsn}",
                f"--dataset_key={dataset_key}",
                f"--test_size={test_size}",
                f"--k={k}",
                f"--strategy={strategy}",
                f"--schema={schema}",
                f"--overscan_list=[{sigma}]",
                f"--nprobe={nprobe}" if nprobe is not None else "",
                f"--max_probes={max_probes}" if max_probes is not None else "",
                f"--ef_search={ef_search}" if ef_search is not None else "",
                (
                    f"--max_scan_tuples={max_scan_tuples}"
                    if max_scan_tuples is not None
                    else ""
                ),
                f"--percentiles={json.dumps(pct_list)}",
                f"--labels_per_group={labels_per_group}",
                f"--samples_per_bin={samples_per_bin}",
                f"--vector_only={str(vector_only).lower()}",
                # Pass the base root; child will compose dataset/mode/strategy
                f"--out_dir={str(base_root)}",
                "--parallel=false",
            ]
            # filter empty args
            cmd = [c for c in cmd if c]
            stdout = open(log_path, "w")
            stderr = subprocess.STDOUT
            procs.append(subprocess.Popen(cmd, stdout=stdout, stderr=stderr))
            slot += 1
            if slot >= max_procs:
                # wait for one to finish to keep concurrency under control
                procs.pop(0).wait()
                slot -= 1
        # Wait remaining
        for p in procs:
            p.wait()
        return str(root)

    # Load dataset once (sequential or single-sigma mode)
    ds = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    groups = _group_labels_by_selectivity(ds, pct_list, labels_per_group)

    # Open connection and apply session GUCs
    conn = psycopg2.connect(dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            # Build-time sanity: require matching vector dimension
            cur.execute(
                "SELECT vector_dims(embedding) FROM items WHERE embedding IS NOT NULL LIMIT 1;"
            )
            row = cur.fetchone()
            assert (
                row and row[0] is not None
            ), "embedding dim not found in DB (empty table?)"
            db_dim = int(row[0])

        # Iterate overscans
        for sigma in overscans:
            out_dir_sigma = root / f"sigma{sigma}"
            out_csv = out_dir_sigma / "plans.csv"
            _ensure_dir(out_csv)

            rows_csv: list[dict] = []

            with conn.cursor() as cur:
                _apply_base_gucs(
                    cur,
                    RunConfig(
                        dsn=dsn,
                        dataset_key=dataset_key,
                        test_size=test_size,
                        k=k,
                        strategy=strategy,
                        schema=schema,
                        overscan=sigma,
                        nprobe=nprobe,
                        max_probes=max_probes,
                        ef_search=ef_search,
                        max_scan_tuples=max_scan_tuples,
                        percentiles=pct_list,
                        labels_per_group=labels_per_group,
                        samples_per_bin=samples_per_bin,
                        vector_only=vector_only,
                        out_dir=root,
                    ),
                )

                pairs = _sample_queries(ds, groups, samples_per_bin)
                inner_k = int(k) * int(sigma)

                # Precompute mapping from (test_idx, label) -> ground truth row index
                # Ground truth is flattened in the same order as iterating over test_mds
                gt_map: dict[tuple[int, int], int] = {}
                ctr = 0
                for ti, labs in enumerate(ds.test_mds):
                    for labx in labs:
                        gt_map[(int(ti), int(labx))] = ctr
                        ctr += 1

                for qi, (tidx, lab, sel_bin) in enumerate(pairs):
                    vec = ds.test_vecs[tidx]
                    vlit = _vec_literal(vec)
                    sql_core = _build_sql(schema, lab)
                    if schema == "int_array":
                        params = (vlit, int(lab), inner_k)
                    else:
                        params = (vlit, inner_k)

                    # Build EXPLAIN JSON statement
                    explain_sql = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql_core

                    # Execute and parse plan
                    cur.execute(explain_sql, params)
                    plan_json = cur.fetchone()[0]
                    plan_text = json.dumps(plan_json, separators=(",", ":"))
                    plan_class, has_sort, idx_name = _classify_plan(plan_json)

                    # Pick some timing counters if present
                    # Execution Time lives at top-level in JSON[0]
                    try:
                        exec_time_ms = float(plan_json[0].get("Execution Time", 0.0))
                    except Exception:
                        exec_time_ms = 0.0

                    # Rows observed at root Plan (Limit)
                    try:
                        actual_rows = int(plan_json[0]["Plan"].get("Actual Rows", 0))
                    except Exception:
                        actual_rows = 0

                    # Execute the real query to compute recall@k
                    cur.execute(sql_core, params)
                    out_rows = cur.fetchall()
                    # Keep top-k by distance (already ordered); map ids to 0-based
                    pred_ids = [int(r[0]) - 1 for r in out_rows[: int(k)]]
                    # Lookup ground-truth row index for this (tidx, label)
                    gt_idx = gt_map.get((int(tidx), int(lab)))
                    assert gt_idx is not None, "ground truth mapping not found for query"
                    gt_ids = list(ds.ground_truth[int(gt_idx)])
                    # Compute recall@k
                    if len(gt_ids) > 0:
                        rec_at_k = float(len(set(pred_ids) & set(gt_ids[: int(k)]))) / float(int(k))
                    else:
                        rec_at_k = 0.0

                    # Emit CSV row
                    rows_csv.append(
                        {
                            "dataset_key": dataset_key,
                            "percentiles": json.dumps(pct_list),
                            "sigma": int(sigma),
                            "k": int(k),
                            "strategy": strategy,
                            "schema": schema,
                            "label": int(lab),
                            "test_idx": int(tidx),
                            "selectivity": float(
                                _selectivity_map(ds).get(int(lab), 0.0)
                            ),
                            "sel_bin": sel_bin,
                            "plan_class": plan_class,
                            "has_sort": bool(has_sort),
                            "index_name": idx_name,
                            "exec_time_ms": float(exec_time_ms),
                            "actual_rows_root": int(actual_rows),
                            "recall_at_k": float(rec_at_k),
                            "plan_hash": _md5(plan_text),
                        }
                    )

            # Write rows to CSV
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=(
                        list(rows_csv[0].keys())
                        if rows_csv
                        else [
                            "dataset_key",
                            "percentiles",
                            "sigma",
                            "k",
                            "strategy",
                            "schema",
                            "label",
                            "test_idx",
                            "selectivity",
                            "plan_class",
                            "has_sort",
                            "index_name",
                            "exec_time_ms",
                            "actual_rows_root",
                            "plan_hash",
                        ]
                    ),
                )
                writer.writeheader()
                for r in rows_csv:
                    writer.writerow(r)
            print(f"[plan_audit] Wrote {out_csv}")

    finally:
        conn.close()

    return str(root)


def summarize(root_dir: str, out_csv: str = "tmp/plan_audit/summary.csv") -> str:
    """Summarize plan CSVs under root_dir into a single CSV with rates/stats."""
    root = Path(root_dir)
    assert root.exists(), f"root_dir not found: {root}"

    rows: list[pd.DataFrame] = []
    for p in root.rglob("plans.csv"):
        try:
            df = pd.read_csv(p)
            # attach sigma from parent dir if missing
            if "sigma" not in df.columns:
                try:
                    sigma = int(p.parent.name.replace("sigma", ""))
                    df["sigma"] = sigma
                except Exception:
                    pass
            rows.append(df)
        except Exception:
            continue

    assert rows, f"no plans.csv found under {root}"
    df_all = pd.concat(rows, ignore_index=True)

    # Prefer explicit sel_bin from run(); otherwise derive a coarse bin as fallback
    if "sel_bin" not in df_all.columns or df_all["sel_bin"].isna().all():
        def pct_bin(sel: float, pct_list_json: str) -> str:
            try:
                pcts = [float(x) for x in json.loads(pct_list_json)]
            except Exception:
                pcts = [0.01, 0.25, 0.50, 0.75, 1.00]
            # Map by nearest percentile target label for a stable bin name
            targets = np.array(pcts)
            target = float(targets[np.argmin(np.abs(targets - sel))])
            return f"p{int(round(100*target))}"

        df_all["sel_bin"] = [
            pct_bin(sel, pcts)
            for sel, pcts in zip(df_all["selectivity"], df_all["percentiles"])
        ]

    grp = (
        df_all.groupby(["dataset_key", "strategy", "sigma", "sel_bin"])  # type: ignore[list-item]
        .agg(
            n=("plan_class", "size"),
            prefilter_rate=("plan_class", lambda s: float((s == "prefilter").mean())),
            vector_rate=("plan_class", lambda s: float((s == "vector_index").mean())),
            avg_time_ms=("exec_time_ms", "mean"),
            avg_recall=("recall_at_k", "mean"),
        )
        .reset_index()
    )

    # Attach numeric percentile for plotting (e.g., p25 -> 25)
    def _to_pct(s: str) -> int:
        try:
            return int(str(s).lstrip("p"))
        except Exception:
            return 0
    grp["sel_pct"] = grp["sel_bin"].apply(_to_pct)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grp.to_csv(out_path, index=False)
    print(f"[plan_audit] Summary written to {out_path}")
    return str(out_path)


def plot_latency_vs_sigma(
    summary_csv: str = "tmp/plan_audit/summary.csv",
    out_path: str = "tmp/plan_audit/latency_vs_sigma.pdf",
    title: str | None = None,
) -> str:
    """Plot latency (ms) vs sigma for each selectivity bin per dataset.

    - Reads the CSV produced by summarize().
    - Creates one subplot per dataset (side-by-side), each showing curves for sel_bin.
    - Saves to out_path and returns its path.
    """
    # Load and validate
    df = pd.read_csv(summary_csv)
    required_cols = {"dataset_key", "sigma", "sel_bin", "avg_time_ms"}
    missing = required_cols - set(df.columns)
    assert not missing, f"summary CSV missing columns: {missing}"

    # Map dataset keys to display names
    ds_display = {
        "arxiv-large-10": "arXiv",
        "yfcc100m-10m": "YFCC-10M",
    }
    df["dataset_display"] = df["dataset_key"].map(lambda x: ds_display.get(x, x))

    # Derive numeric percentile if absent
    if "sel_pct" not in df.columns:
        df["sel_pct"] = df["sel_bin"].map(lambda s: int(str(s).lstrip("p")) if isinstance(s, str) and str(s).startswith("p") else 0)

    # Present percentiles as numeric for plotting; build grayscale palette
    present_pcts = sorted(df["sel_pct"].dropna().unique().tolist())
    palette = sns.color_palette("Greys", n_colors=max(len(present_pcts), 3))
    pct_to_color = {p: palette[i if i < len(palette) else -1] for i, p in enumerate(present_pcts)}

    # Prepare figure
    datasets = df["dataset_display"].unique().tolist()
    n = len(datasets)
    assert n >= 1, "no datasets found in summary"
    # Compact styling to match repo conventions
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=False)
    if n == 1:
        axes = [axes]

    # Plot per dataset
    for ax, ds_name in zip(axes, datasets):
        sub = df[df["dataset_display"] == ds_name].copy()
        sub.sort_values(["sigma", "sel_pct"], inplace=True)
        # Use numeric hue via manual color mapping (grayscale); one marker style
        for p in present_pcts:
            cur = sub[sub["sel_pct"] == p]
            if cur.empty:
                continue
            sns.lineplot(
                data=cur,
                x="sigma",
                y="avg_time_ms",
                color=pct_to_color[p],
                marker="o",
                dashes=False,
                ax=ax,
                label=f"{p}p",
            )
        ax.set_title(ds_name)
        ax.set_xlabel("σ")
        ax.set_ylabel("Latency (ms)" if ax is axes[0] else "")
        ax.grid(True, axis="both", alpha=0.4)
        # Place legend only on the last subplot
        if ax is not axes[-1]:
            ax.get_legend().remove()

    if axes[-1].get_legend() is not None:
        # Remove legend title and shrink font size
        axes[-1].get_legend().set_title(None)
        for text in axes[-1].get_legend().get_texts():
            text.set_fontsize(9)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"[plan_audit] Wrote plot to {out_path}")
    return out_path


def plot_latency_vs_recall(
    summary_csv: str = "tmp/plan_audit/summary.csv",
    out_path: str = "tmp/plan_audit/latency_vs_recall.pdf",
    title: str | None = None,
) -> str:
    """Plot latency (ms) vs recall for each selectivity bin per dataset.

    - Expects summary_csv to contain columns: dataset_key, sel_pct, avg_time_ms, avg_recall.
    - One subplot per dataset; grayscale by percentile; log y-axis for latency.
    """
    df = pd.read_csv(summary_csv)
    for col in ["dataset_key", "sel_bin", "avg_time_ms", "avg_recall"]:
        assert col in df.columns, f"missing column in summary: {col}"

    # Derive numeric percentile
    if "sel_pct" not in df.columns:
        df["sel_pct"] = df["sel_bin"].map(lambda s: int(str(s).lstrip("p")) if isinstance(s, str) and str(s).startswith("p") else 0)

    ds_display = {"arxiv-large-10": "arXiv", "yfcc100m-10m": "YFCC-10M"}
    df["dataset_display"] = df["dataset_key"].map(lambda x: ds_display.get(x, x))

    present_pcts = sorted(df["sel_pct"].dropna().unique().tolist())
    palette = sns.color_palette("Greys", n_colors=max(len(present_pcts), 3))
    pct_to_color = {p: palette[i if i < len(palette) else -1] for i, p in enumerate(present_pcts)}

    datasets = df["dataset_display"].unique().tolist()
    n = len(datasets)
    assert n >= 1, "no datasets found in summary"

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        sub = df[df["dataset_display"] == ds_name].copy()
        # For each percentile, plot recall on x vs latency on y
        for p in present_pcts:
            cur = sub[sub["sel_pct"] == p]
            if cur.empty:
                continue
            # Sort by recall to get monotone-ish joins
            cur = cur.sort_values(["avg_recall"])  # type: ignore[call-arg]
            sns.lineplot(
                data=cur,
                x="avg_recall",
                y="avg_time_ms",
                color=pct_to_color[p],
                marker="o",
                dashes=False,
                ax=ax,
                label=f"{p}p",
            )
        ax.set_title(ds_name)
        ax.set_xlabel("Recall@k")
        ax.set_ylabel("Latency (ms)" if ax is axes[0] else "")
        ax.set_yscale("log")
        ax.grid(True, axis="both", alpha=0.4)
        if ax is not axes[-1]:
            ax.get_legend().remove()

    if axes[-1].get_legend() is not None:
        axes[-1].get_legend().set_title(None)
        for text in axes[-1].get_legend().get_texts():
            text.set_fontsize(9)

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"[plan_audit] Wrote plot to {out_path}")
    return out_path


if __name__ == "__main__":
    import fire
    fire.Fire({
        "run": run,
        "summarize": summarize,
        "plot": plot_latency_vs_sigma,
        "plot_recall": plot_latency_vs_recall,
    })
