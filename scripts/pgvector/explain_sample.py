"""Sample a few queries and print EXPLAIN results.

Parameters:
- dsn: Postgres DSN, e.g. postgresql://postgres:postgres@localhost:5434/curator_bench
- dataset_key: Dataset key, e.g. 'yfcc100m-10m' or 'arxiv-large-10'
- n: Number of (vector, label) queries to issue
- seed: RNG seed for reproducibility
- percentile: Target selectivity percentile (e.g., 0.01 for 1p, 0.001 for 0.1p)
- labels_per_group: Number of labels per selectivity group

This script uses the INT[] schema (`items.tags`) and issues label-filtered
KNN queries against `items(embedding)` with LIMIT k (default k=10).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns

from benchmark.overall_results.plotting.recall_vs_latency import (
    group_labels_by_selectivity,
)
from benchmark.profiler import Dataset

DEFAULTS = {
    "yfcc100m-10m": {
        "test_size": 0.001,
        "dim": 192,
        "nprobe_classic": [32, 64, 128, 256, 512],
        "overscan": [1, 2, 4, 8, 16],
    },
    "arxiv-large-10": {
        "test_size": 0.005,
        "dim": 384,
        "nprobe_classic": [8, 16, 32, 64, 128],
        "overscan": [1, 2, 4, 8, 16],
    },
}


def _vec_to_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in vec.tolist()) + "]"


@dataclass
class ExplainConfig:
    dsn: str
    dataset_key: str
    n: int = 5
    seed: int = 42
    k: int = 10
    test_size: Optional[float] = None
    percentile: Optional[float] = None
    labels_per_group: int = 10
    dataset_cache_path: Optional[str] = None


def run(
    dsn: str,
    dataset_key: str,
    n: int = 5,
    seed: int = 42,
    k: int = 10,
    test_size: float | None = None,
    percentile: float | None = None,
    labels_per_group: int = 10,
    dataset_cache_path: str | None = None,
) -> None:
    """Run EXPLAIN (ANALYZE, BUFFERS) on n sampled label-filtered queries.

    Args:
        dsn: Postgres connection string
        dataset_key: 'yfcc100m-10m' or 'arxiv-large-10'
        n: number of queries to issue
        seed: random seed
        k: top-k
        test_size: optional test split size; if None, uses dataset defaults
        percentile: target selectivity percentile (e.g., 0.01 for 1p, 0.001 for 0.1p);
                   if None, samples labels randomly without selectivity filtering
        labels_per_group: number of labels per selectivity group (default 10)
        dataset_cache_path: path to cached dataset; if None, uses default
                           data/preprocessed/{dataset_key}_test{test_size}
    """
    if dataset_key not in DEFAULTS:
        raise AssertionError(
            f"Unsupported dataset_key={dataset_key}. Expected one of {list(DEFAULTS.keys())}"
        )
    if test_size is None:
        test_size = float(DEFAULTS[dataset_key]["test_size"])

    # Use default cache path if not provided
    if dataset_cache_path is None:
        dataset_cache_path = f"data/preprocessed/{dataset_key}_test{test_size}"

    ds = Dataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        k=k,
        verbose=False,
        cache_path=dataset_cache_path,
    )
    total = len(ds.test_vecs)
    if total == 0:
        raise AssertionError("Empty test set; check dataset cache and test_size")

    rnd = random.Random(int(seed))

    # Group labels by selectivity if percentile is specified
    target_labels = None
    if percentile is not None:
        print(
            f"Grouping labels by selectivity (labels_per_group={labels_per_group})..."
        )
        label_groups = group_labels_by_selectivity(
            labels_per_group=labels_per_group,
            percentiles=[percentile],
            dataset_key=dataset_key,
            test_size=test_size,
            dataset=ds,
        )
        target_group = label_groups[0]
        target_labels = set(target_group["labels"])
        print(
            f"Target selectivity group: {percentile*100:.2f}p "
            f"(selectivity={target_group['selectivity']:.6f}, {len(target_labels)} labels)"
        )

        # Find queries containing any of the target labels
        valid_indices = []
        for qi in range(total):
            query_labels = set(ds.test_mds[qi])
            if query_labels & target_labels:  # intersection
                valid_indices.append(qi)

        if len(valid_indices) == 0:
            raise AssertionError(
                f"No queries found containing labels from {percentile*100:.2f}p selectivity group"
            )

        print(f"Found {len(valid_indices)} queries with target selectivity labels")
        indices = rnd.sample(valid_indices, k=min(int(n), len(valid_indices)))
    else:
        # Random sampling without selectivity filtering
        indices = rnd.sample(range(total), k=min(int(n), total))

    sql = (
        "EXPLAIN (ANALYZE, BUFFERS) "
        "SELECT id, embedding <-> %s::vector AS distance "
        "FROM items WHERE tags @> ARRAY[%s] ORDER BY distance LIMIT %s;"
    )

    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Enable per-session I/O timing to show I/O Timings in EXPLAIN
            cur.execute("SET track_io_timing = on;")
            for idx_i, qi in enumerate(indices, start=1):
                vec = ds.test_vecs[qi]
                labels: List[int] = list(ds.test_mds[qi])
                if not labels:
                    # Skip queries with no labels
                    continue

                # Filter to target labels if specified
                if target_labels is not None:
                    candidate_labels = list(set(labels) & target_labels)
                    if not candidate_labels:
                        continue
                    label = rnd.choice(candidate_labels)
                else:
                    label = rnd.choice(labels)

                vlit = _vec_to_literal(vec)
                selectivity = ds.label_selectivities.get(label, 0.0)

                print("=" * 80)
                print(
                    f"Query {idx_i}/{len(indices)} | label={label} | "
                    f"selectivity={selectivity:.6f} | k={k}"
                )
                cur.execute(sql, (vlit, int(label), int(k)))
                plan_rows = cur.fetchall()
                for (line,) in plan_rows:
                    print(line)


def _detect_prefiltering(plan_lines: List[str]) -> bool:
    """Detect if query plan uses pre-filtering (GIN index scan) or vector index.

    Pre-filtering indicators:
    - "Bitmap Index Scan on items_tags_gin" (GIN index for label filtering)
    - "Index Scan" on GIN index before vector operations

    Vector index indicators:
    - IVF index scan nodes
    - HNSW index scan nodes

    Returns:
        True if pre-filtering is detected, False otherwise
    """
    plan_text = "\n".join(plan_lines).lower()

    # Check for GIN index usage (pre-filtering)
    gin_indicators = [
        "bitmap index scan on items_tags_gin",
        "index scan on items_tags_gin",
        "gin index",
    ]

    for indicator in gin_indicators:
        if indicator in plan_text:
            return True

    return False


def analyze_query_plans(
    dsn: str,
    dataset_key: str,
    output_path: str | Path,
    n_per_group: int = 20,
    seed: int = 42,
    mode: str = "vary_k",
    k_values: List[int] | None = None,
    probes_values: List[int] | None = None,
    base_k: int = 10,
    percentiles: List[float] | None = None,
    labels_per_group: int = 10,
    dataset_cache_path: str | None = None,
) -> None:
    """Analyze query plans across selectivity levels and k/probes values.

    For each (selectivity_percentile, k) or (selectivity_percentile, probes) combination,
    samples queries and checks whether PostgreSQL uses pre-filtering or vector index scan.

    Args:
        dsn: Postgres connection string
        dataset_key: 'yfcc100m-10m' or 'arxiv-large-10'
        output_path: Path to save CSV results
        n_per_group: Number of queries per (percentile, param) group
        seed: Random seed
        mode: 'vary_k' to sweep k values, 'vary_probes' to sweep probes parameter
        k_values: List of k values to test (default: [10, 20, 40, 80, 160])
        probes_values: List of probes values (default: nprobe_classic from DEFAULTS)
        base_k: Base k value for vary_probes mode (k = base_k * overscan)
        percentiles: Selectivity percentiles (default: [0.01, 0.25, 0.50, 0.75, 1.00])
        test_size: Test split size
        labels_per_group: Number of labels per selectivity group
        dataset_cache_path: Path to cached dataset
    """
    if mode not in ["vary_k", "vary_probes"]:
        raise ValueError(f"mode must be 'vary_k' or 'vary_probes', got {mode}")

    if percentiles is None:
        percentiles = [0.01, 0.25, 0.50, 0.75, 1.00]

    if dataset_key not in DEFAULTS:
        raise AssertionError(
            f"Unsupported dataset_key={dataset_key}. Expected one of {list(DEFAULTS.keys())}"
        )
    test_size = DEFAULTS[dataset_key]["test_size"]

    # Load default parameter values if not provided
    if mode == "vary_k":
        if k_values is None:
            k_values = [
                base_k * overscan for overscan in DEFAULTS[dataset_key]["overscan"]
            ]
        param_values = k_values
        print(f"vary_k mode: base_k={base_k}, k_values={k_values}")
    else:  # vary_probes
        if probes_values is None:
            probes_values = DEFAULTS[dataset_key]["nprobe_classic"]
        param_values = probes_values
        print(f"vary_probes mode: probes_values={probes_values}")
    
    assert isinstance(param_values, list), f"param_values must be a list, got {type(param_values)}"

    # Use default cache path if not provided
    if dataset_cache_path is None:
        dataset_cache_path = f"data/preprocessed/{dataset_key}_test{test_size}"

    print(f"Loading dataset from cache: {dataset_cache_path}")
    ds = Dataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        k=10,
        verbose=False,
        cache_path=dataset_cache_path,
    )
    total = len(ds.test_vecs)
    if total == 0:
        raise AssertionError("Empty test set; check dataset cache and test_size")

    rnd = random.Random(int(seed))

    # Group labels by selectivity
    print(f"Grouping labels by selectivity (labels_per_group={labels_per_group})...")
    label_groups = group_labels_by_selectivity(
        labels_per_group=labels_per_group,
        percentiles=percentiles,
        dataset_key=dataset_key,
        test_size=test_size,
        dataset=ds,
    )

    results = []

    with psycopg2.connect(dsn) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Enable iterative scan for vary_k mode (matches benchmark configuration)
            if mode == "vary_k":
                cur.execute("SET ivfflat.iterative_scan = relaxed_order;")
                print("Enabled ivfflat.iterative_scan = relaxed_order for vary_k mode")

            for group in label_groups:
                percentile = group["percentile"]
                selectivity = group["selectivity"]
                target_labels = set(group["labels"])

                print(
                    f"\nProcessing {percentile*100:.1f}p (selectivity={selectivity:.6f}, {len(target_labels)} labels)"
                )

                # Find queries containing target labels
                valid_indices = []
                for qi in range(total):
                    query_labels = set(ds.test_mds[qi])
                    if query_labels & target_labels:
                        valid_indices.append(qi)

                if len(valid_indices) == 0:
                    print(f"  WARNING: No queries found for {percentile*100:.1f}p")
                    continue

                # Sample queries
                sampled_indices = rnd.sample(
                    valid_indices, k=min(n_per_group, len(valid_indices))
                )

                for idx, param_val in enumerate(param_values):
                    prefilter_count = 0

                    # Set k and probes based on mode
                    if mode == "vary_k":
                        k = param_val
                        probes = None
                    else:  # vary_probes
                        k = base_k
                        probes = param_val
                        cur.execute(f"SET ivfflat.probes = {probes};")

                    for qi in sampled_indices:
                        vec = ds.test_vecs[qi]
                        labels_list = list(ds.test_mds[qi])
                        candidate_labels = list(set(labels_list) & target_labels)

                        if not candidate_labels:
                            continue

                        label = rnd.choice(candidate_labels)
                        vlit = _vec_to_literal(vec)

                        # Run EXPLAIN
                        sql = (
                            "EXPLAIN "
                            "SELECT id, embedding <-> %s::vector AS distance "
                            "FROM items WHERE tags @> ARRAY[%s] ORDER BY distance LIMIT %s;"
                        )
                        cur.execute(sql, (vlit, int(label), int(k)))
                        plan_rows = cur.fetchall()
                        plan_lines = [line for (line,) in plan_rows]

                        # Detect pre-filtering
                        is_prefilter = _detect_prefiltering(plan_lines)
                        if is_prefilter:
                            prefilter_count += 1

                    prefilter_freq = prefilter_count / len(sampled_indices)

                    result_row = {
                        "percentile": percentile,
                        "selectivity": selectivity,
                        "k": k,
                        "n_queries": len(sampled_indices),
                        "prefilter_count": prefilter_count,
                        "prefilter_freq": prefilter_freq,
                    }
                    if mode == "vary_probes":
                        result_row["probes"] = probes
                    results.append(result_row)

                    if mode == "vary_k":
                        print(
                            f"  k={k:3d}: {prefilter_count}/{len(sampled_indices)} "
                            f"({prefilter_freq*100:.1f}%) use pre-filtering"
                        )
                    else:
                        print(
                            f"  probes={probes:3d}: {prefilter_count}/{len(sampled_indices)} "
                            f"({prefilter_freq*100:.1f}%) use pre-filtering"
                        )

    # Save results
    df = pd.DataFrame(results)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")


def plot_prefilter_frequency(
    csv_path: str,
    output_path: str | Path,
    mode: str = "vary_k",
    figsize: tuple = (5, 3),
    font_size: int = 12,
) -> None:
    """Plot pre-filtering frequency vs selectivity percentile.

    Args:
        csv_path: Path to CSV file from analyze_query_plans
        output_path: Path to save PDF plot
        mode: 'vary_k' or 'vary_probes' to determine which parameter to plot
        figsize: Figure size (width, height)
        font_size: Font size for plot
    """
    df = pd.read_csv(csv_path)

    # Determine which parameter to use for hue
    if mode == "vary_k":
        hue_col = "k"
        legend_title = "k"
        title = "Pre-Filtering Strategy vs Selectivity and k"
    else:  # vary_probes
        hue_col = "probes"
        legend_title = "probes"
        title = "Pre-Filtering Strategy vs Selectivity and Probes"

    # Set up the plot
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=figsize)

    # Create line plot with parameter as hue
    sns.lineplot(
        data=df,
        x="percentile",
        y="prefilter_freq",
        hue=hue_col,
        marker="o",
        dashes=False,
        ax=ax,
    )

    # Format x-axis as percentile labels
    ax.set_xlabel("Selectivity Percentile")
    ax.set_ylabel("Pre-Filtering Frequency")
    ax.set_title(title)

    # Set explicit x-ticks at data percentiles
    ax.set_xlim(0.0, 1.0)
    x_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(x*100)}p" for x in x_ticks])

    # Set y-axis range
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

    # Add grid
    ax.grid(axis="both", alpha=0.3)

    # Move legend to the right outside plot with smaller font
    ax.legend(
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="medium",
        title_fontsize="medium",
    )

    fig.tight_layout()

    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    fire.Fire(
        {"run": run, "analyze": analyze_query_plans, "plot": plot_prefilter_frequency}
    )
