import json
from functools import cache
from pathlib import Path
from typing import Literal

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from benchmark.selectivity.dataset import SelectivityDataset


@cache
def load_dataset(dataset_cache_dir: str = "data/selectivity/random_yfcc100m"):
    return SelectivityDataset.load(dataset_cache_dir)


def select_best_result(
    output_dir: str = "output/selectivity/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    max_selectivity: float = 1.0,
    min_recall: float = 0.8,
    max_recall: float = 0.9,
    max_latency: float = 1e-2,
):
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    dataset = load_dataset(dataset_cache_dir)

    agg_results = list()
    json_files = list(results_dir.glob("*.json"))
    for json_file in tqdm(
        json_files, total=len(json_files), desc=f"Processing results in {results_dir}"
    ):
        preproc_res_path = json_file.with_suffix(".csv")
        if not preproc_res_path.exists():
            preproc_res = pd.concat(
                [
                    preprocess_per_config_result(result, dataset, max_selectivity)
                    for result in json.load(open(json_file))
                ]
            )
            print(f"Saving preprocessed results to {preproc_res_path} ...")
            preproc_res.to_csv(preproc_res_path, index=False)
        else:
            print(f"Loading preprocessed results from {preproc_res_path} ...")
            preproc_res = pd.read_csv(preproc_res_path)

        agg_results.append(preproc_res)

    agg_results_df = pd.concat(agg_results)
    filtered_df = agg_results_df[
        (agg_results_df["recall"] >= min_recall)
        & (agg_results_df["recall"] <= max_recall)
        & (agg_results_df["latency"] <= max_latency)
    ]
    best_results_df = (
        filtered_df.groupby("selectivity")
        .apply(lambda group: group.loc[group["latency"].idxmin()])  # type: ignore
        .reset_index(drop=True)
    )

    return best_results_df


def preprocess_per_config_result(
    results: dict, dataset: SelectivityDataset, max_selectivity: float = 1.0
):
    stats_gen = iter(
        results.get("query_stats", [{"n_dists": 1} for _ in results["query_recalls"]])
    )
    recalls_gen = iter(results["query_recalls"])
    lats_gen = iter(results["query_latencies"])

    per_selectivity_results = {
        sel: {
            "n_dists": list(),
            "recalls": list(),
            "latencies": list(),
        }
        for sel in dataset.all_selectivities
    }
    for __, access_list in zip(dataset.test_vecs, dataset.test_mds):
        for label in access_list:
            sel = dataset.label_to_selectivity[label]
            if sel > max_selectivity:
                continue
            stats, recall, latency = next(stats_gen), next(recalls_gen), next(lats_gen)
            per_selectivity_results[sel]["n_dists"].append(
                stats.get("n_dists", None) or stats["n_ndists"]
            )
            per_selectivity_results[sel]["recalls"].append(recall)
            per_selectivity_results[sel]["latencies"].append(latency)

    per_selectivity_results_agg = pd.DataFrame(
        {
            sel: {
                "n_dists": np.mean(res["n_dists"]).item(),
                "recall": np.mean(res["recalls"]).item(),
                "latency": np.mean(res["latencies"]).item(),
                "memory_usage_kb": results["index_size_kb"],
            }
            for sel, res in per_selectivity_results.items()
        }
    ).T.reset_index(names=["selectivity"])

    return per_selectivity_results_agg


def plot_overall_results(
    index_keys: list[str] = [
        "curator_opt",
        "shared_hnsw",
        "pre_filtering",
        "filtered_diskann",
        "hybrid_curator",
    ],
    index_keys_readable: list[str] = [
        "Curator",
        "Shared HNSW",
        "Pre-Filtering",
        "Filtered DiskANN",
        "Hybrid Curator",
    ],
    output_dir: str = "output/selectivity",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
    pre_filtering_max_sel: float = 0.1,
    min_recall: float = 0.8,
    max_recall: float = 1.0,
    max_latency: float = 0.02,
    output_path: str = "output/selectivity/figs/overall_results.pdf",
):
    best_results: dict[str, pd.DataFrame] = dict()
    for index_key in index_keys:
        max_sel = pre_filtering_max_sel if index_key == "pre_filtering" else 1.0
        best_results[index_key] = select_best_result(
            output_dir=str(Path(output_dir) / index_key),
            dataset_key=dataset_key,
            test_size=test_size,
            dataset_cache_dir=dataset_cache_dir,
            max_selectivity=max_sel,
            min_recall=min_recall,
            max_recall=max_recall,
            max_latency=max_latency,
        )

    df = pd.concat(
        [
            results.assign(index_type=index_type)
            for index_type, results in best_results.items()
        ]
    ).reset_index(drop=True)
    df["index_type"] = df["index_type"].map(dict(zip(index_keys, index_keys_readable)))
    df["latency_ms"] = df["latency"] * 1000

    memory_usage_results_path = output_path.replace(".pdf", ".csv")
    print(f"Saving memory usage results to {memory_usage_results_path} ...")
    memory_usage_df = (
        df.groupby(["index_type", "selectivity"])
        .agg({"memory_usage_kb": "mean"})
        .reset_index()
    )
    memory_usage_df.to_csv(memory_usage_results_path, index=False)

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.lineplot(
        data=df,
        x="selectivity",
        y="latency_ms",
        hue="index_type",
        style="index_type",
        markers=True,
        dashes=False,
        hue_order=index_keys_readable,
        errorbar=None,
        markersize=6,
        linewidth=1.5,
        ax=ax,
    )
    ax.set_xlabel("Filter Selectivity")
    ax.set_ylabel("Query Latency (ms)")
    ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    legend = plt.legend(
        ax.get_legend().legend_handles,
        index_keys_readable,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.4),
        ncols=(len(index_keys_readable) + 1) // 2,
        fontsize="small",
    )

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


def aggregate_per_config_per_sel_results(
    output_path: str,
    index_key: Literal["curator_opt", "shared_hnsw"],
    results_dir: str | Path = "output/selectivity",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    dataset_cache_dir: str = "data/selectivity/random_yfcc100m",
):
    results_dir = (
        Path(results_dir) / index_key / f"{dataset_key}_test{test_size}" / "results"
    )
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    dataset = load_dataset(dataset_cache_dir)

    agg_results = list()
    json_files = list(results_dir.glob("*.json"))
    for json_file in tqdm(
        json_files, total=len(json_files), desc=f"Processing results in {results_dir}"
    ):
        preproc_res_path = json_file.with_suffix(".csv")
        if not preproc_res_path.exists():
            preproc_res = pd.concat(
                [
                    preprocess_per_config_result(result, dataset)
                    for result in json.load(open(json_file))
                ]
            )
            print(f"Saving preprocessed results to {preproc_res_path} ...")
            preproc_res.to_csv(preproc_res_path, index=False)
        else:
            print(f"Loading preprocessed results from {preproc_res_path} ...")
            preproc_res = pd.read_csv(preproc_res_path)

        num_selectivities = len(dataset.all_selectivities)

        if index_key == "curator_opt":
            param_keys = ["nlist", "max_sl_size", "search_ef", "beam_size"]
        elif index_key == "shared_hnsw":
            param_keys = ["construction_ef", "m", "search_ef"]
        else:
            raise ValueError(f"Invalid index_key: {index_key}")

        params_df = list()
        for result in json.load(open(json_file)):
            params = {k: result[k] for k in param_keys}
            params_df.extend([params] * num_selectivities)
        params_df = pd.DataFrame(params_df)

        assert len(params_df) == len(preproc_res)
        preproc_res = pd.concat([preproc_res, params_df], axis=1)
        agg_results.append(preproc_res)

    print(f"Saving aggregated results to {output_path} ...")
    agg_results_df = pd.concat(agg_results)
    agg_results_df.to_csv(output_path, index=False)


def select_best_index_and_config_per_sel(
    output_path: str,
    curator_results_path: str,
    hnsw_results_path: str,
    recall_threshold: float = 0.9,
):
    curator_df = pd.read_csv(curator_results_path)
    hnsw_df = pd.read_csv(hnsw_results_path)
    combined_df = pd.concat(
        [
            curator_df.assign(index_type="curator"),
            hnsw_df.assign(index_type="hnsw"),
        ]
    )

    def filter_by_recall(group):
        filtered = group[group["recall"] >= recall_threshold]
        if not filtered.empty:
            return filtered
        return group.loc[group["recall"].idxmax()]

    filtered_df = (
        combined_df.groupby("selectivity")
        .apply(filter_by_recall)
        .reset_index(drop=True)
    )

    best_config_df = (
        filtered_df.groupby("selectivity")
        .apply(lambda group: group.loc[group["latency"].idxmin()])
        .reset_index(drop=True)
    )

    print(f"Saving best index and config per selectivity to {output_path} ...")
    best_config_df.to_csv(output_path, index=False)


def plot_revision_results(
    output_path: str = "output/selectivity/figs/revision_latency_vs_selectivity.pdf",
    min_recall: float = 0.9,
    max_recall: float = 1.0,
    max_latency_s: float = 0.1,
    annotation_x: float = 4e-3,  # Fixed x position for annotation
    annotation_y: float = 0.6,  # Fixed y position for annotation
):
    """Plot revision results with curator, pre-filtering, and acorn baselines using log scale dataset."""

    # Hardcoded result paths
    result_paths = {
        "Curator": "output/selectivity/curator_opt/yfcc100m_log/nlist16_maxsl64.csv",
        "Pre-Filtering": "output/selectivity/pre_filtering/yfcc100m_log/no_params.csv",
        "ACORN-10": "output/selectivity/acorn/yfcc100m_log/m32_g10_b64.csv",
        "ACORN-40": "output/selectivity/acorn/yfcc100m_log/m32_g40_b64.csv",
    }

    # Load and process results for each method
    best_results = {}
    for method_name, csv_path in result_paths.items():
        if Path(csv_path).exists():
            print(f"Loading results for {method_name} from {csv_path}")
            df = pd.read_csv(csv_path)

            # ACORN results store latency in milliseconds, convert to seconds for consistency
            if "ACORN" in method_name:
                df = df.copy()
                df["latency"] = df["latency"] / 1000.0
                print(f"Converted ACORN latency from ms to seconds")

            # Filter by recall and latency thresholds
            filtered_df = df[
                (df["recall"] >= min_recall)
                & (df["recall"] <= max_recall)
                & (df["latency"] <= max_latency_s)
            ]

            if not filtered_df.empty:
                # Select best config for each selectivity level (lowest latency)
                best_results_df = (
                    filtered_df.groupby("selectivity")
                    .apply(lambda group: group.loc[group["latency"].idxmin()])  # type: ignore
                    .reset_index(drop=True)
                )
                best_results[method_name] = best_results_df
                print(
                    f"Found {len(best_results_df)} selectivity points for {method_name}"
                )
            else:
                print(f"Warning: No results for {method_name} after filtering")
        else:
            print(f"Warning: Results file not found for {method_name}: {csv_path}")

    if not best_results:
        raise ValueError("No results found after loading and filtering")

    # Combine all best results
    combined_data = []
    for method_name, df in best_results.items():
        df = df.copy()
        df["index_type"] = method_name
        df["latency_ms"] = df["latency"] * 1000
        combined_data.append(df)

    df = pd.concat(combined_data, ignore_index=True)

    # Create plot
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(7, 4))

    # Define method order for consistent plotting
    method_order = ["Curator", "Pre-Filtering", "ACORN-10", "ACORN-40"]
    available_methods = [m for m in method_order if m in df["index_type"].unique()]

    colors = {
        "Curator": "tab:blue",
        "Pre-Filtering": "tab:red",
        "ACORN-10": "tab:green",
        "ACORN-40": "tab:purple",
    }

    sns.lineplot(
        data=df,
        x="selectivity",
        y="latency_ms",
        hue="index_type",
        style="index_type",
        markers=True,
        dashes=False,
        hue_order=available_methods,
        palette=[colors[method] for method in available_methods],
        errorbar=None,
        markersize=6,
        linewidth=1.5,
        ax=ax,
    )

    # Set log scale for both axes
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Filter Selectivity")
    ax.set_ylabel("Query Latency (ms)")

    # Enable grid for both x and y axes
    ax.grid(axis="both", which="major", linestyle="-", alpha=0.6)
    # ax.grid(axis="both", which="minor", linestyle=":", alpha=0.3)

    # Add shaded area only where curator outperforms baselines
    if "Curator" in available_methods and len(available_methods) > 1:
        # Get curator data
        curator_data = df[df["index_type"] == "Curator"].sort_values("selectivity")

        # Get baseline data (pre-filtering, acorn-10, acorn-40)
        baseline_methods = [
            m
            for m in ["Pre-Filtering", "ACORN-10", "ACORN-40"]
            if m in available_methods
        ]
        if baseline_methods:
            baseline_data = df[df["index_type"].isin(baseline_methods)]

            # For each selectivity level, find the minimum latency among baselines
            min_baseline_latency = (
                baseline_data.groupby("selectivity")["latency_ms"].min().reset_index()
            )

            # Merge with curator data on common selectivity levels
            common_selectivities = set(curator_data["selectivity"]) & set(
                min_baseline_latency["selectivity"]
            )
            if common_selectivities:
                curator_subset = curator_data[
                    curator_data["selectivity"].isin(common_selectivities)
                ].sort_values("selectivity")
                baseline_subset = min_baseline_latency[
                    min_baseline_latency["selectivity"].isin(common_selectivities)
                ].sort_values("selectivity")

                # Only shade where curator is better (lower latency) than baselines
                curator_better_mask = (
                    curator_subset["latency_ms"] < baseline_subset["latency_ms"]
                )

                if curator_better_mask.any():
                    # Get the subset where curator is better
                    better_selectivities = curator_subset[curator_better_mask][
                        "selectivity"
                    ]
                    better_curator_latency = curator_subset[curator_better_mask][
                        "latency_ms"
                    ]
                    better_baseline_latency = baseline_subset[curator_better_mask][
                        "latency_ms"
                    ]

                    # Fill area between curator and minimum baseline only where curator is better
                    ax.fill_between(
                        better_selectivities,
                        better_curator_latency,
                        better_baseline_latency,
                        hatch="////",  # Tilted lines pattern
                        facecolor="lightyellow",  # Light background color
                        alpha=0.6,
                        edgecolor="darkorange",  # Contrasting color for hatch lines
                        linewidth=1.0,
                        label="_nolegend_",
                    )

    # Add fixed text annotation for performance gap with white outline
    import matplotlib.patheffects as path_effects

    text = ax.annotate(
        "Perf Gap",
        xy=(annotation_x, annotation_y),
        ha="center",
        va="center",
        fontsize=plt.rcParams["xtick.labelsize"],
        fontfamily=plt.rcParams["font.family"],
        color="black",
    )
    # Add white stroke outline
    text.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    )

    # Adjust legend - single row
    legend = plt.legend(
        ax.get_legend().legend_handles,
        available_methods,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.2),
        ncols=len(available_methods),
        fontsize="small",
    )

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire()
