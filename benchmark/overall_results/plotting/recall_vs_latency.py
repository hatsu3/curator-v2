"""
Recall vs latency plotting for overall results across different selectivity levels.
"""

import json
import pickle as pkl
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmark.profiler import Dataset

# Dataset-specific configuration
# This is necessary because different datasets have different characteristics:
# - labels_per_group: Setting this too large (e.g., larger than the number of unique labels)
#   can cause all labels to be grouped together, leading to empty selectivity groups
# - default_test_size: Common test size used for each dataset
DATASET_LABELS_PER_GROUP = {
    "yfcc100m": {
        "labels_per_group": 100,  # 1000 unique labels
        "default_test_size": 0.001,
    },
    "yfcc100m-10m": {
        "labels_per_group": 100,  # 1000 unique labels
        "default_test_size": 0.001,
    },
    "arxiv-large-10": {
        "labels_per_group": 10,  # 100 unique labels
        "default_test_size": 0.005,
    },
}


def _load_prefilter_model(
    prefilter_model_dir: Union[str, Path], dataset_key: str, test_size: float
) -> Optional[Tuple[float, float]]:
    """Load (a, b) from linreg.json for Pre-Filtering.

    Tries `<prefilter_model_dir>/<dataset_key>_test<test_size>/linreg.json`.
    If not found, falls back to the first `linreg.json` under `prefilter_model_dir`.
    Returns (a, b) or None.
    """
    prefilter_model_dir = Path(prefilter_model_dir)
    exact_dir = prefilter_model_dir / f"{dataset_key}_test{test_size}"
    exact_path = exact_dir / "linreg.json"
    if exact_path.exists():
        data = json.load(open(exact_path))
        return float(data["a"]), float(data["b"])

    for p in prefilter_model_dir.rglob("linreg.json"):
        try:
            data = json.load(open(p))
            return float(data["a"]), float(data["b"])
        except Exception:
            continue
    return None


def _compute_prefilter_for_groups(
    dataset_key: str,
    test_size: float,
    percentiles: List[float],
    labels_per_group: Optional[int],
    a: float,
    b: float,
) -> pd.DataFrame:
    """Compute Pre-Filtering latency for each selectivity group using linreg.

    latency = a * N_avg + b, recall = 1.0, where N_avg is mean qualified count
    over labels in the group.
    """
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    label_groups = group_labels_by_selectivity(
        labels_per_group=labels_per_group,
        percentiles=percentiles,
        dataset_key=dataset_key,
        test_size=test_size,
        dataset=dataset,
    )

    n_train = len(dataset.train_vecs)
    label2sel = dataset.label_selectivities

    rows = []
    for g in label_groups:
        sels = [label2sel[lbl] for lbl in g["labels"] if lbl in label2sel]
        if not sels:
            continue
        sel_avg = float(np.mean(sels))
        n_avg = sel_avg * n_train
        lat_s = a * n_avg + max(b, 0.0)
        rows.append(
            {
                "percentile": g["percentile"],
                "selectivity": g["selectivity"],
                "recall": 1.0,
                "latency": lat_s,
                "index_key": "Pre-Filtering",
            }
        )

    return pd.DataFrame(rows)


def group_labels_by_selectivity(
    labels_per_group: int | None = None,
    percentiles: list[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    dataset_key: str = "yfcc100m-10m",
    test_size: float = 0.001,
    dataset: Dataset | None = None,
):
    """Group labels by selectivity

    Each group contains top-labels_per_group labels with selectivity closest to the percentile
    of the group.

    Args:
        labels_per_group (int): number of labels per group
        percentiles (list[float]): corresponding percentiles of selectivity of each label group
        dataset (Dataset | None): dataset object. if None, will be loaded by dataset_key and test_size

    Returns:
        list[dict]: list of dictionaries, each containing:
            percentile (float): percentile of selectivity of the group
            selectivity (float): selectivity of the group
            labels (list[int]): labels in the group
    """
    # Use dataset-specific default if not provided
    if labels_per_group is None:
        if dataset_key not in DATASET_LABELS_PER_GROUP:
            raise ValueError(
                f"Dataset '{dataset_key}' not found in DATASET_LABELS_PER_GROUP. Available datasets: {list(DATASET_LABELS_PER_GROUP.keys())}"
            )
        labels_per_group = DATASET_LABELS_PER_GROUP[dataset_key]["labels_per_group"]
        print(
            f"Using dataset-specific labels_per_group={labels_per_group} for {dataset_key}"
        )

    if dataset is None:
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    label2sel: dict[int, float] = dataset.label_selectivities
    sorted_sels = sorted(label2sel.values())

    label_groups = []

    for percentile in percentiles:
        percentile_index = int(np.ceil(percentile * len(sorted_sels)) - 1)
        selectivity = sorted_sels[percentile_index]
        sorted_labels = sorted(
            label2sel.keys(), key=lambda label: abs(label2sel[label] - selectivity)
        )
        label_groups.append(
            {
                "percentile": percentile,
                "selectivity": selectivity,
                "labels": sorted_labels[:labels_per_group],
            }
        )

    return label_groups


def preprocess_per_config_result(
    per_config_results: dict, dataset: Dataset, label_groups: list[dict]
):
    """Aggregate profiling results of a specific search configuration

    For each label group, compute the average recall and latency across queries of
    all labels in the group.

    Args:
        per_config_results (dict): profiling results of a specific search configuration
        dataset (Dataset): dataset object
        label_groups (list[dict]): label groups by selectivity.
            Each entry is a dictionary with keys: percentile (0-1), selectivity (0-1), labels (list[int])
    Returns:
        pd.DataFrame: aggregated dataframe with 4 columns:
            percentile, selectivity, recall, latency
    """

    recall_gen = iter(per_config_results["query_recalls"])
    latency_gen = iter(per_config_results["query_latencies"])

    per_sel_results = [
        {
            "percentile": label_group["percentile"],
            "selectivity": label_group["selectivity"],
            "recalls": list(),
            "latencies": list(),
        }
        for label_group in label_groups
    ]

    label_to_group = {
        label: idx
        for idx, label_group in enumerate(label_groups)
        for label in label_group["labels"]
    }

    for __, label_list in zip(dataset.test_vecs, dataset.test_mds):
        for label in label_list:
            recall, latency = next(recall_gen), next(latency_gen)

            # some labels may not be in any label group
            if label not in label_to_group:
                continue

            group_idx = label_to_group[label]
            per_sel_results[group_idx]["recalls"].append(recall)
            per_sel_results[group_idx]["latencies"].append(latency)

    per_sel_results_df = pd.DataFrame(
        [
            {
                "percentile": res["percentile"],
                "selectivity": res["selectivity"],
                "recall": np.mean(res["recalls"]).item(),
                "latency": np.mean(res["latencies"]).item(),
            }
            for res in per_sel_results
        ]
    )

    return per_sel_results_df


def aggregate_per_selectivity_results(
    results_path: Path | str,
    labels_per_group: int | None = None,
    percentiles: list[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    dataset_key: str = "yfcc100m-10m",
    test_size: float = 0.001,
    output_path: Path | str | None = None,
    dataset: Dataset | None = None,
    label_groups: list[dict] | None = None,
    pg_ivf_mode: str | None = None,  # "iter" or "classic"
):
    """Aggregate profiling results of a specific index across all search configurations

    Group labels by selectivity and aggregate profiling results of a specific index across
    all search configurations: each csv file contains profiling results of multiple search
    configurations.

    Args:
        results_path (Path | str): path to the results csv file, which should contain two
            list-valued columns: query_latencies, query_recalls
        output_path (Path | str | None): path to the output csv file. if None, will be set to
            results_path.with_name(f"{results_path.stem}_preproc.csv")
        labels_per_group (int): number of labels per group
        percentiles (list[float]): corresponding percentiles of selectivity of each label group
        dataset (Dataset | None): dataset object. if None, will be loaded by dataset_key and test_size
        label_groups (list[dict] | None): precomputed label groups. if None, will be computed from dataset
    Returns:
        pd.DataFrame: aggregated dataframe with 4 columns:
            percentile, selectivity, recall, latency
            rows with the same (percentile, selectivity) values correspond to different configurations
    """
    results_path = Path(results_path)

    # Use dataset-specific default if not provided
    if labels_per_group is None:
        if dataset_key not in DATASET_LABELS_PER_GROUP:
            raise ValueError(
                f"Dataset '{dataset_key}' not found in DATASET_LABELS_PER_GROUP. Available datasets: {list(DATASET_LABELS_PER_GROUP.keys())}"
            )
        labels_per_group = DATASET_LABELS_PER_GROUP[dataset_key]["labels_per_group"]
        print(
            f"Using dataset-specific labels_per_group={labels_per_group} for {dataset_key}"
        )

    if output_path is None:
        output_path = results_path.with_name(f"{results_path.stem}_preproc.csv")
    else:
        output_path = Path(output_path)

    # Load dataset only if not provided
    if dataset is None:
        print(f"Loading dataset {dataset_key} ...")
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    # Compute label groups only if not provided
    if label_groups is None:
        label_groups = group_labels_by_selectivity(
            labels_per_group, percentiles, dataset_key, test_size, dataset=dataset
        )

    print(f"Loading results from {results_path} ...")
    # convert results to DataFrame with two list-valued columns: query_latencies, query_recalls
    if results_path.suffix == ".pkl":
        results = pkl.load(results_path.open("rb"))
    elif results_path.suffix == ".csv":
        results_df = pd.read_csv(results_path)
        for col in ["query_latencies", "query_recalls"]:
            results_df[col] = results_df[col].apply(json.loads)
        results = results_df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file extension {results_path.suffix}")

    # Filter pgvector IVF results by mode (iter or classic), based on
    # per-config metadata iter_search. 
    def _keep(res: dict) -> bool:
        is_pg = "strategy" in res  # hacky
        is_pg_ivf = is_pg and str(res.get("strategy")) == "ivf"
        
        # only apply checks if we are reading pgvector-ivf results and pg_ivf_mode is set
        if is_pg_ivf and pg_ivf_mode is not None:
            return res["iter_search"] == (pg_ivf_mode == "iter")
        else:
            return True

    results = [res for res in results if _keep(res)]

    print("Preprocessing results ...")
    preproc_res = pd.concat(
        [preprocess_per_config_result(res, dataset, label_groups) for res in results]
    )

    print(f"Saving preprocessed results to {output_path} ...")
    preproc_res.to_csv(output_path, index=False)
    return preproc_res


def discover_baseline_results(
    results_dir: Union[Path, str],
    dataset_key: str = "yfcc100m-10m",
    test_size: float = 0.001,
) -> Dict[str, Optional[str]]:
    """Discover available baseline results from the output directory structure of run_overall_results.sh

    Args:
        results_dir: Root directory containing baseline results (e.g., output/overall_results2)
        dataset_key: Dataset key (e.g., "yfcc100m-10m", "arxiv-large-10")
        test_size: Test size used (e.g., 0.001, 0.005)

    Returns:
        Dictionary mapping baseline names to result file paths
    """
    results_dir = Path(results_dir)
    dataset_subdir = f"{dataset_key}_test{test_size}"

    # Map algorithm directory names to display names
    algorithm_mapping = {
        "curator": "Curator",
        "per_label_hnsw": "Per-Label HNSW",
        "per_label_ivf": "Per-Label IVF",
        "shared_hnsw": "Shared HNSW",
        "shared_ivf": "Shared IVF",
        "parlay_ivf": "Parlay IVF",
        "filtered_diskann": "Filtered DiskANN",
        "acorn_1": "ACORN-1",
        "acorn_gamma": r"ACORN-$\gamma$",
        "pgvector_hnsw": "Pg-HNSW",
        "pgvector_ivf": "Pg-IVF",
    }

    baseline_results = {}

    for algo_dir, display_name in algorithm_mapping.items():
        algo_path = results_dir / algo_dir / dataset_subdir
        results_file = algo_path / "results.csv"

        # Always use raw results file for processing
        if results_file.exists():
            baseline_results[display_name] = str(results_file)
        else:
            baseline_results[display_name] = None

    return baseline_results


def preprocess_all_baselines(
    results_dir: Union[Path, str],
    dataset_key: str,
    test_size: float,
    labels_per_group: int | None = None,
    percentiles: List[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    force_reprocess: bool = False,
    pg_ivf_mode: str | None = None,
) -> Dict[str, str]:
    """Preprocess all available baseline results for plotting

    Args:
        results_dir: Root directory containing baseline results
        dataset_key: Dataset key (e.g., "yfcc100m-10m", "arxiv-large-10")
        test_size: Test size used (e.g., 0.001, 0.005)
        labels_per_group: Number of labels per selectivity group
        percentiles: Selectivity percentiles to analyze
        force_reprocess: Whether to force reprocessing even if preprocessed files exist

    Returns:
        Dictionary mapping baseline names to preprocessed result file paths
    """
    # Use dataset-specific default if not provided
    if labels_per_group is None:
        if dataset_key not in DATASET_LABELS_PER_GROUP:
            raise ValueError(
                f"Dataset '{dataset_key}' not found in DATASET_LABELS_PER_GROUP. Available datasets: {list(DATASET_LABELS_PER_GROUP.keys())}"
            )
        labels_per_group = DATASET_LABELS_PER_GROUP[dataset_key]["labels_per_group"]
        print(
            f"Using dataset-specific labels_per_group={labels_per_group} for {dataset_key}"
        )

    baseline_results = discover_baseline_results(results_dir, dataset_key, test_size)
    preprocessed_results = {}

    # Check if any preprocessing is needed
    needs_preprocessing = []
    for baseline_name, result_path in baseline_results.items():
        if result_path is None:
            print(f"Skipping {baseline_name}: no results found")
            continue

        result_path = Path(result_path)
        preproc_path = result_path.with_name(f"{result_path.stem}_preproc.csv")

        # Skip if preprocessed file exists and we're not forcing reprocessing
        if preproc_path.exists() and not force_reprocess:
            print(
                f"Using existing preprocessed results for {baseline_name}: {preproc_path}"
            )
            preprocessed_results[baseline_name] = str(preproc_path)
        else:
            needs_preprocessing.append((baseline_name, result_path, preproc_path))

    # If any baselines need preprocessing, load dataset once
    if needs_preprocessing:
        print(f"Loading dataset {dataset_key} with test_size={test_size} ...")
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

        print("Computing label groups by selectivity ...")
        label_groups = group_labels_by_selectivity(
            labels_per_group, percentiles, dataset_key, test_size, dataset=dataset
        )

        # Process each baseline that needs preprocessing
        for baseline_name, result_path, preproc_path in needs_preprocessing:
            print(f"Preprocessing {baseline_name} from {result_path}...")
            try:
                aggregate_per_selectivity_results(
                    results_path=result_path,
                    labels_per_group=labels_per_group,
                    percentiles=percentiles,
                    dataset_key=dataset_key,
                    test_size=test_size,
                    output_path=preproc_path,
                    dataset=dataset,
                    label_groups=label_groups,
                    pg_ivf_mode=(pg_ivf_mode if baseline_name == "Pg-IVF" else None),
                )
                preprocessed_results[baseline_name] = str(preproc_path)
                print(f"Preprocessed {baseline_name} -> {preproc_path}")
            except Exception as e:
                print(f"Error preprocessing {baseline_name}: {e}")

    return preprocessed_results


def plot_recall_vs_latency(
    results_dir: Union[Path, str],
    dataset_name: str,
    output_path: Optional[str] = None,
    percentiles: List[float] = [0.01, 0.25, 0.50, 0.75, 1.00],
    labels_per_group: int | None = None,
    force_reprocess: bool = False,
    figsize_per_subplot: tuple = (3, 3),
    font_size: int = 14,
    prefilter_model_dir: Union[str, Path] = "output/overall_results2/pre_filtering",
    y_metric: str = "qps",
    pg_ivf_mode: str = "iter",
):
    """
    Plot recall vs latency across different selectivity levels using results from run_overall_results.sh

    Args:
        results_dir: Root directory containing baseline results (e.g., output/overall_results2)
        dataset_name: Dataset name ("yfcc100m" or "arxiv")
        output_path: Path to save the plot. If None, auto-generated based on dataset and results_dir
        percentiles: Selectivity percentiles to plot
        labels_per_group: Number of labels per selectivity group
        force_reprocess: Whether to force reprocessing of raw results
        figsize_per_subplot: Size of each subplot (width, height)
        font_size: Font size for the plot
        y_metric: Y axis metric: "qps" (default) or "latency" (milliseconds)
        pg_ivf_mode: Which Pg-IVF mode to plot: "iter" (default) or "classic". If the
            selected mode is unavailable in results, falls back to any available mode.
    """

    # Map dataset names to dataset keys and test sizes
    dataset_config = {
        "yfcc100m": {"dataset_key": "yfcc100m-10m", "test_size": 0.001},
        "arxiv": {"dataset_key": "arxiv-large-10", "test_size": 0.005},
    }

    if dataset_name not in dataset_config:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Must be one of {list(dataset_config.keys())}"
        )

    dataset_key = dataset_config[dataset_name]["dataset_key"]
    test_size = dataset_config[dataset_name]["test_size"]

    # Set default output path if not provided
    if output_path is None:
        results_dir_name = Path(results_dir).name
        output_path = f"output/overall_results/figs/recall_vs_latency_{dataset_name}_{results_dir_name}.pdf"

    print(f"=== Plotting Recall vs Latency for {dataset_name} ===")
    print(f"Results directory: {results_dir}")
    print(f"Dataset key: {dataset_key}")
    print(f"Test size: {test_size}")
    print(f"Output path: {output_path}")
    print()

    # Preprocess all available baseline results
    preprocessed_results = preprocess_all_baselines(
        results_dir=results_dir,
        dataset_key=dataset_key,
        test_size=test_size,
        labels_per_group=labels_per_group,
        percentiles=percentiles,
        force_reprocess=force_reprocess,
        pg_ivf_mode=pg_ivf_mode,
    )

    if not preprocessed_results:
        raise ValueError(f"No baseline results found in {results_dir}")

    print(f"\nFound {len(preprocessed_results)} baselines to plot:")
    for baseline_name in preprocessed_results.keys():
        print(f"  - {baseline_name}")
    print()

    # Load all preprocessed results
    all_results = {}
    for baseline_name, preproc_path in preprocessed_results.items():
        try:
            df = pd.read_csv(preproc_path)
            df["index_key"] = baseline_name
            all_results[baseline_name] = df
            print(f"Loaded {len(df)} rows from {baseline_name}")
        except Exception as e:
            print(f"Error loading {baseline_name}: {e}")

    # Aggregate results by percentile
    per_pct_results = {}
    for percentile in percentiles:
        res_list = []
        for baseline_name, df in all_results.items():
            pct_data = df[df["percentile"] == percentile]
            if not pct_data.empty:
                res_list.append(pct_data)

        if res_list:
            res = pd.concat(res_list, ignore_index=True)
            res["latency"] *= 1000.0  # convert to milliseconds
            per_pct_results[percentile] = res

    if not per_pct_results:
        raise ValueError("No data found for any percentile")

    # Inject Pre-Filtering baseline estimated by linear regression model
    pf_params = _load_prefilter_model(prefilter_model_dir, dataset_key, test_size)
    if pf_params is not None:
        a, b = pf_params
        pf_df = _compute_prefilter_for_groups(
            dataset_key=dataset_key,
            test_size=test_size,
            percentiles=percentiles,
            labels_per_group=labels_per_group,
            a=a,
            b=b,
        )
        if not pf_df.empty:
            pf_df = pf_df.copy()
            pf_df["latency"] *= 1000.0  # convert to ms to match others
            for percentile in percentiles:
                if percentile in per_pct_results:
                    per_pct_results[percentile] = pd.concat(
                        [
                            per_pct_results[percentile],
                            pf_df[pf_df["percentile"] == percentile],
                        ],
                        ignore_index=True,
                    )
            print("Added Pre-Filtering estimates from linear model")
    else:
        print("Warning: Pre-Filtering model not found; skipping Pre-Filtering baseline")

    # Get selectivity values for titles
    pct_to_sel = {}
    for percentile, res in per_pct_results.items():
        if not res.empty:
            pct_to_sel[percentile] = res["selectivity"].iloc[0]

    # Set up the plot
    plt.rcParams.update({"font.size": font_size})
    fig_width = figsize_per_subplot[0] * len(percentiles)
    fig_height = figsize_per_subplot[1]
    fig, axes = plt.subplots(1, len(percentiles), figsize=(fig_width, fig_height))

    # Handle case where there's only one subplot
    if len(percentiles) == 1:
        axes = [axes]

    # Define the desired order of baselines for plotting (matches legend order)
    desired_order = [
        "Per-Label HNSW",
        "Per-Label IVF",
        "Parlay IVF",
        "Filtered DiskANN",
        "Shared HNSW",
        "Shared IVF",
        "Pg-HNSW",
        "Pg-IVF",
        "ACORN-1",
        r"ACORN-$\gamma$",
        "Curator",
    ]

    # Use only baselines that are available, in the desired order
    baseline_names = [name for name in desired_order if name in all_results]

    # Add any additional baselines that weren't in the desired order
    additional_baselines = []
    for name in all_results.keys():
        if name not in baseline_names:
            additional_baselines.append(name)
            baseline_names.append(name)

    # Ensure Pre-Filtering is included if present
    present_prefilter = any(
        ("index_key" in df.columns) and (df["index_key"] == "Pre-Filtering").any()
        for df in per_pct_results.values()
    )
    if present_prefilter:
        # Remove any existing occurrence first
        baseline_names = [n for n in baseline_names if n != "Pre-Filtering"]
        # Insert Pre-Filtering right before Curator if present; otherwise, append
        if "Curator" in baseline_names:
            idx = baseline_names.index("Curator")
            baseline_names.insert(idx, "Pre-Filtering")
        else:
            baseline_names.append("Pre-Filtering")

    # Warn about additional baselines
    if additional_baselines:
        print(
            f"WARNING: Found {len(additional_baselines)} additional baseline(s) not in the predefined order:"
        )
        for name in additional_baselines:
            print(f"  - {name}")
        print(
            "These will be added to the end of the legend. Consider updating the desired_order list."
        )
        print()

    # Define colors and markers for different baselines
    colors = {
        "Per-Label HNSW": "tab:blue",
        "Per-Label IVF": "tab:orange",
        "Parlay IVF": "tab:green",
        "Filtered DiskANN": "tab:red",
        "Shared HNSW": "tab:purple",
        "Shared IVF": "tab:brown",
        "Pg-HNSW": "tab:pink",
        "Pg-IVF": "tab:brown",
        "ACORN-1": "tab:pink",
        r"ACORN-$\gamma$": "tab:gray",
        "Curator": "tab:olive",
        "Pre-Filtering": "tab:red",
    }

    markers = {
        "Per-Label HNSW": "o",
        "Per-Label IVF": "D",
        "Parlay IVF": "d",
        "Filtered DiskANN": "P",
        "Shared HNSW": "h",
        "Shared IVF": "s",
        "Pg-HNSW": "<",
        "Pg-IVF": ">",
        "ACORN-1": "*",
        r"ACORN-$\gamma$": "p",
        "Curator": "^",
        "Pre-Filtering": "X",
    }

    # Validate y_metric selection
    if y_metric not in {"qps", "latency"}:
        raise ValueError("y_metric must be one of {'qps', 'latency'}")

    # Plot each percentile
    for i, (ax, percentile) in enumerate(zip(axes, percentiles)):
        if percentile not in per_pct_results:
            ax.set_title(f"No data for {int(percentile * 100)}p")
            continue

        per_pct_df = per_pct_results[percentile]
        per_pct_df["latency_ms"] = per_pct_df["latency"]
        per_pct_df["qps"] = 1000.0 / per_pct_df["latency_ms"].clip(lower=1e-12)

        y_col = "qps" if y_metric == "qps" else "latency_ms"

        # Create line plot
        sns.lineplot(
            data=per_pct_df,
            x="recall",
            y=y_col,
            hue="index_key",
            hue_order=baseline_names,
            style="index_key",
            style_order=baseline_names,
            ax=ax,
            markers={k: v for k, v in markers.items() if k in baseline_names},
            palette={k: v for k, v in colors.items() if k in baseline_names},
            dashes=False,
        )

        ax.set_yscale("log")
        ax.set_xlabel("Recall@10" if i == len(axes) // 2 else "")
        ax.set_ylabel(
            ("QPS" if y_metric == "qps" else "Latency (ms)") if i == 0 else ""
        )

        # Set title with selectivity info
        if percentile in pct_to_sel:
            selectivity = pct_to_sel[percentile]
            display_percentile = 99 if percentile == 1.0 else int(percentile * 100)
            ax.set_title(f"{display_percentile}p Sel ({selectivity:.4f})")
        else:
            ax.set_title(f"{int(percentile * 100)}p")

        ax.grid(axis="x", which="major", linestyle="-", alpha=0.6)
        ax.grid(axis="y", which="major", linestyle="-", alpha=0.6)

    # Set consistent axis limits across all subplots
    if len(axes) > 1:
        global_xlim = (
            min(ax.get_xlim()[0] for ax in axes if ax.get_xlim()[0] > 0),
            max(ax.get_xlim()[1] for ax in axes),
        )
        global_ylim = (
            min(ax.get_ylim()[0] for ax in axes),
            max(ax.get_ylim()[1] for ax in axes),
        )

        for ax in axes:
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)

    # Create legend
    if len(baseline_names) <= 8:
        legend = fig.legend(
            axes[0].get_legend().legend_handles if axes[0].get_legend() else [],
            baseline_names,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(baseline_names),
            columnspacing=1.0,
        )
    else:
        legend = fig.legend(
            axes[0].get_legend().legend_handles if axes[0].get_legend() else [],
            baseline_names,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.2),
            ncol=(len(baseline_names) + 1) // 2,
            columnspacing=1.0,
        )

    # Remove individual legends
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()

    fig.tight_layout()

    # Save the figure
    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")
    print(f"Plot saved successfully!")


if __name__ == "__main__":
    fire.Fire()
