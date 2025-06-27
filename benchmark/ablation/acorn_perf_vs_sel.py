import json
import pickle as pkl
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset.utils import compute_ground_truth
from indexes.acorn import ACORN
from indexes.acorn import write_dataset as write_split

from benchmark.config import IndexConfig
from benchmark.overall_results.acorn import load_latencies_from_binary
from benchmark.profiler import Dataset, IndexProfiler


def synthesize_dataset(
    dataset_key: str,
    test_size: int,
    selectivities: list[float],
    n_label_per_selectivity: int,
    n_queries_per_label: int,
    seed: int,
):
    # Load original dataset
    print(f"Loading dataset {dataset_key} ...")
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

    # Sample vectors from both train and test set
    curr_label = 0
    train_mds = [[] for _ in range(dataset.train_vecs.shape[0])]
    test_mds = [[] for _ in range(dataset.test_vecs.shape[0])]

    np.random.seed(seed)

    for selectivity in selectivities:
        for _ in range(n_label_per_selectivity):
            n_qual_train = int(selectivity * dataset.train_vecs.shape[0])
            if n_qual_train == 0:
                raise ValueError(f"Selectivity {selectivity} is too low")

            qual_train_indices = np.random.choice(
                dataset.train_vecs.shape[0], n_qual_train, replace=False
            )
            qual_test_indices = np.random.choice(
                dataset.test_vecs.shape[0], n_queries_per_label, replace=False
            )

            for i in qual_train_indices:
                train_mds[i].append(curr_label)

            for i in qual_test_indices:
                test_mds[i].append(curr_label)

            curr_label += 1

    # Compute ground truth
    ground_truth, train_cates = compute_ground_truth(
        dataset.train_vecs, train_mds, dataset.test_vecs, test_mds
    )

    # Return dataset
    dataset.train_mds = train_mds
    dataset.test_mds = test_mds
    dataset.ground_truth = ground_truth
    dataset.all_labels = train_cates

    return dataset


def write_dataset(
    dataset_key: str,
    test_size: int,
    min_selectivity: float,
    max_selectivity: float,
    n_selectivities: int,
    n_label_per_selectivity: int,
    n_queries_per_label: int,
    dataset_dir: str | Path,
    log_scale_selectivity: bool = True,
    seed: int = 42,
    overwrite: bool = False,
):
    # Synthesize dataset
    print("Synthesizing dataset ...")
    if log_scale_selectivity:
        selectivities = np.logspace(
            np.log10(min_selectivity), np.log10(max_selectivity), n_selectivities
        )
    else:
        selectivities = np.linspace(min_selectivity, max_selectivity, n_selectivities)

    dataset = synthesize_dataset(
        dataset_key,
        test_size,
        selectivities,
        n_label_per_selectivity,
        n_queries_per_label,
        seed,
    )

    # Write dataset
    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    write_split(dataset.train_vecs, dataset.train_mds, dataset_dir, "train", overwrite)
    write_split(dataset.test_vecs, dataset.test_mds, dataset_dir, "test", overwrite)

    # Write metadata
    metadata = {
        "dataset_key": dataset_key,
        "test_size": test_size,
        "min_selectivity": min_selectivity,
        "max_selectivity": max_selectivity,
        "n_selectivities": n_selectivities,
        "n_label_per_selectivity": n_label_per_selectivity,
        "n_queries_per_label": n_queries_per_label,
        "log_scale_selectivity": log_scale_selectivity,
        "seed": seed,
    }
    json.dump(metadata, open(dataset_dir / "metadata.json", "w"))

    print(f"Dataset written to {dataset_dir} ...")


def load_dataset(dataset_dir: str | Path) -> Dataset:
    dataset_meta = json.load(open(Path(dataset_dir) / "metadata.json"))

    if dataset_meta["log_scale_selectivity"]:
        selectivities = np.logspace(
            np.log10(dataset_meta["min_selectivity"]),
            np.log10(dataset_meta["max_selectivity"]),
            dataset_meta["n_selectivities"],
        )
    else:
        selectivities = np.linspace(
            dataset_meta["min_selectivity"],
            dataset_meta["max_selectivity"],
            dataset_meta["n_selectivities"],
        )

    # Should generate the same dataset and load cached ground truth
    dataset = synthesize_dataset(
        dataset_meta["dataset_key"],
        dataset_meta["test_size"],
        selectivities,
        dataset_meta["n_label_per_selectivity"],
        dataset_meta["n_queries_per_label"],
        dataset_meta["seed"],
    )

    return dataset, dataset_meta


def construct_acorn_index(
    dataset_dir: str | Path,
    dataset_key: str,
    test_size: float,
    index_dir: str | Path,
    m: int,
    gamma: int,
    m_beta: int,
    output_path: str | Path | None = None,
):
    profiler = IndexProfiler()

    print(f"Loading dataset {dataset_key} ...")
    dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)
    profiler.set_dataset(dataset)

    print(f"Building index with m = {m}, gamma = {gamma}, m_beta = {m_beta} ...")
    index_config = IndexConfig(
        index_cls=ACORN,
        index_params={
            "dataset_dir": dataset_dir,
            "index_dir": index_dir,
            "m": m,
            "gamma": gamma,
            "m_beta": m_beta,
        },
        search_params={
            "search_ef": 16,
        },
    )

    build_results = profiler.do_build(
        index_config=index_config,
        do_train=False,
        batch_insert=True,  # acorn only supports batch insert
    )

    memory_usage = json.load(open(Path(index_dir) / "memory_usage.json"))
    build_results["index_size_kb"] = memory_usage["memory_usage_kb"]

    print(f"Index saved to {index_dir} ...")

    construct_params_path = Path(index_dir) / "construct_params.json"
    print(f"Writing construct params to {construct_params_path} ...")
    json.dump(
        {
            "dataset_dir": str(dataset_dir),
            "index_dir": str(index_dir),
            "m": m,
            "gamma": gamma,
            "m_beta": m_beta,
        },
        open(construct_params_path, "w"),
    )

    if output_path is None:
        output_path = Path(index_dir) / "build_results.json"

    print(f"Writing build results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(build_results, open(output_path, "w"))


def evaluate_acorn_index(
    index_dir: str | Path,
    search_ef_space: list[int],
    output_path: str | Path,
    num_runs: int = 1,
    return_verbose: bool = True,
):
    profiler = IndexProfiler()

    print(f"Loading index from {index_dir} ...")
    construct_params = json.load(open(Path(index_dir) / "construct_params.json"))
    print(f"Construct params: {construct_params}")

    dataset_dir = Path(construct_params["dataset_dir"])
    print(f"Loading dataset from {dataset_dir} ...")
    profiler.set_dataset(load_dataset(dataset_dir)[0])

    index_config = IndexConfig(
        index_cls=ACORN,
        index_params={
            "dataset_dir": str(dataset_dir),
            "index_dir": index_dir,
            "m": construct_params["m"],
            "gamma": construct_params["gamma"],
            "m_beta": construct_params["m_beta"],
        },
        search_params={
            "search_ef": search_ef_space[0],
        },
    )
    profiler.set_index(index_config.index_cls(**index_config.index_params))

    results = list()
    for search_ef in search_ef_space:
        print(f"Querying index with search_ef = {search_ef} ...")
        profiler.set_index_search_params({"search_ef": search_ef})
        query_results = profiler.do_query(
            batch_query=True,
            num_runs=num_runs,
            return_verbose=return_verbose,
        )
        if return_verbose:
            query_results["query_latencies"] = load_latencies_from_binary(
                Path(index_dir) / f"search_latency.bin"
            )

        results.append(
            {
                "m": construct_params["m"],
                "gamma": construct_params["gamma"],
                "m_beta": construct_params["m_beta"],
                "search_ef": search_ef,
                **query_results,
            }
        )

    print(f"Saving results to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump(results, open(output_path, "wb"))


def load_profile_results(
    dataset: Dataset,
    dataset_meta: dict,
    results_path: str | Path,
):
    if dataset_meta["log_scale_selectivity"]:
        selectivities = np.logspace(
            np.log10(dataset_meta["min_selectivity"]),
            np.log10(dataset_meta["max_selectivity"]),
            dataset_meta["n_selectivities"],
        )
    else:
        selectivities = np.linspace(
            dataset_meta["min_selectivity"],
            dataset_meta["max_selectivity"],
            dataset_meta["n_selectivities"],
        )

    label = 0
    label_to_sel = dict()
    for selectivity in selectivities:
        for _ in range(dataset_meta["n_label_per_selectivity"]):
            label_to_sel[label] = selectivity
            label += 1

    print(f"Loading results from {results_path} ...")
    results = pkl.load(open(results_path, "rb"))
    assert len(results) > 0 and "query_latencies" in results[0]
    n_queries = sum(len(label_list) for label_list in dataset.test_mds)
    # assert all(len(res["query_latencies"]) == n_queries for res in results)
    # TODO: len(res["query_latencies"]) is incorrect: 30000 vs 20000 (expected)

    print("Flattening results into a DataFrame ...")
    flattened_results = list()
    for res in results:
        label_gen = (label for label_list in dataset.test_mds for label in label_list)
        for label, latency, recall in zip(
            label_gen, res["query_latencies"], res["query_recalls"]
        ):
            flattened_results.append(
                {
                    "m": res["m"],
                    "gamma": res["gamma"],
                    "m_beta": res["m_beta"],
                    "search_ef": res["search_ef"],
                    "selectivity": label_to_sel[label],
                    "latency": latency,
                    "recall": recall,
                }
            )

    return pd.DataFrame(flattened_results)


def plot_latency_vs_selectivity(
    gamma_space: list[int],
    results_path_format: str,  # e.g. gamma{gamma}.pkl
    dataset_dir: str | Path,
    output_path: str | Path,
    min_recall: float = 0.90,
    max_selectivity: float = 0.4,
):
    print(f"Loading dataset from {dataset_dir} ...")
    dataset, dataset_meta = load_dataset(dataset_dir)

    results_dfs = list()
    for gamma in gamma_space:
        results_path = results_path_format.format(gamma=gamma)
        results_df = load_profile_results(dataset, dataset_meta, results_path)
        results_dfs.append(results_df)

    results_df = pd.concat(results_dfs)

    assert results_df["m"].nunique() == 1 and results_df["m_beta"].nunique() == 1
    m, m_beta = results_df["m"].iloc[0], results_df["m_beta"].iloc[0]
    results_df = results_df.drop(columns=["m", "m_beta"])

    results_df = (
        results_df.groupby(["gamma", "selectivity", "search_ef"])
        .agg({"latency": "mean", "recall": "mean"})
        .reset_index()
    )
    print(results_df.head())

    print(f"Saving results to {Path(output_path).with_suffix('.csv.1')} ...")
    results_df.sort_values(by=["gamma", "selectivity"]).to_csv(
        Path(output_path).with_suffix(".csv.1"), index=False
    )

    results_df = results_df[results_df["selectivity"] <= max_selectivity]
    results_df = results_df[results_df["recall"] >= min_recall]

    # TODO: assumes lower search_ef means lower latency
    results_df = (
        results_df.groupby(["gamma", "selectivity"])
        .agg({"search_ef": "min", "latency": "min", "recall": "min"})
        .reset_index()
    )
    print(results_df.head())

    print(f"Saving results to {Path(output_path).with_suffix('.csv')} ...")
    results_df.sort_values(by=["gamma", "selectivity"]).to_csv(
        Path(output_path).with_suffix(".csv"), index=False
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    plt.rcParams.update({"font.size": 12})
    sns.lineplot(
        data=results_df,
        x="selectivity",
        y="latency",
        hue="gamma",
        marker="o",
        markersize=5,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Selectivity")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"ACORN on YFCC1M (m={m}, m_beta={m_beta})", fontsize="small")
    ax.legend(title="gamma", fontsize="small")

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    fire.Fire()
