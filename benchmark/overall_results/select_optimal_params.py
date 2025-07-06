import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_all_results(
    output_dir: str = "output/overall_results/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
):
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    all_results = list()
    for csv_file in results_dir.glob("*.csv"):
        for res in pd.read_csv(csv_file).to_dict(orient="records"):
            all_results.append(
                {
                    "filename": csv_file.name,
                    **res,
                }
            )

    all_results_df = pd.DataFrame(all_results)
    all_results_df = all_results_df.reset_index(drop=True)
    return all_results_df


def plot_pareto_optimal_configs(
    output_dir: str = "output/overall_results",
    plot_dir: str = "output/overall_results/best_configs/yfcc100m",
    best_config_path: str = "output/overall_results/best_configs_yfcc100m.json",
    index_types: list[str] = [
        "per_label_hnsw",
        "per_label_ivf",
        "parlay_ivf_seq",
        "filtered_diskann",
        "shared_hnsw",
        "shared_ivf",
        "curator_opt",
        "acorn",
    ],
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
):
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    best_configs = json.load(open(best_config_path))

    dataset_to_num_queries = {
        ("yfcc100m", 0.01): 134868,
        ("arxiv-large-10", 0.005): 99492,
    }

    for index_type in index_types:
        best_config = best_configs[index_type]

        for config_idx, config in enumerate(best_config):
            results = load_all_results(
                output_dir=str(Path(output_dir) / index_type),
                dataset_key=dataset_key,
                test_size=test_size,
            )
            if "query_lat_avg" not in results.columns:
                assert "batch_query_latency" in results.columns
                results["query_lat_avg"] = (
                    results["batch_query_latency"]
                    / dataset_to_num_queries[(dataset_key, test_size)]
                )

            results["query_lat_avg"] = results["query_lat_avg"] * 1000

            config_results = results.query(
                " & ".join([f"{k} == {v}" for k, v in config.items() if k != "_count"])
            )

            ax = sns.scatterplot(
                data=results,
                x="query_lat_avg",
                y="recall_at_k",
                c="lightgrey",
            )

            sns.scatterplot(
                data=config_results,
                x="query_lat_avg",
                y="recall_at_k",
                s=100,
                c="red",
                ax=ax,
            )

            ax.set_xscale("log")
            ax.set_xlabel("Query Latency (ms)")
            ax.set_ylabel("Recall@10")
            ax.set_title(f"{index_type} - Count {config['_count']:.2f}")

            output_path = Path(plot_dir) / f"{index_type}_{config_idx}.pdf"
            print(f"Saving figure to {output_path} ...")
            plt.savefig(output_path)

            plt.clf()


if __name__ == "__main__":
    fire.Fire(plot_pareto_optimal_configs)
