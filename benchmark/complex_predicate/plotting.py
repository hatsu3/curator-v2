import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_algo_results(
    base_output_dir: str = "output/complex_predicate",
    algorithm_name: str = "curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
):
    """Load results from algorithm output directory."""
    algo_dir = Path(base_output_dir) / algorithm_name
    results_dir = algo_dir / f"{dataset_key}_test{test_size}" / "results"

    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    all_results = list()
    for json_file in results_dir.glob("*.json"):
        for result in json.load(open(json_file)):
            for template, per_template_result in result["per_template_results"].items():
                all_results.append({"template": template, **per_template_result})

    all_results_df = pd.DataFrame(all_results)
    all_results_df["template"] = all_results_df["template"].str.split(" ").str[0]
    return all_results_df


def select_pareto_front(
    df: pd.DataFrame,
    x_key: str = "query_lat_avg",
    y_key: str = "recall_at_k",
    min_x: bool = True,  # minimize x
    min_y: bool = False,  # maximize y
):
    def is_dominated(r1, r2):
        x_worse = r1[x_key] > r2[x_key] if min_x else r1[x_key] < r2[x_key]
        y_worse = r1[y_key] > r2[y_key] if min_y else r1[y_key] < r2[y_key]
        return x_worse and y_worse

    pareto_front = []
    for i, r1 in df.iterrows():
        if any(is_dominated(r1, r2) for _, r2 in df.iterrows()):
            continue
        pareto_front.append(r1)

    return pd.DataFrame(pareto_front)


def plot_overall_results(
    index_keys: list[str] = [
        "curator",
        "curator_with_index",
        "shared_hnsw",
        "per_predicate_hnsw",
        "shared_ivf",
        "per_predicate_ivf",
        "parlay_ivf",
        "acorn",
    ],
    index_keys_readable: list[str] = [
        "Curator",
        "Curator (Indexed)",
        "Shared HNSW",
        "Per-Pred HNSW",
        "Shared IVF",
        "Per-Pred IVF",
        "Parlay IVF",
        "ACORN",
    ],
    output_dir: str = "output/complex_predicate",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["OR", "AND"],
    output_path: str = "output/complex_predicate/figs/overall_results.pdf",
):
    """Plot overall results comparing all algorithms."""
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, len(templates), figsize=(2.6 * len(templates), 3))

    # Handle single template case
    if len(templates) == 1:
        axes = [axes]

    # Custom color palette with special meanings:
    # - Blues for Curator variants (our main algorithms)
    # - Greens for HNSW baselines
    # - Oranges for IVF baselines
    # - Red for ACORN (external baseline)
    colors = sns.color_palette("tab10", n_colors=5)
    custom_palette = {
        "curator": colors[0],
        "curator_with_index": colors[0],
        "shared_hnsw": colors[1],
        "per_predicate_hnsw": colors[1],
        "shared_ivf": colors[2],
        "per_predicate_ivf": colors[2],
        "parlay_ivf": colors[3],
        "acorn": colors[4],
    }

    for i, (template, ax) in enumerate(zip(templates, axes)):
        index_results = dict()

        for index_key in index_keys:
            try:
                all_results_df = load_algo_results(
                    base_output_dir=output_dir,
                    algorithm_name=index_key,
                    dataset_key=dataset_key,
                    test_size=test_size,
                )

                template_results_df = all_results_df.query(f"template == '{template}'")
                if len(template_results_df) == 0:
                    raise ValueError(
                        f"No results found for algorithm '{index_key}' with template '{template}'"
                    )

                best_results_df = select_pareto_front(
                    template_results_df, x_key="query_lat_avg", y_key="recall_at_k"
                )
                index_results[index_key] = best_results_df
            except Exception as e:
                raise ValueError(
                    f"Failed to load results for algorithm '{index_key}': {e}"
                )

        agg_df = pd.concat(
            [
                results.assign(index_type=index_type)
                for index_type, results in index_results.items()
            ]
        )
        agg_df["query_lat_avg"] *= 1000  # convert to ms

        sns.lineplot(
            data=agg_df,
            x="query_lat_avg",
            y="recall_at_k",
            hue="index_type",
            hue_order=index_keys,
            style="index_type",
            style_order=index_keys,
            ax=ax,
            markers=True,
            dashes=False,
            palette=custom_palette,
        )

        if i == 0:
            ax.set_ylabel("Recall@10")
        else:
            ax.set_ylabel("")

        ax.set_xlabel("Query Latency (ms)")
        ax.set_xscale("log")
        ax.set_title(template)
        ax.set_ylim(0.15, 1.05)
        ax.minorticks_off()
        ax.grid(visible=True, which="major", axis="both", linestyle="-", alpha=0.6)

    legend = fig.legend(
        axes[0].get_legend().legend_handles,
        index_keys_readable,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncols=(len(index_keys_readable) + 1) // 2,
    )

    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire()
