import json
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def result_metric(
    recall: float, latency_ms: float, alpha: float = 1, min_recall: float | None = None
) -> float:
    score = recall - alpha * latency_ms
    if min_recall is not None and recall < min_recall:
        score -= 1e5

    return score


def select_best_result(
    output_dir: str = "output/complex_predicate/curator",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    alpha: float = 1,
    min_recall: float = 0.8,
) -> list[dict]:
    results_dir = Path(output_dir) / f"{dataset_key}_test{test_size}" / "results"
    if not results_dir.exists():
        raise ValueError(f"Results directory {results_dir} does not exist")

    agg_results = {template: list() for template in templates}
    for json_file in results_dir.glob("*.json"):
        for result in json.load(open(json_file)):
            for template, per_template_result in result["per_template_results"].items():
                agg_results[template].append(per_template_result)

    best_results = list()
    for template, results in agg_results.items():
        if len(results) == 0:  # experiment skipped
            continue

        best_result = max(
            results,
            key=lambda result: result_metric(
                result["recall_at_k"],
                result["query_lat_avg"] * 1000,
                alpha=alpha,
                min_recall=min_recall,
            ),
        )
        best_results.append(
            {
                "template": template,
                "recall": best_result["recall_at_k"],
                "query_latency_ms": best_result["query_lat_avg"] * 1000,
            }
        )

    return best_results


def plot_overall_results(
    index_keys: list[str] = [
        "shared_ivf"
        "shared_hnsw",
        "curator",
        "curator_with_index",
        "per_predicate_hnsw",
        "parlay_ivf",
    ],
    index_keys_readable: list[str] = [
        "Shared IVF"
        "Shared HNSW",
        "Curator",
        "Per-Pred Curator",
        "Per-Pred HNSW",
        "Parlay IVF",
    ],
    output_dir: str = "output/complex_predicate",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "OR {0} {1}", "AND {0} {1}"],
    alpha: float = 1,
    min_recall: float = 0.7,
    output_path: str = "output/complex_predicate/figs/overall_results.pdf",
):
    best_results: dict[str, list[dict]] = dict()
    for index_key in index_keys:
        best_results[index_key] = select_best_result(
            output_dir=str(Path(output_dir) / index_key),
            dataset_key=dataset_key,
            test_size=test_size,
            templates=templates,
            alpha=alpha,
            min_recall=min_recall,
        )

    df = pd.concat(
        [
            pd.DataFrame(results).assign(index_type=index_type)
            for index_type, results in best_results.items()
        ]
    )
    df["template"] = df["template"].str.split(" ").str[0]
    templates = [template.split(" ")[0] for template in templates]

    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    sns.barplot(
        data=df,
        x="template",
        y="recall",
        hue="index_type",
        ax=axes[0],
        order=templates,
        hue_order=index_keys,
    )
    axes[0].set_xlabel("Query Type")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_ylim(0.7, 1.0)
    axes[0].get_legend().remove()
    axes[0].grid(axis="y", which="major", linestyle="-", alpha=0.6)

    for p in axes[0].patches:
        height = p.get_height()
        if pd.isna(height):
            axes[0].text(
                p.get_x() + p.get_width() / 2,
                0.71,
                r"$\times$",
                ha="center",
                va="center",
                color="red",
                fontsize=14,
                fontweight="bold",
            )

    sns.barplot(
        data=df,
        x="template",
        y="query_latency_ms",
        hue="index_type",
        ax=axes[1],
        order=templates,
        hue_order=index_keys,
    )
    axes[1].set_xlabel("Query Type")
    axes[1].set_ylabel("Query Latency (ms)")
    axes[1].set_yscale("log")
    axes[1].get_legend().remove()
    axes[1].grid(axis="y", which="major", linestyle="-", alpha=0.6)

    for p in axes[1].patches:
        height = p.get_height()
        if pd.isna(height):
            axes[1].text(
                p.get_x() + p.get_width() / 2,
                7e-3,
                r"$\times$",
                ha="center",
                va="center",
                color="red",
                fontsize=14,
                fontweight="bold",
            )

    legend = fig.legend(
        axes[1].patches[:: len(templates)],
        index_keys_readable,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncols=(len(index_keys) + 1) // 2,
    )

    fig.tight_layout()

    print(f"Saving plot to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_extra_artists=(legend,), bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire()
