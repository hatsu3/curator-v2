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


def plot_ablation_results(
    output_dir: str | Path = "output/ablation",
    metric: str = "beam_size",
    metric_readable: str = "Beam Size",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_path: str = "output/ablation/figs/ablation_beam_size.pdf",
):
    output_dir = Path(output_dir) / metric
    df = load_all_results(
        output_dir=str(output_dir),
        dataset_key=dataset_key,
        test_size=test_size,
    )
    df["query_lat_avg"] = df["query_lat_avg"] * 1000

    plt.rcParams.update({"font.size": 14})
    fig = plt.figure(figsize=(4, 4))
    ax = sns.lineplot(
        data=df,
        x="query_lat_avg",
        y="recall_at_k",
        hue=metric,
        marker="o",
        markersize=8,
        linewidth=2,
    )
    ax.set_xlabel("Query Latency (ms)")
    ax.set_ylabel("Recall@10")
    ax.get_legend().set_title(metric_readable)
    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


def plot_ablation_results_combined(
    output_dir: str | Path = "output/ablation",
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    output_path: str = "output/ablation/figs/ablation_combined.pdf",
):
    nlist_df = load_all_results(
        output_dir=str(Path(output_dir) / "nlist"),
        dataset_key=dataset_key,
        test_size=test_size,
    )
    bufcap_df = load_all_results(
        output_dir=str(Path(output_dir) / "max_sl_size"),
        dataset_key=dataset_key,
        test_size=test_size,
    )
    beam_df = load_all_results(
        output_dir=str(Path(output_dir) / "beam_size"),
        dataset_key=dataset_key,
        test_size=test_size,
    )

    nlist_df["query_lat_avg"] = nlist_df["query_lat_avg"] * 1000
    bufcap_df["query_lat_avg"] = bufcap_df["query_lat_avg"] * 1000
    beam_df["query_lat_avg"] = beam_df["query_lat_avg"] * 1000

    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
    sns.lineplot(
        data=nlist_df,
        x="query_lat_avg",
        y="recall_at_k",
        hue="nlist",
        marker="o",
        markersize=6,
        linewidth=1,
        ax=ax1,
    )
    ax1.set_title("Branch Factor")
    ax1.set_xlabel("Query Latency (ms)")
    ax1.set_ylabel("Recall@10")
    ax1.legend(fontsize="small", title="")

    sns.lineplot(
        data=bufcap_df,
        x="query_lat_avg",
        y="recall_at_k",
        hue="max_sl_size",
        marker="o",
        markersize=6,
        linewidth=1,
        ax=ax2,
    )
    ax2.set_title("Buffer Capacity")
    ax2.set_xlabel("Query Latency (ms)")
    ax2.set_ylabel("")
    ax2.legend(fontsize="small", title="")

    sns.lineplot(
        data=beam_df,
        x="query_lat_avg",
        y="recall_at_k",
        hue="beam_size",
        marker="o",
        markersize=6,
        linewidth=1,
        ax=ax3,
    )
    ax3.set_title("Beam Size")
    ax3.set_xlabel("Query Latency (ms)")
    ax3.set_ylabel("")
    ax3.legend(fontsize="small", title="")

    # axes = [ax1, ax2, ax3]
    # shared_ylim = (
    #     min(ax.get_ylim()[0] for ax in axes),
    #     max(ax.get_ylim()[1] for ax in axes),
    # )
    # for ax in axes:
    #     ax.set_ylim(shared_ylim)

    fig.tight_layout()

    print(f"Saving figure to {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)


if __name__ == "__main__":
    fire.Fire(plot_ablation_results_combined)
