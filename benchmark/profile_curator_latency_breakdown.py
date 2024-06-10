import pickle as pkl
import time
from pathlib import Path

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

from benchmark.profile_curator_ood import (
    get_curator_index,
    load_dataset,
    load_or_generate_ood_dataset,
)
from benchmark.utils import get_dataset_config, recall
from indexes.ivf_hier_faiss import IVFFlatMultiTenantBFHierFaiss as CuratorIndex


def run_query(
    index: CuratorIndex,
    train_vecs: np.ndarray,
    test_vecs: np.ndarray,
    train_mds: list[list[int]],
    test_mds: list[list[int]],
    ground_truth: np.ndarray,
):
    # train index
    print("Training index...")
    try:
        index.train(train_vecs, train_mds)
    except NotImplementedError:
        print("Training not necessary, skipping...")

    # insert vectors into index
    for i, (vec, mds) in tqdm(
        enumerate(zip(train_vecs, train_mds)),
        total=len(train_vecs),
        desc="Building index",
    ):
        if not mds:
            continue

        index.create(vec, i, mds[0])
        for md in mds[1:]:
            index.grant_access(i, md)

    # query index
    index.enable_stats_tracking(True)

    query_results = list()
    query_latencies = list()
    query_stats = list()

    for vec, tenant_ids in tqdm(
        zip(test_vecs, test_mds),
        total=len(test_vecs),
        desc="Querying index",
    ):
        for tenant_id in tenant_ids:
            start = time.time()
            ids = index.query(vec, k=10, tenant_id=int(tenant_id))
            query_latencies.append(time.time() - start)
            query_stats.append(index.get_search_stats())  # type: ignore
            query_results.append(ids)

    query_stats = pd.DataFrame(query_stats)

    recalls = list()
    for res, gt in zip(query_results, ground_truth):
        recalls.append(recall([res], gt[None]))

    return query_latencies, recalls, query_stats


def multivariate_linear_regression(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    print(f"Feature names: {X.columns.to_list()}")

    print("Training linear regression model...")
    linreg = LinearRegression().fit(X_train, y_train)
    linreg_r2_score = linreg.score(X_test, y_test)
    print(f"Linear regression R^2 score: {linreg_r2_score}")
    print(f"Linear regression coefficients: {linreg.coef_}")

    print("Training Lasso regression models...")
    lasso = Lasso(random_state=seed)
    lasso_params = {"alpha": np.logspace(-4, 4, 20)}
    grid_search = GridSearchCV(lasso, lasso_params, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_lasso = grid_search.best_estimator_
    lasso_r2_score = best_lasso.score(X_test, y_test)
    print(f"Lasso R^2 score: {lasso_r2_score}")
    print(f"Lasso coefficients: {best_lasso.coef_}")

    print("Training Ridge regression models...")
    ridge = Ridge(random_state=seed)
    ridge_params = {"alpha": np.logspace(-4, 4, 20)}
    grid_search = GridSearchCV(ridge, ridge_params, cv=5, scoring="r2", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_ridge = grid_search.best_estimator_
    ridge_r2_score = best_ridge.score(X_test, y_test)
    print(f"Ridge R^2 score: {ridge_r2_score}")
    print(f"Ridge coefficients: {best_ridge.coef_}")

    results = [
        {
            "model_type": "linreg",
            "r2_score": linreg_r2_score,
            "coefficients": linreg.coef_,
            "best_model": linreg,
        },
        {
            "model_type": "lasso",
            "r2_score": lasso_r2_score,
            "coefficients": best_lasso.coef_,
            "best_model": best_lasso,
        },
        {
            "model_type": "ridge",
            "r2_score": ridge_r2_score,
            "coefficients": best_ridge.coef_,
            "best_model": best_ridge,
        },
    ]

    return results


def exp_curator_latency_breakdown(
    nlist: int = 16,
    prune_thres: float = 1.6,
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    ood: bool = False,
    ood_version: str = "v2",
    ood_dataset_cache_path: str = "yfcc100m_ood_v2.npz",
    output_path: str = "curator_latency_breakdown.pkl",
):
    print(f"Loading dataset...", flush=True)
    dataset_config, __ = get_dataset_config(dataset_key, test_size=test_size)

    if not ood:
        train_vecs, test_vecs, train_mds, test_mds, ground_truth, __ = load_dataset(
            dataset_config
        )
    else:
        train_vecs, test_vecs, train_mds, test_mds, ground_truth = (
            load_or_generate_ood_dataset(
                dataset_config, ood_dataset_cache_path, ood_version
            )
        )

    index = get_curator_index(
        dim=train_vecs.shape[1], nlist=nlist, prune_thres=prune_thres
    )
    latencies, recalls, stats_df = run_query(
        index, train_vecs, test_vecs, train_mds, test_mds, ground_truth
    )

    print(f"Saving results to {output_path}...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump(
        {
            "latencies": latencies,
            "recalls": recalls,
            "stats": stats_df,
            "nlist": nlist,
            "prune_thres": prune_thres,
            "dataset_key": dataset_key,
            "test_size": test_size,
            "ood": ood,
            "ood_version": ood_version,
            "ood_dataset_cache_path": ood_dataset_cache_path,
        },
        open(output_path, "wb"),
    )


def exp_train_linreg_model(
    results_path: str = "curator_latency_breakdown.pkl",
    test_size: float = 0.2,
    seed: int = 42,
    use_features: list[str] = ["n_dists", "n_bf_queries"],
    output_path: str = "curator_trained_models.pkl",
):
    results = pkl.load(open(results_path, "rb"))
    stats_df, latencies = results["stats"], results["latencies"]
    stats_df = stats_df[use_features]
    assert isinstance(stats_df, pd.DataFrame)

    results = multivariate_linear_regression(
        stats_df, pd.Series(latencies), test_size=test_size, seed=seed
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pkl.dump(results, open(output_path, "wb"))


def plot_trained_model_performance(
    results_path: str = "curator_latency_breakdown.pkl",
    trained_model_path: str = "curator_trained_models.pkl",
    test_size: float = 0.2,
    seed: int = 42,
    use_features: list[str] = ["n_dists", "n_bf_queries"],
    output_path: str = "curator_trained_model_performance.png",
):
    print("Loading profiling results...", flush=True)
    results = pkl.load(open(results_path, "rb"))
    stats_df, latencies = results["stats"], results["latencies"]
    assert isinstance(stats_df, pd.DataFrame)

    __, X_test, __, y_test = train_test_split(
        stats_df, pd.Series(latencies), test_size=test_size, random_state=seed
    )
    X_test = X_test[use_features]

    print("Loading trained models...", flush=True)
    trained_models = pkl.load(open(trained_model_path, "rb"))

    __, axes = plt.subplots(1, 3, figsize=(24, 6))
    for res, ax in zip(trained_models, axes):
        y_pred = res["best_model"].predict(X_test)
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Measured latencies")
        ax.set_ylabel("Predicted latencies (s)")
        ax.set_title(f"{res['model_type'].capitalize()} (R^2={res['r2_score']:.2f})")
        lims = [0, min(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k-", alpha=0.75)

    plt.tight_layout()

    print(f"Saving plot to {output_path} ...", flush=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)


def load_best_model(
    results_path: str = "curator_latency_breakdown.pkl",
    trained_model_path: str = "curator_trained_models.pkl",
    model_type: str = "lasso",
    use_features: list[str] = ["n_dists", "n_bf_queries"],
):
    print("Loading profiling results...", flush=True)
    stats_df = pkl.load(open(results_path, "rb"))["stats"]
    stats_df = stats_df[use_features]
    assert isinstance(stats_df, pd.DataFrame)

    print("Loading trained model...", flush=True)
    trained_models = pkl.load(open(trained_model_path, "rb"))
    best_model = next(
        (
            res["best_model"]
            for res in trained_models
            if res["model_type"] == model_type
        ),
        None,
    )

    if best_model is None:
        raise ValueError(f"Model type {model_type} not found in {trained_model_path}")

    print(f"Best {model_type} model coefficients:")
    print(pd.DataFrame({"feature": stats_df.columns, "coefficient": best_model.coef_}))

    return best_model


def plot_curator_latency_breakdown(
    results_path: str = "curator_latency_breakdown.pkl",
    output_path: str = "curator_latency_breakdown.png",
):
    results = pkl.load(open(results_path, "rb"))

    latencies = results["latencies"]


if __name__ == "__main__":
    fire.Fire()
