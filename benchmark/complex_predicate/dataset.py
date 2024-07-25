import fire
import numpy as np
from tqdm import tqdm

from benchmark.complex_predicate.utils import (
    compute_ground_truth,
    evaluate_predicate,
    generate_random_filters,
)
from benchmark.profiler import Dataset


class ComplexPredicateDataset:
    def __init__(
        self,
        train_vecs: np.ndarray,
        train_mds: list[list[int]],
        test_vecs: np.ndarray,
        template_to_filters: dict[str, list[str]],
        filter_to_ground_truth: dict[str, list[list[int]]],
        filter_to_selectivity: dict[str, float],
    ):
        self.train_vecs = train_vecs
        self.train_mds = train_mds
        self.test_vecs = test_vecs
        self.template_to_filters = template_to_filters
        self.filter_to_ground_truth = filter_to_ground_truth
        self.filter_to_selectivity = filter_to_selectivity

    @property
    def dim(self) -> int:
        return self.train_vecs.shape[1]

    @property
    def num_filters(self) -> int:
        return sum(len(fs) for fs in self.template_to_filters.values())

    @property
    def templates(self) -> list[str]:
        return list(self.template_to_filters.keys())

    @classmethod
    def from_dataset_key(
        cls,
        dataset_key: str,
        test_size: float,
        templates: list[str],
        n_filters_per_template: int,
        n_queries_per_filter: int,
        gt_compute_batch_size: int = 8,
        gt_cache_dir: str | None = None,
        seed: int = 42,
    ):
        dataset = Dataset.from_dataset_key(dataset_key, test_size=test_size)

        template_to_filters = generate_random_filters(
            templates,
            n_filters_per_template,
            all_labels=dataset.num_labels,
            seed=seed,
        )

        query_vecs = dataset.test_vecs[
            np.random.choice(
                dataset.test_vecs.shape[0],
                n_queries_per_filter,
                replace=False,
            )
        ]

        all_filters = [
            filter for filters in template_to_filters.values() for filter in filters
        ]
        filter_to_gt, filter_to_sel = compute_ground_truth(
            query_vecs,
            all_filters,
            dataset.train_vecs,
            dataset.train_mds,
            k=10,
            cache_dir=gt_cache_dir,
            batch_size=gt_compute_batch_size,
        )

        return cls(
            dataset.train_vecs,
            dataset.train_mds,
            query_vecs,
            template_to_filters,
            filter_to_gt,
            filter_to_sel,
        )


class PerPredicateDataset(Dataset):
    def __init__(
        self,
        train_vecs: np.ndarray,
        train_mds: list[list[int]],
        test_vecs: np.ndarray,
        test_mds: list[list[int]],
        ground_truth: np.ndarray,
        all_labels: set[int],
        template_to_filters: dict[str, list[str]],
        filter_to_tid: dict[str, int],
    ):
        super().__init__(
            train_vecs, train_mds, test_vecs, test_mds, ground_truth, all_labels
        )
        self.template_to_filters = template_to_filters
        self.filter_to_tid = filter_to_tid

    @property
    def templates(self) -> list[str]:
        return list(self.template_to_filters.keys())

    def get_template_split(self, template: str) -> Dataset:
        all_tids = set(
            self.filter_to_tid[filter] for filter in self.template_to_filters[template]
        )

        train_mds = [
            [tid for tid in access_list if tid in all_tids]
            for access_list in self.train_mds
        ]
        test_mds = [
            [tid for tid in access_list if tid in all_tids]
            for access_list in self.test_mds
        ]

        ground_truth = list()
        orig_gt_gen = iter(self.ground_truth)
        for access_list in self.test_mds:
            for tid in access_list:
                gt = next(orig_gt_gen)
                if tid in all_tids:
                    ground_truth.append(gt)
        ground_truth = np.array(ground_truth)

        return Dataset(
            self.train_vecs,
            train_mds,
            self.test_vecs,
            test_mds,
            ground_truth,
            all_tids,
        )

    @classmethod
    def from_complex_predicate_dataset(cls, dataset: ComplexPredicateDataset):
        filters = sorted(dataset.filter_to_ground_truth.keys())
        filter_to_tid = {filter: i for i, filter in enumerate(filters)}

        new_train_mds = list()
        for access_list in tqdm(
            dataset.train_mds,
            total=len(dataset.train_mds),
            desc="Generating filtered dataset",
        ):
            new_access_list = list()
            for filter in filters:
                if evaluate_predicate(filter, access_list):
                    new_access_list.append(filter_to_tid[filter])
            new_train_mds.append(new_access_list)

        new_test_mds = [list(range(len(filters))) for _ in dataset.test_vecs]

        filter_to_gt = dataset.filter_to_ground_truth
        ground_truth = list()
        for i, access_list in enumerate(new_test_mds):
            for tenant_id in access_list:
                gt = filter_to_gt[filters[tenant_id]][i]
                ground_truth.append(gt)
        ground_truth = np.array(ground_truth)

        all_labels = set(range(len(filters)))

        return cls(
            dataset.train_vecs,
            new_train_mds,
            dataset.test_vecs,
            new_test_mds,
            ground_truth,
            all_labels,
            dataset.template_to_filters,
            filter_to_tid,
        )


def generate_dataset(
    dataset_key: str = "yfcc100m",
    test_size: float = 0.01,
    templates: list[str] = ["NOT {0}", "AND {0} {1}", "OR {0} {1}"],
    n_filters_per_template: int = 100,
    n_queries_per_filter: int = 100,
    gt_cache_dir: str = "data/ground_truth/complex_predicate",
):
    print(f"Generating dataset {dataset_key} ...")
    print(f"Storing ground truth in {gt_cache_dir} ...")
    ComplexPredicateDataset.from_dataset_key(
        dataset_key,
        test_size=test_size,
        templates=templates,
        n_filters_per_template=n_filters_per_template,
        n_queries_per_filter=n_queries_per_filter,
        gt_cache_dir=gt_cache_dir,
    )

if __name__ == "__main__":
    fire.Fire()
