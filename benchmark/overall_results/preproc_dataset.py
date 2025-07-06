from pathlib import Path

import fire

from benchmark.profiler import Dataset


def preproc_dataset(dataset_key: str, test_size: float, output_dir: str | Path):
    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"Dataset already exists in {output_dir}")
        return

    Dataset.from_dataset_key(dataset_key, test_size=test_size, cache_path=output_dir)
    print(f"Dataset saved to {output_dir}")


if __name__ == "__main__":
    fire.Fire(preproc_dataset)
