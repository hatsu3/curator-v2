import os
from pathlib import Path

import fire
import numpy as np

from benchmark.profiler import Dataset


# Helper to write .fvecs format (FAISS standard)
def save_fvecs(fname, arr):
    arr = np.asarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(fname, "wb") as f:
        for i in range(n):
            f.write(np.int32(d).tobytes())
            f.write(arr[i].astype(np.float32).tobytes())


def save_mds_txt(mds, fname):
    with open(fname, "w") as f:
        for row in mds:
            f.write(" ".join(str(x) for x in row) + "\n")


def convert(dataset_key, test_size=0.1, output_dir="."):
    os.makedirs(output_dir, exist_ok=True)
    dataset = Dataset.from_dataset_key(dataset_key, test_size)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    save_fvecs(output_path / "train_vecs_cpp.fvecs", dataset.train_vecs)
    save_fvecs(output_path / "test_vecs_cpp.fvecs", dataset.test_vecs)
    save_mds_txt(dataset.train_mds, output_path / "train_mds_cpp.txt")
    save_mds_txt(dataset.test_mds, output_path / "test_mds_cpp.txt")

    print(f"Output files saved to {output_dir}")


if __name__ == "__main__":
    # python convert_dataset_for_cpp.py yfcc100m --test_size 0.01 --output_dir data/yfcc100m_cpp
    fire.Fire(convert)
