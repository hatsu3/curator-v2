# Curator: Efficient Tree-Based Vector Indexing for Filtered Search

At its core, Curator constructs a memory-efficient clustering tree that indexes all vectors and embeds multiple per-label indexes as sub-trees. These per-label indexes are not only extremely lightweight but also capture the unique vector distribution of each label, leading to high search performance and a low memory footprint. Furthermore, each per-label index can be constructed and updated independently with minimal cost, and multiple per-label indexes can be flexibly composed to handle queries with complex filter predicates.

## Repository Structure

- `3rd_party/faiss`: C++ impl of Curator and baselines
  
  - `MultiTenantIndexIVFHierarchical.cpp`: Curator
  - `MultiTenantIndexIVFFlat.cpp`: IVF with metadata filtering
  - `MultiTenantIndexIVFFlatSep.cpp`: IVF with per-tenant indexing
  - `MultiTenantIndexHNSW.cpp`: HNSW with metadata filtering

- `indexes`: Python API for indexes

  - `ivf_hier_faiss.py`: Curator
  - `ivf_flat_mt_faiss.py`: IVF with metadata filtering
  - `ivf_flat_sepidx_faiss.py`: IVF with per-tenant indexing
  - `hnsw_mt_hnswlib.py`: HNSW with metadata filtering
  - `hnsw_sepidx_hnswlib.py`: HNSW with per-tenant indexing

- `dataset`: code for evaluation datasets

  - `arxiv_dataset.py`: arXiv dataset
  - `yfcc100m_dataset.py`: YFCC100M dataset

- `benchmark`: code for running benchmarks

## How to Use

### Install Dependencies

We assume that you have installed Anaconda. To install the required Python packages, run the following command:

```bash
conda env create -f environment.yml -n ann_bench
conda activate ann_bench
```

### Build from Source

```bash
cd 3rd_party/faiss

cmake -B build . \
  -DFAISS_ENABLE_GPU=OFF \
  -DFAISS_ENABLE_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBUILD_TESTING=ON

make -C build -j32 faiss_avx2
make -C build -j32 swigfaiss_avx2
cd build/faiss/python
python setup.py install
```

### Generate Datasets

```bash
mkdir -p data/yfcc100m
yfcc100m_base_url="https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M"
wget -P data/yfcc100m ${yfcc100m_base_url}/base.10M.u8bin
wget -P data/yfcc100m ${yfcc100m_base_url}/base.metadata.10M.spmat

mkdir -p data/arxiv
# manually download arxiv dataset from https://www.kaggle.com/datasets/Cornell-University/arxiv
# and put it at data/arxiv/arxiv-metadata-oai-snapshot.json

python -m dataset.yfcc100m_dataset
python -m dataset.arxiv_dataset
```

### Run Benchmarks

Please refer to scripts in `benchmark/` folder for running experiments. See `benchmark/profiler.py` for the main benchmarking interface.
