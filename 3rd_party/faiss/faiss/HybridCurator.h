#pragma once

#include <vector>
#include <queue>

#include <omp.h>

#include <faiss/Index.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/Heap.h>
#include <faiss/BloomFilter.h>

namespace faiss {

struct VisitedTable;
struct DistanceComputer;

struct Level0HNSW {
    // only store logical pointers to the vectors
    using storage_idx_t = int32_t;

    // pair of (distance, id) used during search
    typedef std::pair<float, storage_idx_t> Node;

    // sort pairs of (id, distance) from nearest to fathest
    struct NodeDistCloser {
        float d;
        int id;
        NodeDistCloser(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistCloser& obj1) const {
            return d < obj1.d;
        }
    };

    // sort pairs of (id, distance) from fathest to nearest
    struct NodeDistFarther {
        float d;
        int id;
        NodeDistFarther(float d, int id) : d(d), id(id) {}
        bool operator<(const NodeDistFarther& obj1) const {
            return d > obj1.d;
        }
    };

    // neighbors[M*i:M*(i+1)] is the list of neighbors of vector i
    std::vector<storage_idx_t> neighbors;

    // number of neighbors at the base level
    int nbNeighbors = 32;

    // expansion factor at construction time
    int efConstruction = 40;

    // expansion factor at query time
    int efSearch = 16;

    // whether or not to check whether the next best distance is good enough
    // during search
    bool check_relative_distance = true;

    // range of entries in the neighbors list
    void neighbor_range(idx_t no, size_t* begin, size_t* end) const;

    explicit Level0HNSW(int M = 32);

    void add_links_starting_from(DistanceComputer& ptdis, storage_idx_t pt_id, storage_idx_t nearest, float d_nearest, omp_lock_t* locks, VisitedTable& vt);

    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_id,
            storage_idx_t entry_pt,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt);

    void search(DistanceComputer& qdis, int k, idx_t* I, float* D, storage_idx_t entry_pt, VisitedTable& vt) const;
};

} // namespace faiss