#pragma once

#include <memory>
#include <queue>
#include <vector>

#include <omp.h>

#include <faiss/BloomFilter.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/Heap.h>

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

    void add_links_starting_from(
            DistanceComputer& ptdis,
            storage_idx_t pt_id,
            storage_idx_t nearest,
            float d_nearest,
            omp_lock_t* locks,
            VisitedTable& vt);

    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_id,
            storage_idx_t entry_pt,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt);

    void search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            storage_idx_t entry_pt,
            VisitedTable& vt) const;
};

struct HierarchicalZoneMap {
    using storage_idx_t = Level0HNSW::storage_idx_t;

    struct TreeNode {
        using buffer_t = std::vector<storage_idx_t>;

        /* information about the tree structure */

        // use the path from the root to the node as the node id
        std::vector<int8_t> node_id;

        // physical pointer to the parent node and children nodes
        TreeNode* parent = nullptr;
        std::vector<std::unique_ptr<TreeNode>> children;

        // index storing the centroids of the children nodes
        // used to find the nearest child node during search
        IndexFlatL2 quantizer;

        /* aggregate information about the cluster */

        // buffers[tenant_id] stores the logical pointers to the vectors
        // accessible to the tenant within the cluster represented by this node
        // buffers will be pushed down to the children nodes when they are full
        std::unordered_map<tid_t, buffer_t> buffers;
        size_t buf_capacity;

        // a compact representation of the set of tenants that have access to
        // some vectors within the cluster represented by this node
        std::unique_ptr<bloom_filter> bf = nullptr;
        size_t bf_capacity;
        float bf_error_rate;

        /* leaf nodes */

        // leaf nodes stores logical pointers to the vectors within the cluster
        bool is_leaf;
        std::vector<storage_idx_t> points;

        explicit TreeNode(
                size_t d,
                std::vector<int8_t>&& node_id,
                TreeNode* parent,
                size_t bf_capacity,
                float bf_error_rate,
                size_t buf_capacity);
    };

    // dimension of the vectors
    size_t d;

    // parameters for the aggregate data structures
    size_t bf_capacity;
    float bf_error_rate;
    size_t buf_capacity;

    // parameters for the tree structure
    size_t branch_factor;

   private:
    // prevent swig from generating a setter for this attribuet
    std::unique_ptr<TreeNode> tree_root = nullptr;

   public:
    explicit HierarchicalZoneMap(
            size_t d,
            size_t branch_factor,
            size_t bf_capacity,
            float bf_error_rate,
            size_t buf_capacity);

    // train the clustering tree using a representative set of vectors
    // !! caveat: the vectors will be sorted in place during training
    void train(float* x, size_t n);

    // update the tree structure and the aggregate data structures
    // to reflect the addition of a new vector to the index
    void insert(const float* x, storage_idx_t label, tid_t tenant);

    // navigate the tree to the leaf node that is closest to the query vector
    const TreeNode* seek(const float* qv) const;
};

} // namespace faiss