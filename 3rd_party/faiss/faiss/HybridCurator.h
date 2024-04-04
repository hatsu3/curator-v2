#pragma once

#include <memory>
#include <queue>
#include <vector>

#include <omp.h>

#include <faiss/BloomFilter.h>
#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/MultiTenantIndex.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/utils/Heap.h>

namespace faiss {

struct VisitedTable;
struct DistanceComputer;

struct Level0HNSW {
    // pair of (distance, id) used during search
    typedef std::pair<float, idx_t> Node;

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
    std::vector<idx_t> neighbors;

    // number of neighbors at the base level
    int nbNeighbors{32};

    // expansion factor at construction time
    int efConstruction{40};

    // expansion factor at query time
    int efSearch{16};

    // whether or not to check whether the next best distance is good enough
    // during search
    bool check_relative_distance = true;

    // range of entries in the neighbors list
    void neighbor_range(idx_t no, size_t* begin, size_t* end) const;

    explicit Level0HNSW(int M = 32, int efConstruction = 40, int efSearch = 16);

    void add_links_starting_from(
            DistanceComputer& ptdis,
            idx_t pt_id,
            idx_t nearest,
            float d_nearest,
            omp_lock_t* locks,
            VisitedTable& vt);

    void add_with_locks(
            DistanceComputer& ptdis,
            int pt_id,
            idx_t entry_pt,
            std::vector<omp_lock_t>& locks,
            VisitedTable& vt);

    void search(
            DistanceComputer& qdis,
            int k,
            idx_t* I,
            float* D,
            idx_t entry_pt,
            VisitedTable& vt) const;
};

struct HierarchicalZoneMap {
    friend struct HybridCuratorV2;

    using buffer_t = std::vector<idx_t>;
    using node_id_t = std::vector<int8_t>;

    struct TreeNode {
        /* information about the tree structure */

        // use the path from the root to the node as the node id
        node_id_t node_id;

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

        // buffers in leaf nodes have no capacity limit
        bool is_leaf;

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

    // backlink from vectors to containing leaf nodes
    std::unordered_map<idx_t, TreeNode*> vec2leaf;

    // number of nodes in the tree
    size_t size{0};

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
    void train(float* x, size_t n, size_t split_thres = 32);

    void train_with_depth_limit(float* x, size_t n, size_t max_depth);

    // update the tree structure and the aggregate data structures
    // to reflect the addition of a new vector to the index
    void insert(const float* x, idx_t label, tid_t tenant);

    void grant_access(idx_t label, tid_t tenant);

    // navigate the tree to the node that (1) is closest to the query vector
    // and (2) contains a buffer of the querying tenant
    const TreeNode* seek(const float* qv, tid_t tid) const;

    // find an enclosing node for a given leaf node id that contains a buffer
    // for the querying tenant
    const TreeNode* seek(const node_id_t& leaf_id, tid_t tid) const;

    // find the ancestor node of the given leaf node id that is (1) highest in
    // the tree and (2) does not contain any vectors accessible to the tenant
    const TreeNode* seek_empty(const TreeNode* leaf, tid_t tid) const;
};

template <typename T>
struct VisitedSubtreeTable {
    using node_id_t = std::vector<T>;
    using tree_size_t = uint16_t;

    struct TrieNode {
        std::unordered_map<T, tree_size_t> children;
        bool is_end_of_prefix{false};
    };

    std::vector<TrieNode> trie;
    tree_size_t size{1};

    explicit VisitedSubtreeTable(tree_size_t capacity) : trie(capacity) {}

    tree_size_t capacity() const {
        return trie.size();
    }

    void reserve(tree_size_t new_capacity) {
        trie.resize(new_capacity);
    }

    void set(const node_id_t& node_id);

    bool get(const node_id_t& node_id) const;

    void clear();
};

struct HybridCurator {
    using AccessList = std::unordered_set<tid_t>;
    using AccessMatrix = std::unordered_map<idx_t, AccessList>;

    IndexFlatL2 storage;   // storage for the vectors
    Level0HNSW base_level; // graph index at the base level

   private:
    HierarchicalZoneMap zone_map; // zone map capturing per-tenant dist

   public:
    AccessMatrix access_matrix; // store the access control lists

    // TODO: hierarchical zone map and access matrix are not yet thread-safe

    size_t d{0};
    bool is_trained{false};
    idx_t entry_pt{-1};

    explicit HybridCurator(
            size_t d,
            size_t M,
            size_t branch_factor,
            size_t bf_capacity,
            float bf_error_rate,
            size_t buf_capacity);

    ~HybridCurator();

    void train(idx_t n, const float* x, tid_t tid);

    void add_vector(idx_t n, const float* x, tid_t tid);

    void grant_access(idx_t xid, tid_t tid);

    // bool remove_vector(idx_t xid);

    // bool revoke_access(idx_t xid, tid_t tid);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels) const;
};

struct HybridCuratorV2 : MultiTenantIndex {
    using AccessList = HybridCurator::AccessList;
    using AccessMatrix = HybridCurator::AccessMatrix;
    using TreeNode = HierarchicalZoneMap::TreeNode;

   private:
    IndexFlatL2 storage;          // storage for the vectors
    std::unordered_map<idx_t, idx_t> storage_idmap; // maps internal id to external id
    HierarchicalZoneMap zone_map; // zone map capturing per-tenant dist
    AccessMatrix access_matrix;   // store the access control lists

    // graph index for the lowest levels of the zone map
    std::unordered_map<int, Level0HNSW> level_indexes;
    // stores centroids of the nodes at each level
    std::unordered_map<int, IndexFlatL2> level_storages;
    std::unordered_map<int, std::unique_ptr<DistanceComputer>> level_dist_comps;
    // maps (level, local centroid id) to the corresponding tree node
    std::unordered_map<int, std::unordered_map<idx_t, TreeNode*>> idx2node;
    std::unordered_map<const TreeNode*, std::pair<int, idx_t>> node2idx;

    size_t tree_depth{0};
    size_t ef_construction{40};
    float alpha{1.0};

   public:
    explicit HybridCuratorV2(
            size_t d,
            size_t M,
            size_t tree_depth,
            size_t branch_factor,
            float alpha,
            size_t ef_construction,
            size_t bf_capacity,
            float bf_error_rate,
            size_t buf_capacity);

    ~HybridCuratorV2() override = default;

    void train(idx_t n, const float* x, tid_t tid) override;

    void add_vector(idx_t n, const float* x, tid_t tid) override;

    void add_vector_with_ids(idx_t n, const float* x, const idx_t* xids, tid_t tid) override;

    void grant_access(idx_t xid, tid_t tid) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels, 
            const SearchParameters* params = nullptr) const override;
    
    bool remove_vector(idx_t xid, tid_t tid) override NOT_IMPLEMENTED

    bool revoke_access(idx_t xid, tid_t tid) override NOT_IMPLEMENTED
};

} // namespace faiss