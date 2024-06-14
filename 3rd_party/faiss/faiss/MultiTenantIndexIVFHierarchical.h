#pragma once

#include <stdint.h>
#include <limits>
#include <unordered_map>

#include <faiss/BloomFilter.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/MultiTenantIndexIVFFlat.h>
#include <faiss/complex_predicate.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

/*
 * We define the following constraints for the index because later we will
 * use vector IDs (of type vid_t) to encode the location of vectors in the
 * index. This leads to gaps in the vector ID space, which requires us to
 * use a wider type for vector IDs than necessary.
 */
constexpr size_t CURATOR_MAX_BRANCH_FACTOR_LOG2 = 5;
constexpr size_t CURATOR_MAX_LEAF_SIZE_LOG2 = 8;
constexpr size_t CURATOR_MAX_TREE_DEPTH =
        (sizeof(vid_t) * 8 - CURATOR_MAX_LEAF_SIZE_LOG2) /
        CURATOR_MAX_BRANCH_FACTOR_LOG2;
constexpr size_t CURATOR_MAX_BRANCH_FACTOR = 1
        << CURATOR_MAX_BRANCH_FACTOR_LOG2;
constexpr size_t CURATOR_MAX_LEAF_SIZE = 1 << CURATOR_MAX_LEAF_SIZE_LOG2;

template <typename ExtLabel, typename IntLabel>
struct IdAllocator {
    /*
     * This class is used to manage the mapping between external _label_s and
     * internal _id_s (non-negative integers). Internal ids are allocated in a
     * contiguous manner starting from 0.
     */

    static const IntLabel INVALID_ID;

    std::unordered_set<IntLabel> free_list;
    std::unordered_map<ExtLabel, IntLabel> label_to_id;
    std::vector<ExtLabel> id_to_label;

    IntLabel allocate_id(ExtLabel label);

    ExtLabel allocate_reserved_label() {
        // count down from the maximum possible label value to avoid conflicts
        // with user-provided labels
        for (ExtLabel label = std::numeric_limits<ExtLabel>::max();
             label != std::numeric_limits<ExtLabel>::min();
             label--) {
            if (!has_label(label)) {
                return label;
            }
        }

        FAISS_THROW_MSG("No available reserved label");
    }

    void free_id(ExtLabel label);

    bool has_label(ExtLabel label) const {
        return label_to_id.find(label) != label_to_id.end();
    }

    const IntLabel get_id(ExtLabel label) const {
        auto it = label_to_id.find(label);
        FAISS_THROW_IF_NOT_MSG(it != label_to_id.end(), "label does not exist");
        return it->second;
    }

    const IntLabel get_or_create_id(ExtLabel label) {
        if (has_label(label)) {
            return get_id(label);
        } else {
            return allocate_id(label);
        }
    }

    const ExtLabel get_label(IntLabel id) const {
        if (id >= id_to_label.size() || id_to_label[id] == INVALID_ID) {
            FAISS_THROW_MSG("id does not exist");
        }
        return id_to_label[id];
    }
};

// Unlike IdAllocator, IdMapping is not responsible for managing the allocation
// of internal IDs. Useful when we want a customize allocation strategy.
template <typename ExtLabel, typename IntLabel>
struct IdMapping {
    std::unordered_map<ExtLabel, IntLabel> label_to_id;
    std::unordered_map<IntLabel, ExtLabel> id_to_label;

    void add_mapping(ExtLabel label, IntLabel id) {
        label_to_id[label] = id;
        id_to_label[id] = label;
    }

    void remove_mapping(ExtLabel label) {
        FAISS_THROW_IF_NOT_MSG(has_label(label), "label does not exist");
        id_to_label.erase(label_to_id[label]);
        label_to_id.erase(label);
    }

    bool has_label(ExtLabel label) const {
        return label_to_id.find(label) != label_to_id.end();
    }

    bool has_id(IntLabel id) const {
        return id_to_label.find(id) != id_to_label.end();
    }

    const IntLabel get_id(ExtLabel label) const {
        FAISS_THROW_IF_NOT_MSG(has_label(label), "label does not exist");
        return label_to_id.at(label);
    }

    const ExtLabel get_label(IntLabel id) const {
        FAISS_THROW_IF_NOT_MSG(has_id(id), "id does not exist");
        return id_to_label.at(id);
    }
};

// TODO: use a string to represent external tenant label
using VectorIdAllocator = IdMapping<label_t, vid_t>;
using TenantIdAllocator = IdAllocator<tid_t, tid_t>;

struct VectorStore {
    virtual ~VectorStore() {}

    virtual void add_vector(const float* vec, vid_t vid) = 0;

    virtual void remove_vector(vid_t vid) = 0;

    virtual const float* get_vec(vid_t vid) const = 0;
};

struct OrderedVectorStore: VectorStore {
    size_t d;
    std::vector<float> vecs;

    OrderedVectorStore(size_t d) : d(d) {}

    void add_vector(const float* vec, vid_t vid) override {
        size_t offset = vid * d;
        if (offset >= vecs.size()) {
            vecs.resize((vid + 1) * d);
        }

        // we assume that the slot is not occupied
        std::memcpy(vecs.data() + offset, vec, sizeof(float) * d);
    }

    void remove_vector(vid_t vid) override {
        size_t offset = vid * d;

        if (offset >= vecs.size()) {
            return;
        } else if (offset == vecs.size() - d) {
            vecs.resize(offset);
        } else {
            std::memset(vecs.data() + offset, 0, sizeof(float) * d);
        }
    }

    const float* get_vec(vid_t vid) const override {
        vid_t offset = vid * d;
        FAISS_THROW_IF_NOT_MSG(offset < vecs.size(), "vector does not exist");

        // we assume the slot contains a valid vector
        return vecs.data() + offset;
    }
};

struct UnorderedVectorStore: VectorStore {
    size_t d;
    std::unordered_map<vid_t, std::vector<float>> vecs;

    UnorderedVectorStore(size_t d) : d(d) {}

    void add_vector(const float* vec, vid_t vid) override {
        vecs[vid] = std::vector<float>(vec, vec + d);
    }

    void remove_vector(vid_t vid) override {
        vecs.erase(vid);
    }

    const float* get_vec(vid_t vid) const override {
        auto it = vecs.find(vid);
        FAISS_THROW_IF_NOT_MSG(it != vecs.end(), "vector does not exist");
        return it->second.data();
    }
};

struct OrderedAccessMatrix {
    std::vector<std::vector<tid_t>> access_matrix;

    const std::vector<tid_t>& get_access_list(vid_t vid) const {
        if (vid >= access_matrix.size()) {
            FAISS_THROW_MSG("vector does not exist");
        }
        return access_matrix[vid];
    }

    void add_vector(vid_t vid, tid_t tid) {
        if (vid >= access_matrix.size()) {
            access_matrix.resize(vid + 1);
        }

        // we assume the slot is not occupied
        access_matrix[vid].push_back(tid);
    }

    void remove_vector(vid_t vid, tid_t tid) {
        // we assume the slot contains a valid access list
        access_matrix[vid].clear();
    }

    void grant_access(vid_t vid, tid_t tid) {
        // we assume the slot contains a valid access list
        access_matrix[vid].push_back(tid);
    }

    void revoke_access(vid_t vid, tid_t tid) {
        // we assume the slot contains a valid access list and tid is in the
        // list
        auto& access_list = access_matrix[vid];
        access_list.erase(
                std::remove(access_list.begin(), access_list.end(), tid),
                access_list.end());
    }

    bool has_access(vid_t vid, tid_t tid) const {
        if (vid >= access_matrix.size()) {
            return false;
        }

        // we assume the slot contains a valid access list
        auto& access_list = access_matrix[vid];
        return std::find(access_list.begin(), access_list.end(), tid) !=
                access_list.end();
    }
};

struct UnorderedAccessMatrix {
    std::unordered_map<vid_t, std::vector<tid_t>> access_matrix;

    const std::vector<tid_t>& get_access_list(vid_t vid) const {
        auto it = access_matrix.find(vid);
        FAISS_THROW_IF_NOT_MSG(it != access_matrix.end(), "vector does not exist");
        return it->second;
    } 

    void add_vector(vid_t vid, tid_t tid) {
        auto it = access_matrix.find(vid);
        FAISS_THROW_IF_NOT_MSG(
                it == access_matrix.end(), "vector already exists");
        
        access_matrix[vid] = std::vector<tid_t>{tid};
    }

    void remove_vector(vid_t vid, tid_t tid) {
        auto it = access_matrix.find(vid);
        FAISS_THROW_IF_NOT_MSG(it != access_matrix.end(), "vector does not exist");
        access_matrix.erase(it);
    }

    void grant_access(vid_t vid, tid_t tid) {
        auto it = access_matrix.find(vid);
        FAISS_THROW_IF_NOT_MSG(it != access_matrix.end(), "vector does not exist");
        it->second.push_back(tid);
    }

    void revoke_access(vid_t vid, tid_t tid) {
        auto it = access_matrix.find(vid);
        FAISS_THROW_IF_NOT_MSG(it != access_matrix.end(), "vector does not exist");
        auto& access_list = it->second;
        
        auto it2 = std::find(access_list.begin(), access_list.end(), tid);
        FAISS_THROW_IF_NOT_MSG(it2 != access_list.end(), "tenant does not have access");
        access_list.erase(it2);
    }

    bool has_access(vid_t vid, tid_t tid) const {
        auto it = access_matrix.find(vid);
        if (it == access_matrix.end()) {
            return false;
        }

        auto& access_list = it->second;
        return std::find(access_list.begin(), access_list.end(), tid) !=
                access_list.end();
    }
};

struct RunningMean {
    int n;
    double sum;

    RunningMean() : sum(0.0), n(0) {}

    void add(double x) {
        sum += x;
        n++;
    }

    void remove(double x) {
        FAISS_ASSERT_MSG(n > 0, "no elements to remove");

        sum -= x;
        n--;

        if (n == 0) {
            // reset the sum to avoid numerical issues
            sum = 0.0;
        }
    }

    double get_mean() const {
        if (n > 0) {
            return sum / n;
        } else {
            return 0.0;
        }
    }
};

struct TreeNode {
    /* information about the tree structure */
    size_t level;      // the level of this node in the tree
    size_t sibling_id; // the id of this node among its siblings
    TreeNode* parent;
    std::vector<TreeNode*> children;
    vid_t node_id;

    /* information about the cluster */
    float* centroid;
    IndexFlatL2 quantizer;
    RunningMean variance;

    /* available for all nodes */
    bloom_filter bf;
    std::unordered_map<tid_t, std::vector<vid_t>> shortlists;

    /* only for leaf nodes */
    std::vector<vid_t> vector_indices; // vectors assigned to this leaf node
    std::unordered_map<tid_t, uint32_t> n_vectors_per_tenant;

    TreeNode(
            size_t level,
            size_t sibling_id,
            TreeNode* parent,
            float* centroid,
            size_t d,
            size_t bf_capacity,
            float bf_false_pos);

    ~TreeNode() {
        delete[] centroid;
        for (TreeNode* child : children) {
            delete child;
        }
    }
};

struct MultiTenantIndexIVFHierarchical : MultiTenantIndexIVFFlat {
    /* construction parameters */
    size_t bf_capacity;
    float bf_false_pos;
    size_t max_sl_size;
    size_t n_clusters;
    size_t clus_niter;
    size_t update_bf_interval;
    size_t max_leaf_size;

    /* search parameters */
    size_t nprobe;
    float prune_thres;
    float variance_boost;

    /* main data structures */
    TreeNode* tree_root;
    VectorIdAllocator id_allocator;
    TenantIdAllocator tid_allocator;
    UnorderedVectorStore vec_store;
    UnorderedAccessMatrix access_matrix;

    /* auxiliary data structures */
    size_t update_bf_after;
    std::unordered_map<label_t, TreeNode*> label_to_leaf;
    std::unordered_map<std::string, tid_t> filter_to_label;

    bool track_stats = false;
    mutable std::vector<int> search_stats;

    MultiTenantIndexIVFHierarchical(
            Index* quantizer,
            size_t d,
            size_t n_clusters,
            MetricType metric = METRIC_L2,
            size_t bf_capacity = 1000,
            float bf_false_pos = 0.01,
            size_t max_sl_size = 128,
            size_t update_bf_interval = 100,
            size_t clus_niter = 20,
            size_t max_leaf_size = 128,
            size_t nprobe = 3000,
            float prune_thres = 1.6,
            float variance_boost = 0.4);

    ~MultiTenantIndexIVFHierarchical() override {
        delete tree_root;
    }

    void enable_stats_tracking(bool enable) {
        track_stats = enable;
    }

    std::vector<int> get_search_stats() const {
        return search_stats;
    }

    /*
     * API functions
     */

    void train(idx_t n, const float* x, tid_t tid) override;

    void train_helper(TreeNode* node, idx_t n, const float* x);

    void add_vector_with_ids(
            idx_t n,
            const float* x,
            const idx_t* labels,
            tid_t tid) override;

    void grant_access(idx_t label, tid_t tid) override;

    void grant_access_helper(
            TreeNode* node,
            label_t label,
            tid_t tid,
            std::vector<idx_t>& path);

    bool remove_vector(idx_t label, tid_t tid) override;

    bool revoke_access(idx_t label, tid_t tid) override;

    void update_shortlists_helper(
            TreeNode* leaf,
            vid_t vid,
            const std::vector<tid_t>& tenants);

    void update_bf_helper(TreeNode* leaf);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            const std::string& filter,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    void search_one(
            const float* x,
            idx_t k,
            tid_t tid,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    void search_one(
            const float* x,
            idx_t k,
            const std::string& filter,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    void search_one(
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    /*
     * Helper functions
     */

    TreeNode* assign_vec_to_leaf(const float* x);

    std::vector<idx_t> get_vector_path(label_t label) const;

    void split_short_list(TreeNode* node, tid_t tid);

    bool merge_short_list(TreeNode* node, tid_t tid);

    bool merge_short_list_recursively(TreeNode* node, tid_t tid);

    std::vector<size_t> get_node_path(const TreeNode* node) const;

    void locate_vector(label_t label) const;

    void print_tree_info() const;

    std::string convert_complex_predicate(const std::string& filter) const;

    std::vector<vid_t> find_all_qualified_vecs(const std::string& filter) const;

    void batch_grant_access(const std::vector<vid_t>& vids, tid_t tid);

    void batch_grant_access_helper(
            TreeNode* node,
            const std::vector<vid_t>& vids,
            tid_t tid);

    void build_index_for_filter(const std::string& filter);

    void sanity_check() const;
};

} // namespace faiss
