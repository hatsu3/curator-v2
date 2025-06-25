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

using ext_vid_t = label_t; // external vector ID
using int_vid_t = vid_t;   // internal vector ID
using ext_lid_t = tid_t;   // external label/tenant ID
using int_lid_t = tid_t;   // internal label/tenant ID

/*
 * We define the following constraints for the index because later we will
 * use vector IDs (of type vid_t) to encode the location of vectors in the
 * index. This leads to gaps in the vector ID space, which requires us to
 * use a wider type for vector IDs than necessary.
 */
constexpr size_t CURATOR_MAX_BRANCH_FACTOR_LOG2 = 6;
constexpr size_t CURATOR_MAX_LEAF_SIZE_LOG2 = 10;
constexpr size_t CURATOR_MAX_TREE_DEPTH =
        (sizeof(int_vid_t) * 8 - CURATOR_MAX_LEAF_SIZE_LOG2) /
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

using VectorIdAllocator = IdMapping<ext_vid_t, int_vid_t>;
using TenantIdAllocator = IdAllocator<ext_lid_t, int_lid_t>;

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

template <typename T>
struct SortedList {
    std::vector<T> data;

    SortedList() {}

    SortedList(const std::vector<T>& data_) : data(data_) {
        std::sort(data.begin(), data.end());
    }

    SortedList(const SortedList& other) : data(other.data) {}

    SortedList(SortedList&& other) noexcept : data(std::move(other.data)) {}

    SortedList& operator=(const SortedList& other) {
        if (this != &other) {
            data = other.data;
        }
        return *this;
    }

    SortedList& operator=(SortedList&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
        }
        return *this;
    }

    typename std::vector<T>::iterator begin() {
        return data.begin();
    }

    typename std::vector<T>::iterator end() {
        return data.end();
    }

    typename std::vector<T>::const_iterator begin() const {
        return data.begin();
    }

    typename std::vector<T>::const_iterator end() const {
        return data.end();
    }

    void insert(const T& item) {
        auto it = std::lower_bound(data.begin(), data.end(), item);
        data.insert(it, item);
    }

    void erase(const T& item) {
        auto it = std::lower_bound(data.begin(), data.end(), item);
        FAISS_ASSERT_MSG(it != data.end() && *it == item, "item not found");
        data.erase(it);
    }

    bool contains(const T& item) const {
        return std::binary_search(data.begin(), data.end(), item);
    }

    size_t size() const {
        return data.size();
    }

    SortedList merge(const SortedList& other) {
        SortedList result;
        std::merge(
                data.begin(),
                data.end(),
                other.data.begin(),
                other.data.end(),
                std::back_inserter(result.data));
        return result;
    }
};

using ShortList = SortedList<int_vid_t>;

struct TreeNode {
    /* information about the tree structure */
    size_t level;      // the level of this node in the tree
    size_t sibling_id; // the id of this node among its siblings
    TreeNode* parent;
    std::vector<TreeNode*> children;
    int_vid_t node_id;

    /* information about the cluster */
    float* centroid;
    RunningMean variance;

    /* available for all nodes */
    size_t bf_capacity;
    float bf_false_pos;
    bloom_filter bf;
    std::unordered_map<int_lid_t, ShortList> shortlists;

    /* only for leaf nodes */
    ShortList vector_indices; // vectors assigned to this leaf node

    TreeNode(
            size_t level,
            size_t sibling_id,
            TreeNode* parent,
            float* centroid,
            size_t d,
            size_t bf_capacity,
            float bf_false_pos);

    ~TreeNode() {
        free(centroid);
        for (TreeNode* child : children) {
            delete child;
        }
    }

    bloom_filter init_bloom_filter() const {
        bloom_parameters bf_params;
        bf_params.projected_element_count = bf_capacity;
        bf_params.false_positive_probability = bf_false_pos;
        bf_params.random_seed = 0xA5A5A5A5;
        bf_params.compute_optimal_parameters();
        return bloom_filter(bf_params);
    }

    bloom_filter recompute_bloom_filter() const {
        auto bf = init_bloom_filter();

        for (const auto& [tid, shortlist] : shortlists) {
            bf.insert(tid);
        }

        for (const auto& child : children) {
            bf |= child->bf;
        }

        return bf;
    }
};

struct MultiTenantIndexIVFHierarchical : MultiTenantIndex {
    /* construction parameters */
    size_t bf_capacity;
    float bf_false_pos;
    size_t max_sl_size;
    size_t n_clusters;
    size_t clus_niter;
    size_t max_leaf_size;

    /* search parameters */
    size_t nprobe;
    float prune_thres;
    float variance_boost;

    /* main data structures */
    TreeNode* tree_root;
    VectorIdAllocator id_allocator;
    TenantIdAllocator tid_allocator;

    // hack (to be removed later): since the sequential storage
    // does not support external labels, we need to map the
    // internal vector IDs to offset in the sequential storage
    std::unordered_map<int_vid_t, size_t> vid_to_storage_idx;

    // sequential storage for the vectors
    bool own_fields;
    IndexFlat* storage;

    /* auxiliary data structures */
    std::unordered_map<std::string, ext_lid_t> filter_to_label;
    bool track_stats = false;
    mutable std::vector<int> search_stats;

    /* experimental */
    size_t search_ef;
    size_t beam_size;

    MultiTenantIndexIVFHierarchical(
            size_t d,
            size_t n_clusters,
            MetricType metric = METRIC_L2,
            size_t bf_capacity = 1000,
            float bf_false_pos = 0.01,
            size_t max_sl_size = 128,
            size_t clus_niter = 20,
            size_t max_leaf_size = 128,
            size_t nprobe = 3000,
            float prune_thres = 1.6,
            float variance_boost = 0.4,
            size_t search_ef = 0,
            size_t beam_size = 2);

    MultiTenantIndexIVFHierarchical(
            IndexFlat* storage,
            size_t n_clusters,
            size_t bf_capacity = 1000,
            float bf_false_pos = 0.01,
            size_t max_sl_size = 128,
            size_t clus_niter = 20,
            size_t max_leaf_size = 128,
            size_t nprobe = 3000,
            float prune_thres = 1.6,
            float variance_boost = 0.4,
            size_t search_ef = 0,
            size_t beam_size = 2);

    ~MultiTenantIndexIVFHierarchical() override {
        delete tree_root;
        if (own_fields) {
            delete storage;
        }
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

    void train(idx_t n, const float* x, ext_lid_t tid) override;

    void train_helper(TreeNode* node, idx_t n, const float* x);

    void add_vector_with_ids(idx_t n, const float* x, const idx_t* labels)
            override;

    void grant_access(idx_t label, ext_lid_t tid) override;

    void grant_access_helper(TreeNode* node, int_vid_t vid, int_lid_t tid);

    bool remove_vector(idx_t label) override;

    bool revoke_access(idx_t label, ext_lid_t tid) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            ext_lid_t tid,
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
            int_lid_t tid,
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

    void add_vector(idx_t n, const float* x) override {
        FAISS_THROW_MSG("add_vector not supported");
    }

    void reset() override {
        FAISS_THROW_MSG("reset not supported");
    }

    /*
     * Helper functions
     */

    TreeNode* assign_vec_to_leaf(const float* x);

    std::vector<idx_t> get_vector_path(ext_vid_t label) const;

    void split_short_list(TreeNode* node, int_lid_t tid);

    bool merge_short_list(TreeNode* node, int_lid_t tid);

    void locate_vector(ext_vid_t label) const;

    void print_tree_info() const;

    std::string convert_complex_predicate(const std::string& filter) const;

    std::vector<int_vid_t> find_all_qualified_vecs(
            const std::string& filter) const;

    void batch_grant_access(const std::vector<int_vid_t>& vids, int_lid_t tid);

    void build_index_for_filter(const std::string& filter);

    void sanity_check() const;

    TreeNode* find_assigned_leaf(ext_vid_t label) const;

    void memory_usage() const;
};

namespace complex_predicate {

struct TempIndexNode {
    int start, end;
    std::vector<int> children;
    float* centroid; // Non-owning pointer to TreeNode's centroid
};

void build_temp_index_for_filter(
        const MultiTenantIndexIVFHierarchical* index,
        const std::vector<int_vid_t>& sorted_qualified_vecs,
        std::vector<TempIndexNode>& nodes);

void search_temp_index(
        const MultiTenantIndexIVFHierarchical* index,
        const std::vector<int_vid_t>& qualified_vecs,
        const std::vector<TempIndexNode>& nodes,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params);

} // namespace complex_predicate

} // namespace faiss
