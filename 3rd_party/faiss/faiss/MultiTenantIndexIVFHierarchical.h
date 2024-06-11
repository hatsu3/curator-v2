#pragma once

#include <stdint.h>
#include <unordered_map>

#include <faiss/BloomFilter.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/MultiTenantIndexIVFFlat.h>
#include <faiss/complex_predicate.h>
#include <faiss/impl/FaissAssert.h>

namespace faiss {

struct IdAllocator {
    std::unordered_set<vid_t> free_list;
    std::unordered_map<label_t, vid_t> label_to_id;
    std::vector<label_t> id_to_label;

    vid_t allocate_id(label_t label);

    void free_id(label_t label);

    const vid_t get_id(label_t label) const {
        auto it = label_to_id.find(label);
        FAISS_THROW_IF_NOT_MSG(it != label_to_id.end(), "label does not exist");
        return it->second;
    }

    const label_t get_label(vid_t vid) const {
        FAISS_THROW_IF_NOT_MSG(
                vid < id_to_label.size() || id_to_label[vid] == -1,
                "id does not exist");
        return id_to_label[vid];
    }
};

struct VectorStore {
    size_t d;
    std::vector<float> vecs;

    VectorStore(size_t d) : d(d) {}

    void add_vector(const float* vec, vid_t vid) {
        size_t offset = vid * d;
        if (offset >= vecs.size()) {
            vecs.resize((vid + 1) * d);
        }

        // we assume that the slot is not occupied
        std::memcpy(vecs.data() + offset, vec, sizeof(float) * d);
    }

    void remove_vector(vid_t vid) {
        size_t offset = vid * d;

        if (offset >= vecs.size()) {
            return;
        } else if (offset == vecs.size() - d) {
            vecs.resize(offset);
        } else {
            std::memset(vecs.data() + offset, 0, sizeof(float) * d);
        }
    }

    const float* get_vec(vid_t vid) const {
        vid_t offset = vid * d;
        FAISS_THROW_IF_NOT_MSG(offset < vecs.size(), "vector does not exist");

        // we assume the slot contains a valid vector
        return vecs.data() + offset;
    }
};

struct AccessMatrix {
    std::vector<std::vector<tid_t>> access_matrix;

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
    IdAllocator id_allocator;
    VectorStore vec_store;
    AccessMatrix access_matrix;

    /* auxiliary data structures */
    size_t update_bf_after;
    std::unordered_map<label_t, TreeNode*> label_to_leaf;

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
            std::vector<tid_t>& tenants);

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
};

} // namespace faiss
