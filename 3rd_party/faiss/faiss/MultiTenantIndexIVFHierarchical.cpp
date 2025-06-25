// -*- c++ -*-

#include <faiss/MultiTenantIndexIVFHierarchical.h>

// Debug output can be enabled/disabled at compile time
#ifdef CURATOR_ENABLE_DEBUG
#define DEBUG_PRINTF(...) printf(__VA_ARGS__)
#else
#define DEBUG_PRINTF(...) do {} while(0)
#endif

#include <omp.h>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <queue>
#include <set>

#include <faiss/Clustering.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/utils.h>
#include "MultiTenantIndexIVFHierarchical.h"

namespace faiss {
namespace complex_predicate {

void build_temp_index_for_filter(
        const MultiTenantIndexIVFHierarchical* index,
        const std::vector<int_vid_t>& sorted_qualified_vecs,
        std::vector<TempIndexNode>& nodes) {
    // Verify that the input vector is sorted
    FAISS_THROW_IF_NOT_MSG(
            std::is_sorted(
                    sorted_qualified_vecs.begin(), sorted_qualified_vecs.end()),
            "Input vector must be sorted in ascending order");

    // Check for duplicates in input
    std::set<int_vid_t> unique_vids(
            sorted_qualified_vecs.begin(), sorted_qualified_vecs.end());
    if (unique_vids.size() != sorted_qualified_vecs.size()) {
        printf("[ERROR] ❌ INPUT HAS DUPLICATES: %zu unique out of %zu total\n",
               unique_vids.size(),
               sorted_qualified_vecs.size());

        // Print first few duplicates
        std::map<int_vid_t, int> vid_count;
        for (auto vid : sorted_qualified_vecs) {
            vid_count[vid]++;
        }

        int shown = 0;
        for (const auto& [vid, count] : vid_count) {
            if (count > 1 && shown < 10) {
                DEBUG_PRINTF("[DEBUG] - VID %d appears %d times\n", vid, count);
                shown++;
            }
        }
    } else {
        DEBUG_PRINTF("[DEBUG] ✅ Input vector has no duplicates: %zu unique vectors\n",
               sorted_qualified_vecs.size());
    }

    // Build the tree
    // Helper function to recursively build temp index tree
    std::function<int(int, int, TreeNode*)> build_temp_tree =
            [&](int start, int end, TreeNode* curr_node) -> int {
        // Create new node
        int curr_node_idx = nodes.size();
        float* centroid = curr_node->centroid;
        nodes.push_back(TempIndexNode{start, end, /*children*/ {}, centroid});

        DEBUG_PRINTF("[DEBUG] Building node %d: range [%d, %d), size=%d\n",
               curr_node_idx,
               start,
               end,
               end - start);

        // Base case - leaf node or # qualfied vectors is small enough for
        // buffering
        if (end - start <= index->max_sl_size || curr_node->children.empty()) {
            DEBUG_PRINTF("[DEBUG] Node %d is leaf (size=%d, max_sl_size=%zu, has_children=%s)\n",
                   curr_node_idx,
                   end - start,
                   index->max_sl_size,
                   curr_node->children.empty() ? "false" : "true");
            return curr_node_idx;
        }

        // Calculate offset and mask to extract branch idx from vector ID
        auto level = curr_node->level;
        auto offset = sizeof(int_vid_t) * 8 -
                CURATOR_MAX_BRANCH_FACTOR_LOG2 * (level + 1);
        auto mask = (CURATOR_MAX_BRANCH_FACTOR - 1);

        DEBUG_PRINTF("[DEBUG] Node %d: level=%zu, offset=%d, mask=%d\n",
               curr_node_idx,
               level,
               offset,
               mask);

        // Find ranges of vector IDs for each child using binary search
        // We only compare the part of bits in vector IDs that corresponds to
        // the current level
        std::vector<std::pair<int, int>> child_ranges;
        child_ranges.reserve(index->n_clusters);

        for (int child_idx = 0; child_idx < index->n_clusters; child_idx++) {
            // Find first element with current prefix
            int first = std::lower_bound(
                                sorted_qualified_vecs.begin() + start,
                                sorted_qualified_vecs.begin() + end,
                                child_idx,
                                [offset, mask](int_vid_t vid, int child_idx) {
                                    return ((vid >> offset) & mask) < child_idx;
                                }) -
                    sorted_qualified_vecs.begin();

            // Find first element with next prefix
            int last = std::lower_bound(
                               sorted_qualified_vecs.begin() + first,
                               sorted_qualified_vecs.begin() + end,
                               child_idx + 1,
                               [offset, mask](int_vid_t vid, int child_idx) {
                                   return ((vid >> offset) & mask) < child_idx;
                               }) -
                    sorted_qualified_vecs.begin();

            child_ranges.emplace_back(first, last);

            if (first != last) {
                DEBUG_PRINTF("[DEBUG] Child %d: range [%d, %d), size=%d\n",
                       child_idx,
                       first,
                       last,
                       last - first);
            }
        }

        // Recursively build child nodes
        for (int child_idx = 0; child_idx < index->n_clusters; child_idx++) {
            auto& range = child_ranges[child_idx];
            // If at least one vector ID belongs to this child, build the child
            if (range.first != range.second) {
                TreeNode* child_node = curr_node->children[child_idx];
                int child_node_idx =
                        build_temp_tree(range.first, range.second, child_node);
                nodes[curr_node_idx].children.push_back(child_node_idx);
            }
        }

        return curr_node_idx;
    };

    // Start building from root node over all qualified vectors
    if (!sorted_qualified_vecs.empty()) {
        DEBUG_PRINTF("[DEBUG] Starting temp index build with %zu vectors\n",
               sorted_qualified_vecs.size());
        build_temp_tree(0, sorted_qualified_vecs.size(), index->tree_root);
        DEBUG_PRINTF("[DEBUG] Temp index build complete: %zu nodes created\n",
               nodes.size());
    }
}

} // namespace complex_predicate

template <>
const int_lid_t TenantIdAllocator::INVALID_ID = -1;

template <typename ExtLabel, typename IntLabel>
IntLabel IdAllocator<ExtLabel, IntLabel>::allocate_id(ExtLabel label) {
    FAISS_THROW_IF_NOT_MSG(
            label_to_id.find(label) == label_to_id.end(),
            "label already exists");

    IntLabel id;
    if (free_list.empty()) {
        id = id_to_label.size();
        id_to_label.push_back(INVALID_ID);
    } else {
        id = *free_list.begin();
        free_list.erase(free_list.begin());
    }

    label_to_id.emplace(label, id);
    id_to_label[id] = label;

    return id;
}

template <typename ExtLabel, typename IntLabel>
void IdAllocator<ExtLabel, IntLabel>::free_id(ExtLabel label) {
    auto it = label_to_id.find(label);
    FAISS_THROW_IF_NOT_MSG(it != label_to_id.end(), "label does not exist");

    label_to_id.erase(label);

    IntLabel id = it->second;
    if (id == id_to_label.size() - 1) {
        id_to_label.pop_back();
    } else {
        id_to_label[id] = INVALID_ID;
        free_list.emplace(id);
    }
}

TreeNode::TreeNode(
        size_t level,
        size_t sibling_id,
        TreeNode* parent,
        float* centroid,
        size_t d,
        size_t bf_capacity,
        float bf_false_pos)
        : level(level),
          sibling_id(sibling_id),
          parent(parent),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos) {
    if (parent != nullptr) {
        auto offset =
                sizeof(int_vid_t) * 8 - level * CURATOR_MAX_BRANCH_FACTOR_LOG2;
        this->node_id = parent->node_id | (sibling_id << offset);
    } else {
        this->node_id = 0;
    }

    if (centroid == nullptr) {
        this->centroid = nullptr;
    } else {
        this->centroid =
                static_cast<float*>(std::aligned_alloc(64, d * sizeof(float)));
        std::memcpy(this->centroid, centroid, sizeof(float) * d);
    }

    this->bf = init_bloom_filter();
}

MultiTenantIndexIVFHierarchical::MultiTenantIndexIVFHierarchical(
        size_t d,
        size_t n_clusters,
        MetricType metric,
        size_t bf_capacity,
        float bf_false_pos,
        size_t max_sl_size,
        size_t clus_niter,
        size_t max_leaf_size,
        size_t nprobe,
        float prune_thres,
        float variance_boost,
        size_t search_ef,
        size_t beam_size)
        : MultiTenantIndex(d, metric),
          storage(new IndexFlat(d, metric)),
          own_fields(true),
          n_clusters(n_clusters),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos),
          max_sl_size(max_sl_size),
          clus_niter(clus_niter),
          max_leaf_size(max_leaf_size),
          nprobe(nprobe),
          prune_thres(prune_thres),
          variance_boost(variance_boost),
          search_ef(search_ef),
          beam_size(beam_size) {
    FAISS_ASSERT_FMT(
            n_clusters <= CURATOR_MAX_BRANCH_FACTOR,
            "n_clusters should be less than or equal to %zu",
            CURATOR_MAX_BRANCH_FACTOR);

    FAISS_ASSERT_FMT(
            max_leaf_size <= CURATOR_MAX_LEAF_SIZE,
            "max_leaf_size should be less than or equal to %zu",
            CURATOR_MAX_LEAF_SIZE);

    tree_root =
            new TreeNode(0, 0, nullptr, nullptr, d, bf_capacity, bf_false_pos);
}

MultiTenantIndexIVFHierarchical::MultiTenantIndexIVFHierarchical(
        IndexFlat* storage,
        size_t n_clusters,
        size_t bf_capacity,
        float bf_false_pos,
        size_t max_sl_size,
        size_t clus_niter,
        size_t max_leaf_size,
        size_t nprobe,
        float prune_thres,
        float variance_boost,
        size_t search_ef,
        size_t beam_size)
        : MultiTenantIndex(storage->d, storage->metric_type),
          storage(storage),
          own_fields(false),
          n_clusters(n_clusters),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos),
          max_sl_size(max_sl_size),
          clus_niter(clus_niter),
          max_leaf_size(max_leaf_size),
          nprobe(nprobe),
          prune_thres(prune_thres),
          variance_boost(variance_boost),
          search_ef(search_ef),
          beam_size(beam_size) {
    FAISS_ASSERT_FMT(
            n_clusters <= CURATOR_MAX_BRANCH_FACTOR,
            "n_clusters should be less than or equal to %zu",
            CURATOR_MAX_BRANCH_FACTOR);

    FAISS_ASSERT_FMT(
            max_leaf_size <= CURATOR_MAX_LEAF_SIZE,
            "max_leaf_size should be less than or equal to %zu",
            CURATOR_MAX_LEAF_SIZE);

    tree_root =
            new TreeNode(0, 0, nullptr, nullptr, d, bf_capacity, bf_false_pos);
}

void MultiTenantIndexIVFHierarchical::train(
        idx_t n,
        const float* x,
        ext_lid_t tid) {
    train_helper(tree_root, n, x);
}

void MultiTenantIndexIVFHierarchical::train_helper(
        TreeNode* node,
        idx_t n,
        const float* x) {
    if (node->centroid == nullptr) {
        node->centroid = new float[d];
        for (size_t i = 0; i < d; i++) {
            node->centroid[i] = 0;
        }

        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < d; j++) {
                node->centroid[j] += x[i * d + j];
            }
        }

        for (size_t i = 0; i < d; i++) {
            node->centroid[i] /= n;
        }
    }

    // stop if there are too few samples to cluster
    if (n <= max_leaf_size || node->level >= CURATOR_MAX_TREE_DEPTH) {
        return;
    }

    // partition the data into n_clusters clusters
    IndexFlatL2 quantizer(d);
    ClusteringParameters cp;
    cp.niter = clus_niter;
    cp.min_points_per_centroid = 1;
    Clustering clus(d, n_clusters, cp);
    clus.train(n, x, quantizer);
    quantizer.is_trained = true;

    std::vector<idx_t> cluster_ids(n);
    quantizer.assign(n, x, cluster_ids.data());

    // sort the vectors by cluster
    std::vector<float> sorted_x(n * d);
    std::vector<size_t> cluster_size(n_clusters, 0);
    std::vector<size_t> cluster_offsets(n_clusters, 0);

    for (size_t i = 0; i < n; i++) {
        size_t cluster = cluster_ids[i];
        cluster_size[cluster]++;
    }

    cluster_offsets[0] = 0;
    for (size_t i = 1; i < n_clusters; i++) {
        cluster_offsets[i] = cluster_offsets[i - 1] + cluster_size[i - 1];
    }

    std::vector<size_t> tmp_offsets = cluster_offsets;

    for (size_t i = 0; i < n; ++i) {
        size_t cluster = cluster_ids[i];
        size_t curr_offset = tmp_offsets[cluster]++;
        std::memcpy(
                sorted_x.data() + curr_offset * d,
                x + i * d,
                sizeof(float) * d);
    }

    // recursively train children
    for (size_t clus_id = 0; clus_id < n_clusters; clus_id++) {
        TreeNode* child = new TreeNode(
                node->level + 1,
                clus_id,
                node,
                clus.centroids.data() + clus_id * d,
                d,
                bf_capacity,
                bf_false_pos);

        train_helper(
                child,
                cluster_size[clus_id],
                sorted_x.data() + cluster_offsets[clus_id] * d);

        node->children.push_back(child);
    }
}

void MultiTenantIndexIVFHierarchical::add_vector_with_ids(
        idx_t n,
        const float* x,
        const idx_t* labels) {
    for (size_t i = 0; i < n; i++) {
        ext_vid_t label = labels[i];
        const float* xi = x + i * d;

        // add the vector to the leaf node
        TreeNode* leaf = assign_vec_to_leaf(xi);
        auto offset = sizeof(int_vid_t) * 8 -
                leaf->level * CURATOR_MAX_BRANCH_FACTOR_LOG2 -
                CURATOR_MAX_LEAF_SIZE_LOG2;
        int_vid_t local_vid =
                static_cast<int_vid_t>(leaf->vector_indices.size());
        int_vid_t vid = leaf->node_id | (local_vid << offset);

        // add the vector to the vector store and access matrix
        id_allocator.add_mapping(label, vid);
        vid_to_storage_idx[vid] = ntotal;
        if (own_fields) {
            // only modify storage if we own it
            // otherwise, the owner is responsible for adding the vector
            storage->add(1, xi);
        }
        ntotal++;
        leaf->vector_indices.insert(vid);

        TreeNode* curr = leaf;
        while (curr != nullptr) {
            float dist = fvec_L2sqr(xi, curr->centroid, d);
            curr->variance.add(dist);
            curr = curr->parent;
        }
    }
}

void MultiTenantIndexIVFHierarchical::grant_access(
        idx_t label,
        ext_lid_t ext_tid) {
    int_vid_t vid = id_allocator.get_id(label);
    int_lid_t int_tid = tid_allocator.get_or_create_id(ext_tid);
    grant_access_helper(tree_root, vid, int_tid);
}

void MultiTenantIndexIVFHierarchical::grant_access_helper(
        TreeNode* node,
        int_vid_t vid,
        int_lid_t tid) {
    if (node->children.empty()) {
        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            it->second.insert(vid);
        } else {
            node->shortlists.emplace(tid, std::vector<int_vid_t>{vid});
        }

        node->bf.insert(tid);
    } else {
        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            it->second.insert(vid);
            if (it->second.size() > max_sl_size) {
                split_short_list(node, tid);
            }
        } else if (!node->bf.contains(tid)) {
            node->shortlists.emplace(tid, std::vector<int_vid_t>{vid});
        } else {
            auto offset = sizeof(int_vid_t) * 8 -
                    (node->level + 1) * CURATOR_MAX_BRANCH_FACTOR_LOG2;
            auto child_id = (vid >> offset) & (CURATOR_MAX_BRANCH_FACTOR - 1);
            grant_access_helper(node->children[child_id], vid, tid);
        }

        node->bf.insert(tid);
    }
}

bool MultiTenantIndexIVFHierarchical::remove_vector(idx_t label) {
    FAISS_THROW_MSG("remove_vector is not supported");
}

bool MultiTenantIndexIVFHierarchical::revoke_access(
        idx_t label,
        ext_lid_t tid) {
    int_lid_t int_tid = tid_allocator.get_id(tid);
    int_vid_t vid = id_allocator.get_id(label);
    auto leaf = find_assigned_leaf(label);

    // phase 1: find the node that contain shortlist
    auto curr = leaf;
    while (curr != nullptr) {
        if (curr->shortlists.find(int_tid) != curr->shortlists.end()) {
            break;
        }
        curr = curr->parent;
    }

    FAISS_THROW_IF_NOT_MSG(
            curr != nullptr,
            "Cannot find the node that contains the shortlist of the tenant");

    // phase 2: remove the vector from the shortlist
    auto& shortlist = curr->shortlists.at(int_tid);
    shortlist.erase(vid);
    if (shortlist.size() == 0) {
        curr->shortlists.erase(int_tid);
        curr->bf = curr->recompute_bloom_filter();
    }

    // phase 3: recursively merge shortlists and update bloom filters
    while (curr && merge_short_list(curr, int_tid)) {
        curr = curr->parent;
    }

    return true;
}

void MultiTenantIndexIVFHierarchical::search(
        idx_t n,
        const float* x,
        idx_t k,
        ext_lid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (tid < 0) {
        // perform unfiltered search
        search(n, x, k, distances, labels, params);
        return;
    }

    int_lid_t int_tid = tid_allocator.get_id(tid);

    bool inter_query_parallel = getenv("BATCH_QUERY") != nullptr;
    if (inter_query_parallel) {
#pragma omp parallel for schedule(dynamic) if (n > 1)
        for (idx_t i = 0; i < n; i++) {
            search_one(
                    x + i * d,
                    k,
                    int_tid,
                    distances + i * k,
                    labels + i * k,
                    params);
        }
    } else {
        for (idx_t i = 0; i < n; i++) {
            search_one(
                    x + i * d,
                    k,
                    int_tid,
                    distances + i * k,
                    labels + i * k,
                    params);
        }
    }
}

void MultiTenantIndexIVFHierarchical::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    for (idx_t i = 0; i < n; i++) {
        search_one(x + i * d, k, distances + i * k, labels + i * k, params);
    }
}

template <typename T>
using MinHeap = std::priority_queue<T, std::vector<T>, std::greater<T>>;
using HeapForL2 = CMax<float, idx_t>;

namespace {

struct RunningList {
    std::vector<int_vid_t> vids;
    std::vector<float> dists;
    std::vector<int_vid_t> vids_tmp;
    std::vector<float> dists_tmp;
    int capacity;

    RunningList(int capacity) : capacity(capacity) {
        vids.reserve(capacity + 1);
        dists.reserve(capacity + 1);
        vids_tmp.reserve(capacity + 1);
        dists_tmp.reserve(capacity + 1);
    }

    bool insert(int_vid_t vid, float dist) {
        auto it = std::lower_bound(dists.begin(), dists.end(), dist);
        auto pos = it - dists.begin();
        bool updated = false;

        if (dists.size() < capacity) {
            dists.insert(it, dist);
            vids.insert(vids.begin() + pos, vid);
            updated = true;
        } else if (dist < dists.back()) {
            dists.insert(it, dist);
            dists.pop_back();
            vids.insert(vids.begin() + pos, vid);
            vids.pop_back();
            updated = true;
        }

        return updated;
    }

    bool batch_insert(const std::vector<std::pair<float, int_vid_t>>& cands) {
        bool updated = dists.size() < capacity || cands[0].first < dists.back();
        if (!updated) {
            return false;
        }

        int i = 0, j = 0;
        while (i < dists.size() && j < cands.size()) {
            if (dists[i] <= cands[j].first) {
                vids_tmp.push_back(vids[i]);
                dists_tmp.push_back(dists[i]);
                i++;
            } else {
                vids_tmp.push_back(cands[j].second);
                dists_tmp.push_back(cands[j].first);
                j++;
            }

            if (vids_tmp.size() == capacity) {
                break;
            }
        }

        while (vids_tmp.size() < capacity && i < dists.size()) {
            vids_tmp.push_back(vids[i]);
            dists_tmp.push_back(dists[i]);
            i++;
        }

        while (vids_tmp.size() < capacity && j < cands.size()) {
            vids_tmp.push_back(cands[j].second);
            dists_tmp.push_back(cands[j].first);
            j++;
        }

        std::swap(vids, vids_tmp);
        std::swap(dists, dists_tmp);
        vids_tmp.clear();
        dists_tmp.clear();

        return true;
    }

    void resize(int new_capacity) {
        if (new_capacity <= capacity) {
            return;
        }
        capacity = new_capacity;
        vids.reserve(capacity + 1);
        dists.reserve(capacity + 1);
        vids_tmp.reserve(capacity + 1);
        dists_tmp.reserve(capacity + 1);
    }

    void reset() {
        vids.clear();
        dists.clear();
        vids_tmp.clear();
        dists_tmp.clear();
    }
};

using Candidate = std::pair<float, const TreeNode*>;

inline float node_score(
        const TreeNode* node,
        const float* x,
        size_t d,
        float var_boost) {
    float dist = fvec_L2sqr(x, node->centroid, d);
    if (var_boost == 0.0) {
        return dist;
    } else {
        float var = node->variance.get_mean();
        return dist - var_boost * var;
    }
}

inline std::vector<Candidate> beam_search(
        const MultiTenantIndexIVFHierarchical& index,
        const float* x,
        int_lid_t tid,
        size_t beam_width,
        std::vector<Candidate>& unexpanded) {
    std::vector<Candidate> beam;
    std::vector<Candidate> next_beam;

    if (!index.tree_root->bf.contains(tid)) {
        return beam;
    }

    float score = node_score(index.tree_root, x, index.d, index.variance_boost);
    beam.emplace_back(score, index.tree_root);

    while (true) {
        bool updated = false;
        for (auto [score, node] : beam) {
            if (node->shortlists.find(tid) != node->shortlists.end()) {
                next_beam.emplace_back(score, node);
            } else {
                updated = true;

                if (!node->children.empty()) {
                    auto centroid = node->children[0]->centroid;
                    for (size_t i = 0; i < node->children.size() - 1; i++) {
                        auto next_centroid = node->children[i + 1]->centroid;
                        prefetch_L1(next_centroid);
                        auto score = node_score(
                                node->children[i],
                                x,
                                index.d,
                                index.variance_boost);
                        next_beam.emplace_back(score, node->children[i]);
                        centroid = next_centroid;
                    }
                    auto score = node_score(
                            node->children.back(),
                            x,
                            index.d,
                            index.variance_boost);
                    next_beam.emplace_back(score, node->children.back());
                }
            }
        }

        if (!updated) {
            break;
        }

        std::sort(next_beam.begin(), next_beam.end());

        auto n_keep = std::min(beam_width, next_beam.size());
        for (size_t i = n_keep; i < next_beam.size(); i++) {
            unexpanded.push_back(next_beam[i]);
        }
        next_beam.resize(n_keep);

        std::swap(beam, next_beam);
        next_beam.clear();
    }

    return beam;
}

inline void compute_dists_with_prefetch(
        const MultiTenantIndexIVFHierarchical& index,
        const std::vector<int_vid_t>& vids,
        const float* x,
        std::vector<std::pair<float, int_vid_t>>& output) {
    if (vids.empty()) {
        return;
    }

    size_t n_vids = vids.size();
    auto disc = dynamic_cast<FlatCodesDistanceComputer*>(
            index.storage->get_distance_computer());
    disc->set_query(x);

    auto get_label = [&](int i) -> ext_vid_t {
        return index.vid_to_storage_idx.at(vids[i]);
    };

    if (n_vids >= 8) {
        std::array<float, 4> dists;

        prefetch_L1(disc->codes + get_label(0) * disc->code_size);
        prefetch_L1(disc->codes + get_label(1) * disc->code_size);
        prefetch_L1(disc->codes + get_label(2) * disc->code_size);
        prefetch_L1(disc->codes + get_label(3) * disc->code_size);

        size_t limit = (n_vids / 4 - 1) * 4;
        for (size_t i = 0; i < limit; i += 4) {
            prefetch_L1(disc->codes + get_label(i + 4) * disc->code_size);
            prefetch_L1(disc->codes + get_label(i + 5) * disc->code_size);
            prefetch_L1(disc->codes + get_label(i + 6) * disc->code_size);
            prefetch_L1(disc->codes + get_label(i + 7) * disc->code_size);

            disc->distances_batch_4(
                    get_label(i),
                    get_label(i + 1),
                    get_label(i + 2),
                    get_label(i + 3),
                    dists[0],
                    dists[1],
                    dists[2],
                    dists[3]);

            output.emplace_back(dists[0], vids[i]);
            output.emplace_back(dists[1], vids[i + 1]);
            output.emplace_back(dists[2], vids[i + 2]);
            output.emplace_back(dists[3], vids[i + 3]);
        }

        disc->distances_batch_4(
                get_label(limit),
                get_label(limit + 1),
                get_label(limit + 2),
                get_label(limit + 3),
                dists[0],
                dists[1],
                dists[2],
                dists[3]);

        output.emplace_back(dists[0], vids[limit]);
        output.emplace_back(dists[1], vids[limit + 1]);
        output.emplace_back(dists[2], vids[limit + 2]);
        output.emplace_back(dists[3], vids[limit + 3]);
    }

    size_t i = n_vids < 8 ? 0 : n_vids / 4 * 4;
    if (i == n_vids) {
        return;
    }

    prefetch_L1(disc->codes + get_label(i) * disc->code_size);
    for (; i < n_vids - 1; i++) {
        prefetch_L1(disc->codes + get_label(i + 1) * disc->code_size);
        float dist = disc->operator()(get_label(i));
        output.emplace_back(dist, vids[i]);
    }

    float dist = disc->operator()(get_label(n_vids - 1));
    output.emplace_back(dist, vids.back());
}

inline void compute_child_scores_with_prefetch(
        const MultiTenantIndexIVFHierarchical& index,
        const TreeNode* node,
        const float* x,
        MinHeap<Candidate>& output) {
    if (node->children.empty()) {
        return;
    }

    auto var_boost = index.variance_boost;

    auto centroid = node->children[0]->centroid;
    for (size_t i = 0; i < node->children.size() - 1; i++) {
        auto next_centroid = node->children[i + 1]->centroid;
        prefetch_L1(next_centroid);
        auto score = node_score(node->children[i], x, index.d, var_boost);
        output.emplace(score, node->children[i]);
        centroid = next_centroid;
    }

    auto score = node_score(node->children.back(), x, index.d, var_boost);
    output.emplace(score, node->children.back());
}

void filtered_search_experimental(
        const MultiTenantIndexIVFHierarchical& index,
        const float* x,
        idx_t k,
        int_lid_t tid,
        float* distances,
        idx_t* labels,
        int search_ef,
        int beam_size = 0) {
    MinHeap<Candidate> frontier;

    if (beam_size > 0) {
        std::vector<Candidate> unexpanded;
        auto beam = beam_search(index, x, tid, beam_size, unexpanded);
        for (auto& cand : unexpanded) {
            frontier.push(cand);
        }
        for (auto& cand : beam) {
            frontier.push(cand);
        }
    } else {
        float score =
                node_score(index.tree_root, x, index.d, index.variance_boost);
        frontier.emplace(score, index.tree_root);
    }

    RunningList cand_vectors(search_ef);
    std::vector<std::pair<float, int_vid_t>> sorted_cands;
    sorted_cands.reserve(index.max_sl_size);
    std::vector<float> child_dists;
    child_dists.reserve(index.n_clusters);

    while (!frontier.empty()) {
        auto [score, node] = frontier.top();
        frontier.pop();

        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            compute_dists_with_prefetch(
                    index, it->second.data, x, sorted_cands);
            std::sort(sorted_cands.begin(), sorted_cands.end());

            bool updated = cand_vectors.batch_insert(sorted_cands);
            sorted_cands.clear();
            if (!updated) {
                break;
            }
        } else if (node->bf.contains(tid)) {
            compute_child_scores_with_prefetch(index, node, x, frontier);
        }
    }

    heap_heapify<HeapForL2>(k, distances, labels);

    auto n_results = std::min(static_cast<size_t>(k), cand_vectors.vids.size());
    for (size_t i = 0; i < n_results; i++) {
        labels[i] = index.id_allocator.get_label(cand_vectors.vids[i]);
        distances[i] = cand_vectors.dists[i];
    }

    heap_reorder<HeapForL2>(k, distances, labels);
}
}; // namespace

void MultiTenantIndexIVFHierarchical::search_one(
        const float* x,
        idx_t k,
        int_lid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    using Candidate = std::pair<float, const TreeNode*>;

    if (search_ef > 0) {
        filtered_search_experimental(
                *this, x, k, tid, distances, labels, search_ef, beam_size);
        return;
    }

    int n_dists = 0;
    int n_nodes_visited = 0;
    int n_bucks_unpruned = 0;
    int n_bucks_pruned = 0;
    int n_vecs_visited = 0;
    int n_steps = 0;
    int n_bf_queries = 0;
    int n_var_queries = 0;

    auto node_priority = [&](TreeNode* node) {
        n_dists++;
        n_nodes_visited++;
        n_var_queries++;

        // Directly compute distance between the query and node centroids
        float dist = fvec_L2sqr(x, node->centroid, d);
        float var = node->variance.get_mean();
        float score = dist - this->variance_boost * var;
        return score;
    };

    MinHeap<Candidate> pq;
    pq.emplace(node_priority(tree_root), tree_root);

    size_t n_cand_vecs = 0;
    std::vector<Candidate> buckets;

    // Compute distances between the query and indexed vectors
    DistanceComputer* disc = storage->get_distance_computer();
    disc->set_query(x);

    while (!pq.empty() && n_cand_vecs < this->nprobe) {
        n_steps++;
        auto [score, node] = pq.top();
        pq.pop();

        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            n_var_queries++;
            float var = node->variance.get_mean();
            float dist = score + this->variance_boost * var;
            buckets.emplace_back(dist, node);
            n_cand_vecs += it->second.size();
            continue;
        }

        n_bf_queries++;
        if (!node->bf.contains(tid)) {
            continue;
        }

        if (!node->children.empty()) {
            for (auto child : node->children) {
                pq.emplace(node_priority(child), child);
            }
            continue;
        }

        // We should only reach here due to false positives in the bloom filter
    }

    std::sort(buckets.begin(), buckets.end());
    n_bucks_unpruned = buckets.size();

    heap_heapify<HeapForL2>(k, distances, labels);

    if (buckets.empty()) {
        return;
    }

    float min_buck_dist = buckets[0].first;

    for (auto [dist, node] : buckets) {
        if (dist > prune_thres * min_buck_dist) {
            break;
        }

        n_bucks_pruned++;
        for (auto vid : node->shortlists.at(tid)) {
            n_dists++;
            n_vecs_visited++;
            ext_vid_t lbl = id_allocator.get_label(vid);
            float dist = disc->operator()(lbl);
            if (dist < distances[0]) {
                maxheap_replace_top(k, distances, labels, dist, lbl);
            }
        }
    }

    heap_reorder<HeapForL2>(k, distances, labels);

    if (track_stats) {
        search_stats.clear();
        search_stats.push_back(n_dists);
        search_stats.push_back(n_nodes_visited);
        search_stats.push_back(n_bucks_unpruned);
        search_stats.push_back(n_bucks_pruned);
        search_stats.push_back(n_vecs_visited);
        search_stats.push_back(n_steps);
        search_stats.push_back(n_bf_queries);
        search_stats.push_back(n_var_queries);
    }
}

void MultiTenantIndexIVFHierarchical::search_one(
        const float* x,
        idx_t k,
        const std::string& filter,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    using namespace complex_predicate;
    using Candidate = std::tuple<float, const TreeNode*, VarMap, State>;
    using Bucket = std::tuple<float, const TreeNode*, State>;

    DistanceComputer* disc = storage->get_distance_computer();
    disc->set_query(x);

    // update var map based on short lists and bloom filter of current node
    auto update_var_map = [&](const TreeNode* node, VarMap var_map) -> VarMap {
        auto new_var_map = std::unordered_map<std::string, State>();

        for (const auto& var : var_map->unresolved_vars()) {
            int_lid_t tid = std::stoi(var);
            auto it = node->shortlists.find(tid);
            if (it != node->shortlists.end()) {
                new_var_map[var] =
                        make_state(Type::SOME, true, it->second.data);
            } else if (!node->bf.contains(tid)) {
                new_var_map[var] = STATE_NONE;
            }
        }

        return var_map->update(new_var_map);
    };

    // remove vectors in buffer that are not in the subtree rooted at node
    auto remove_external_vecs = [&](const TreeNode* node,
                                    const Buffer& buffer) -> Buffer {
        auto shift = sizeof(int_vid_t) * 8 -
                CURATOR_MAX_BRANCH_FACTOR_LOG2 * node->level;
        auto shifted_node_id = node->node_id >> shift;

        Buffer new_buffer;
        for (auto vid : buffer) {
            if ((vid >> shift) == shifted_node_id) {
                new_buffer.push_back(vid);
            }
        }

        return new_buffer;
    };

    // helper functions for updating the result heap
    auto add_buffer_to_heap = [&](const Buffer& buffer) {
        for (auto vid : buffer) {
            ext_vid_t label = id_allocator.get_label(vid);
            float dist = disc->operator()(label);
            if (dist < distances[0]) {
                maxheap_replace_top(k, distances, labels, dist, label);
            }
        }
    };

    auto node_priority = [&](TreeNode* node) {
        float dist = fvec_L2sqr(x, node->centroid, d);
        float var = node->variance.get_mean();
        float score = dist - this->variance_boost * var;
        return score;
    };

    // parse the filter expression
    auto var_map_data = std::unordered_map<std::string, State>();
    auto filter_expr = parse_formula(filter, &var_map_data);
    auto var_map = make_var_map(std::move(var_map_data));

    // initialize the search frontier and result heap
    MinHeap<Candidate> pq;
    float root_priority = node_priority(tree_root);
    pq.emplace(root_priority, tree_root, var_map, STATE_UNKNOWN);

    int n_cand_vecs = 0;
    std::vector<Bucket> buckets;

    while (!pq.empty() && n_cand_vecs < nprobe) {
        auto [score, node, vmap, state] = pq.top();
        pq.pop();

        // if not a terminal state, update the var map first
        if (*state == Type::UNKNOWN) {
            vmap = update_var_map(node, vmap);
            state = filter_expr->evaluate(vmap, false);
        }

        if (*state == Type::NONE) {
            continue;
        }

        if (*state == Type::SOME) {
            state = filter_expr->evaluate(vmap, true);
            state = make_state(
                    Type::SOME,
                    true,
                    remove_external_vecs(node, state->short_list));

            if (*state == Type::SOME) {
                auto var = node->variance.get_mean();
                auto dist = score + this->variance_boost * var;
                buckets.emplace_back(dist, node, state);
                n_cand_vecs += state->short_list.size();
            }

            continue;
        }

        if (node->children.empty()) {
            auto var = node->variance.get_mean();
            auto dist = score + this->variance_boost * var;

            switch (state->type) {
                case Type::ALL:
                    buckets.emplace_back(dist, node, state);
                    n_cand_vecs += node->vector_indices.size();
                    break;
                case Type::MOST: {
                    state = filter_expr->evaluate(vmap, true);
                    state = make_state(
                            Type::MOST,
                            true,
                            remove_external_vecs(node, state->exclude_list));

                    buckets.emplace_back(dist, node, state);
                    n_cand_vecs += node->vector_indices.size();
                    n_cand_vecs -= state->exclude_list.size();
                    break;
                }
                default:
                    // should only reach here due to false positives
                    printf("False positive in bloom filter\n");
            }
        } else {
            // state could be either ALL, MOST, or UNKNOWN
            // in any case, we need to recurse into the children nodes
            for (auto child : node->children) {
                auto child_score = node_priority(child);
                // var map and state may contain vectors that are not in the
                // subtree rooted at child
                pq.emplace(child_score, child, vmap, state);
            }
        }
    }

    heap_heapify<HeapForL2>(k, distances, labels);

    if (buckets.empty()) {
        return;
    }

    float min_buck_dist = std::get<0>(buckets[0]);
    for (auto [dist, node, state] : buckets) {
        if (dist > this->prune_thres * min_buck_dist) {
            break;
        }

        if (*state == Type::SOME) {
            add_buffer_to_heap(state->short_list);
        } else if (*state == Type::ALL) {
            add_buffer_to_heap(node->vector_indices.data);
        } else if (*state == Type::MOST) {
            auto node_vecs = node->vector_indices.data;
            auto diff = buffer_difference(node_vecs, state->exclude_list);
            add_buffer_to_heap(diff);
        }
    }

    heap_reorder<HeapForL2>(k, distances, labels);
}

void MultiTenantIndexIVFHierarchical::search_one(
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    using Candidate = std::pair<float, const TreeNode*>;

    auto node_priority = [&](TreeNode* node) {
        float dist = fvec_L2sqr(x, node->centroid, d);
        float var = node->variance.get_mean();
        float score = dist - this->variance_boost * var;
        return score;
    };

    MinHeap<Candidate> pq;
    pq.emplace(node_priority(tree_root), tree_root);

    int n_cand_vecs = 0;
    std::vector<Candidate> buckets;

    // Compute distances between the query and indexed vectors
    DistanceComputer* disc = storage->get_distance_computer();
    disc->set_query(x);

    while (!pq.empty() && n_cand_vecs < nprobe) {
        auto [score, node] = pq.top();
        pq.pop();

        if (!node->children.empty()) {
            for (auto child : node->children) {
                pq.emplace(node_priority(child), child);
            }
        } else {
            float var = node->variance.get_mean();
            float dist = score + this->variance_boost * var;
            buckets.emplace_back(dist, node);
            n_cand_vecs += node->vector_indices.size();
        }
    }

    std::sort(buckets.begin(), buckets.end());

    heap_heapify<HeapForL2>(k, distances, labels);

    if (buckets.empty()) {
        return;
    }

    float min_buck_dist = buckets[0].first;
    for (auto [dist, node] : buckets) {
        if (dist > this->prune_thres * min_buck_dist) {
            break;
        }

        for (auto vid : node->vector_indices) {
            ext_vid_t lbl = id_allocator.get_label(vid);
            float dist = disc->operator()(lbl);
            if (dist < distances[0]) {
                maxheap_replace_top(k, distances, labels, dist, lbl);
            }
        }
    }

    heap_reorder<HeapForL2>(k, distances, labels);
}

TreeNode* MultiTenantIndexIVFHierarchical::assign_vec_to_leaf(const float* x) {
    TreeNode* curr = tree_root;

    while (!curr->children.empty()) {
        idx_t child_id;
        float min_dist = std::numeric_limits<float>::max();

        for (auto i = 0; i < curr->children.size(); i++) {
            auto child_centroid = curr->children[i]->centroid;
            float dist = fvec_L2sqr(x, child_centroid, d);
            if (dist < min_dist) {
                min_dist = dist;
                child_id = i;
            }
        }

        curr = curr->children[child_id];
    }
    return curr;
}

std::vector<idx_t> MultiTenantIndexIVFHierarchical::get_vector_path(
        ext_vid_t label) const {
    std::vector<idx_t> path;
    int_vid_t vid = id_allocator.get_id(label);
    auto curr = tree_root;
    auto shift = sizeof(int_vid_t) * 8 - CURATOR_MAX_BRANCH_FACTOR_LOG2;
    while (!curr->children.empty()) {
        auto child_id = (vid >> shift) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        path.push_back(child_id);
        curr = curr->children[child_id];
        shift -= CURATOR_MAX_BRANCH_FACTOR_LOG2;
    }
    return path;
}

void MultiTenantIndexIVFHierarchical::split_short_list(
        TreeNode* node,
        int_lid_t tid) {
    if (node->children.empty() ||
        node->shortlists.at(tid).size() <= max_sl_size) {
        return;
    }

    std::vector<std::vector<int_vid_t>> child_sls(node->children.size());
    for (int_vid_t vid : node->shortlists.at(tid)) {
        auto offset = sizeof(int_vid_t) * 8 -
                CURATOR_MAX_BRANCH_FACTOR_LOG2 * (node->level + 1);
        auto child_id = (vid >> offset) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        child_sls[child_id].push_back(vid);
    }

    node->shortlists.erase(tid);
    for (size_t i = 0; i < node->children.size(); i++) {
        if (child_sls[i].empty()) {
            continue;
        }

        TreeNode* child = node->children[i];
        child->shortlists.emplace(tid, child_sls[i]);
        child->bf.insert(tid);
    }

    // in rare cases, the short list of a child node may still exceed the
    // threshold
    for (size_t i = 0; i < node->children.size(); i++) {
        if (child_sls[i].size() > max_sl_size) {
            split_short_list(node->children[i], tid);
        }
    }
}

bool MultiTenantIndexIVFHierarchical::merge_short_list(
        TreeNode* node,
        int_lid_t tid) {
    if (node->parent == nullptr) {
        return false;
    }

    size_t total_sl_size = 0;
    for (TreeNode* sibling : node->parent->children) {
        auto it = sibling->shortlists.find(tid);
        if (it != sibling->shortlists.end()) {
            total_sl_size += it->second.size();
        } else if (sibling->bf.contains(tid)) {
            return false;
        }
    }

    if (total_sl_size > max_sl_size) {
        return false;
    }

    ShortList combined_sl;
    for (TreeNode* sibling : node->parent->children) {
        auto it = sibling->shortlists.find(tid);
        if (it != sibling->shortlists.end()) {
            combined_sl = combined_sl.merge(it->second);
            sibling->shortlists.erase(tid);
            sibling->bf = sibling->recompute_bloom_filter();
        }
    }
    node->parent->shortlists.emplace(tid, combined_sl);

    return true;
}

void MultiTenantIndexIVFHierarchical::locate_vector(ext_vid_t label) const {
    auto path = get_vector_path(label);

    printf("Found vector %u at path: ", label);
    for (auto id : path) {
        printf("%lu ", id);
    }
    printf("\n");
}

void MultiTenantIndexIVFHierarchical::print_tree_info() const {
    size_t total_nodes = 0;
    size_t total_leaf_nodes = 0;
    double total_depth = 0;
    size_t total_vectors = 0;
    std::vector<size_t> leaf_sizes;

    std::queue<const TreeNode*> q;
    q.push(tree_root);

    while (!q.empty()) {
        auto node = q.front();
        q.pop();

        total_nodes++;
        if (node->children.empty()) {
            total_leaf_nodes++;
            total_depth += node->level;
            leaf_sizes.push_back(node->vector_indices.size());
            total_vectors += node->vector_indices.size();
        } else {
            for (auto child : node->children) {
                q.push(child);
            }
        }
    }

    double leaf_size_avg = total_vectors / total_leaf_nodes;
    double leaf_size_var = 0;
    for (size_t size : leaf_sizes) {
        leaf_size_var += (size - leaf_size_avg) * (size - leaf_size_avg);
    }
    leaf_size_var /= total_leaf_nodes;
    double leaf_size_std = sqrt(leaf_size_var);

    auto max_leaf_size =
            *std::max_element(leaf_sizes.begin(), leaf_sizes.end());
    std::vector<size_t> leaf_size_hist(max_leaf_size + 1, 0);
    for (size_t size : leaf_sizes) {
        leaf_size_hist[size]++;
    }

    printf("Total number of tree nodes: %lu\n", total_nodes);
    printf("Total number of leaf nodes: %lu\n", total_leaf_nodes);
    printf("Average depth of leaf nodes: %.2f\n",
           total_depth / total_leaf_nodes);

    printf("Average leaf node size: %.2f\n", leaf_size_avg);
    printf("Standard deviation of leaf node sizes: %.2f\n", leaf_size_std);
    printf("Leaf node size histogram: ");
    for (size_t i = 0; i < leaf_size_hist.size(); i++) {
        if (leaf_size_hist[i] > 0) {
            printf("(%lu, %lu) ", i, leaf_size_hist[i]);
        }
    }
    fflush(stdout);
}

std::string MultiTenantIndexIVFHierarchical::convert_complex_predicate(
        const std::string& filter) const {
    using namespace complex_predicate;

    std::string converted_filter;
    auto tokens = tokenize_formula(filter);

    for (auto i = 0; i < tokens.size(); i++) {
        auto token = tokens[i];

        if (token != "AND" && token != "OR" && token != "NOT") {
            int_lid_t tenant_id = tid_allocator.get_id(std::stol(token));
            token = std::to_string(tenant_id);
        }

        converted_filter = converted_filter + token;
        if (i < tokens.size() - 1) {
            converted_filter = converted_filter + " ";
        }
    }

    return converted_filter;
}

std::vector<int_vid_t> MultiTenantIndexIVFHierarchical::find_all_qualified_vecs(
        const std::string& filter) const {
    using namespace complex_predicate;
    using Candidate = std::tuple<const TreeNode*, VarMap, State>;

    auto update_var_map = [&](const TreeNode* node, VarMap var_map) -> VarMap {
        auto new_var_map = std::unordered_map<std::string, State>();

        for (const auto& var : var_map->unresolved_vars()) {
            int_lid_t tid = std::stoi(var);
            auto it = node->shortlists.find(tid);
            if (it != node->shortlists.end()) {
                auto state = make_state(Type::SOME, true, it->second.data);
                new_var_map[var] = state;
            } else if (!node->bf.contains(tid)) {
                new_var_map[var] = STATE_NONE;
            }
        }

        return var_map->update(new_var_map);
    };

    auto remove_external_vecs = [&](const TreeNode* node,
                                    const Buffer& buffer) -> Buffer {
        auto shift = sizeof(int_vid_t) * 8 -
                CURATOR_MAX_BRANCH_FACTOR_LOG2 * node->level;
        auto shifted_node_id = node->node_id >> shift;

        Buffer new_buffer;
        for (auto vid : buffer) {
            if ((vid >> shift) == shifted_node_id) {
                new_buffer.push_back(vid);
            }
        }

        return new_buffer;
    };

    auto var_map_data = std::unordered_map<std::string, State>();
    auto filter_expr = parse_formula(filter, &var_map_data);
    auto var_map = make_var_map(std::move(var_map_data));

    std::vector<int_vid_t> qual_vecs;
    std::queue<Candidate> frontier;
    frontier.emplace(tree_root, var_map, STATE_UNKNOWN);

    while (!frontier.empty()) {
        auto [node, vmap, state] = frontier.front();
        frontier.pop();

        if (*state == Type::UNKNOWN) {
            vmap = update_var_map(node, vmap);
            state = filter_expr->evaluate(vmap, false);
        }

        if (*state == Type::NONE) {
            continue;
        }

        if (*state == Type::SOME) {
            state = filter_expr->evaluate(vmap, true);
            auto buffer = remove_external_vecs(node, state->short_list);
            qual_vecs.insert(qual_vecs.end(), buffer.begin(), buffer.end());
            continue;
        }

        if (node->children.empty()) {
            if (*state == Type::ALL) {
                auto& buffer = node->vector_indices.data;
                qual_vecs.insert(qual_vecs.end(), buffer.begin(), buffer.end());
            } else if (*state == Type::MOST) {
                state = filter_expr->evaluate(vmap, true);
                auto buffer = remove_external_vecs(node, state->exclude_list);
                auto& leaf_vecs = node->vector_indices.data;
                auto diff = buffer_difference(leaf_vecs, buffer);
                qual_vecs.insert(qual_vecs.end(), diff.begin(), diff.end());
            } else {
                // printf("False positive in bloom filter\n");
            }
        } else {
            for (auto child : node->children) {
                frontier.emplace(child, vmap, state);
            }
        }
    }

    return qual_vecs;
}

void MultiTenantIndexIVFHierarchical::batch_grant_access(
        const std::vector<int_vid_t>& vids,
        int_lid_t tid) {
    tree_root->shortlists.emplace(tid, vids);
    tree_root->bf.insert(tid);
    split_short_list(tree_root, tid);
}

void MultiTenantIndexIVFHierarchical::build_index_for_filter(
        const std::string& filter) {
    printf("Building index for filter: %s\n", filter.c_str());
    auto converted_filter = convert_complex_predicate(filter);

    auto qualified_vecs = find_all_qualified_vecs(converted_filter);
    printf("Found %lu qualified vectors\n", qualified_vecs.size());
    if (qualified_vecs.empty()) {
        return;
    }

    ext_lid_t filter_label = tid_allocator.allocate_reserved_label();
    int_lid_t filter_tid = tid_allocator.allocate_id(filter_label);
    printf("Assigned external ID: %u, internal ID: %u\n",
           filter_label,
           filter_tid);
    filter_to_label.emplace(filter, filter_label);

    batch_grant_access(qualified_vecs, filter_tid);
}

namespace {
bool check_bloom_filter(
        const MultiTenantIndexIVFHierarchical& index,
        const TreeNode& node) {
    for (auto child : node.children) {
        if (!check_bloom_filter(index, *child)) {
            return false;
        }
    }

    auto expected_bf = node.recompute_bloom_filter();
    return node.bf == expected_bf;
}

std::pair<bool, std::set<int_lid_t>> check_shortlists(
        const MultiTenantIndexIVFHierarchical& index,
        const TreeNode& node) {
    // recursively check the shortlists in the descendant nodes

    std::set<int_lid_t> shortlists_in_desc;
    for (auto child : node.children) {
        auto [success, sls] = check_shortlists(index, *child);
        if (!success) {
            return {false, {}};
        }
        shortlists_in_desc.insert(sls.begin(), sls.end());
    }

    // check 1. shortlist size should not exceed the threshold

    for (const auto& [tenant, shortlist] : node.shortlists) {
        if (shortlist.size() > index.max_sl_size) {
            printf("Oversized shortlist\n");
            return {false, {}};
        }
    }

    // check 2. shortlist merging should be done correctly

    if (!node.children.empty()) {
        std::set<int_lid_t> all_tenants;
        for (auto child : node.children) {
            for (const auto& [tenant, shortlist] : child->shortlists) {
                all_tenants.insert(tenant);
            }
        }

        for (auto tenant : all_tenants) {
            size_t total_sl_size = 0;
            for (auto child : node.children) {
                if (!child->bf.contains(tenant)) {
                    continue;
                } else if (
                        child->shortlists.find(tenant) !=
                        child->shortlists.end()) {
                    total_sl_size += child->shortlists.at(tenant).size();
                } else {
                    total_sl_size += index.max_sl_size + 1;
                    break;
                }
            }

            if (total_sl_size <= index.max_sl_size) {
                printf("Fail to merge\n");
                return {false, {}};
            }
        }
    }

    // check 3. there should not be two shortlists of the same tenant in any
    // path

    std::set<int_lid_t> shortlists_in_node;
    for (const auto& [tenant, shortlist] : node.shortlists) {
        shortlists_in_node.insert(tenant);
    }

    std::set<int_lid_t> intersect;
    std::set_intersection(
            shortlists_in_desc.begin(),
            shortlists_in_desc.end(),
            shortlists_in_node.begin(),
            shortlists_in_node.end(),
            std::inserter(intersect, intersect.begin()));

    if (intersect.size() > 0) {
        printf("Duplicate shortlists\n");
        return {false, {}};
    }

    std::set<int_lid_t> all_shortlists;
    std::set_union(
            shortlists_in_desc.begin(),
            shortlists_in_desc.end(),
            shortlists_in_node.begin(),
            shortlists_in_node.end(),
            std::inserter(all_shortlists, all_shortlists.begin()));

    return {true, all_shortlists};
}
} // namespace

void MultiTenantIndexIVFHierarchical::sanity_check() const {
    printf("Sanity check:\n");
    printf("Checking bloom filters...\n");
    check_bloom_filter(*this, *tree_root);
    printf("Checking shortlists...\n");
    check_shortlists(*this, *tree_root);
}

TreeNode* MultiTenantIndexIVFHierarchical::find_assigned_leaf(
        ext_vid_t label) const {
    int_vid_t vid = id_allocator.get_id(label);
    auto curr = tree_root;
    auto shift = sizeof(int_vid_t) * 8 - CURATOR_MAX_BRANCH_FACTOR_LOG2;
    while (!curr->children.empty()) {
        auto child_id = (vid >> shift) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        curr = curr->children[child_id];
        shift -= CURATOR_MAX_BRANCH_FACTOR_LOG2;
    }
    return curr;
}

void MultiTenantIndexIVFHierarchical::memory_usage() const {
    auto unordered_map_memory_usage = [](const auto& map) {
        const auto& [key, value] = *map.begin();
        size_t per_elem = sizeof(key) + sizeof(value) + sizeof(void*);
        return map.bucket_count() * sizeof(void*) + map.size() * per_elem;
    };

    auto vector_memory_usage = [](const auto& vec) {
        return vec.capacity() * sizeof(*vec.begin());
    };

    auto vector_payload_size = [](const auto& vec) {
        return vec.size() * sizeof(*vec.begin());
    };

    auto bloom_filter_size = [](bloom_filter& bf) {
        return sizeof(unsigned int) * bf.hash_count() + bf.size() / 8 +
                sizeof(unsigned int) + 4 * sizeof(unsigned long long int) +
                sizeof(double);
    };

    auto short_list_size = [&](const ShortList& sl) {
        return vector_memory_usage(sl.data);
    };

    auto short_lists_size = [&](const auto& shortlists) {
        size_t size = unordered_map_memory_usage(shortlists);
        for (const auto& [lid, sl] : shortlists) {
            size += short_list_size(sl);
        }
        return size;
    };

    auto short_lists_payload_size = [&](const auto& shortlists) {
        size_t size = 0;
        for (const auto& [lid, sl] : shortlists) {
            size += vector_payload_size(sl.data);
        }
        return size;
    };

    auto node_attrs_size = [&](const TreeNode* node) {
        return sizeof(size_t) * 3 + sizeof(void*) * 2 + sizeof(int_vid_t) +
                sizeof(float) + sizeof(RunningMean) + sizeof(node->children) +
                sizeof(node->shortlists);
    };

    size_t id_allocator_size =
            unordered_map_memory_usage(id_allocator.label_to_id) +
            unordered_map_memory_usage(id_allocator.id_to_label);

    size_t tid_allocator_size =
            unordered_map_memory_usage(tid_allocator.label_to_id) +
            vector_memory_usage(tid_allocator.id_to_label);

    size_t id_allocator_total_size = id_allocator_size + tid_allocator_size;

    size_t num_tree_nodes = 0;
    size_t bloom_filter_total_size = 0;
    size_t short_lists_total_size = 0;
    size_t short_lists_payload_total_size = 0;
    size_t vector_indices_total_size = 0;
    size_t centroid_total_size = 0;
    size_t node_attrs_total_size = 0;

    std::queue<TreeNode*> q;
    q.push(tree_root);

    while (!q.empty()) {
        TreeNode* node = q.front();
        q.pop();

        num_tree_nodes++;
        bloom_filter_total_size += bloom_filter_size(node->bf);
        short_lists_total_size += short_lists_size(node->shortlists);
        short_lists_payload_total_size +=
                short_lists_payload_size(node->shortlists);
        centroid_total_size += sizeof(float) * this->d;
        node_attrs_total_size += node_attrs_size(node);

        if (node->children.empty()) {
            vector_indices_total_size += short_list_size(node->vector_indices);
        } else {
            for (TreeNode* child : node->children) {
                q.push(child);
            }
        }
    }

    size_t total_memory_usage = 0;
    total_memory_usage = sizeof(MultiTenantIndexIVFHierarchical);
    total_memory_usage += id_allocator_total_size;
    total_memory_usage += bloom_filter_total_size;
    total_memory_usage += short_lists_total_size;
    total_memory_usage += vector_indices_total_size;
    total_memory_usage += centroid_total_size;
    total_memory_usage += node_attrs_total_size;

    printf("Memory usage breakdown:\n");
    printf("Number of tree nodes: %lu\n", num_tree_nodes);
    printf("Total memory usage: %lu bytes\n", total_memory_usage);
    printf("ID allocator: %lu bytes\n", id_allocator_total_size);
    printf("Bloom filters: %lu bytes\n", bloom_filter_total_size);
    printf("Short lists: %lu bytes\n", short_lists_total_size);
    printf("Short lists payload: %lu bytes\n", short_lists_payload_total_size);
    printf("Vector indices: %lu bytes\n", vector_indices_total_size);
    printf("Centroids: %lu bytes\n", centroid_total_size);
    printf("Node attributes: %lu bytes\n", node_attrs_total_size);
}

namespace complex_predicate {

// This search algorithm is a modified version of the single-label filtering
// algorithm using search_ef to terminate the search. Beam search is used to
// initialize the search frontier. Variance boost is disabled for simplicity.
// void search_temp_index(
//         const MultiTenantIndexIVFHierarchical* index,
//         const std::vector<int_vid_t>& qualified_vecs,
//         const std::vector<TempIndexNode>& nodes,
//         const float* x,
//         idx_t k,
//         float* distances,
//         idx_t* labels,
//         const SearchParameters* params) {
//     using Candidate = std::pair<float, int>; // (score, node idx in temp index)

//     DEBUG_PRINTF("[DEBUG] search_temp_index: qualified_vecs.size()=%zu, nodes.size()=%zu, k=%ld\n",
//            qualified_vecs.size(),
//            nodes.size(),
//            k);

//     // Return if the temp index is empty
//     if (nodes.empty()) {
//         DEBUG_PRINTF("[DEBUG] search_temp_index: nodes is empty, returning\n");
//         return;
//     }
//     if (index->beam_size == 0) {
//         DEBUG_PRINTF("[DEBUG] search_temp_index: beam_size is 0, throwing error\n");
//         FAISS_THROW_MSG("beam_size must be greater than 0");
//     }

//     DEBUG_PRINTF("[DEBUG] search_temp_index: beam_size=%zu, search_ef=%zu\n",
//            index->beam_size,
//            index->search_ef);

//     /* STAGE 1: BEAM SEARCH */

//     std::vector<Candidate> beam;       // beam of current step
//     std::vector<Candidate> next_beam;  // beam of next step
//     std::vector<Candidate> unexpanded; // unexpanded nodes during beam search

//     // Initialize beam with root node
//     float score = fvec_L2sqr(x, nodes[0].centroid, index->d);
//     beam.emplace_back(score, 0);

//     while (true) {
//         bool updated =
//                 false; // if beam is updated by successfully expanding a node
//         for (auto [score, node_idx] : beam) {
//             const TempIndexNode& node = nodes[node_idx];
//             if (node.children.empty()) {
//                 next_beam.emplace_back(score, node_idx);
//             } else {
//                 updated = true;
//                 for (int child_idx : node.children) {
//                     // Variance boost is disabled for simplicity
//                     float child_score =
//                             fvec_L2sqr(x, nodes[child_idx].centroid, index->d);
//                     next_beam.emplace_back(child_score, child_idx);
//                 }
//             }
//         }

//         // Break if beam is not updated (i.e., next beam = current beam)
//         if (!updated) {
//             break;
//         }

//         // Only expand the top-beam-width nodes in the next step
//         // Move the rest of the nodes to unexpanded list
//         std::sort(next_beam.begin(), next_beam.end());

//         auto n_keep = std::min(index->beam_size, next_beam.size());
//         for (size_t i = n_keep; i < next_beam.size(); i++) {
//             unexpanded.push_back(next_beam[i]);
//         }
//         next_beam.resize(n_keep);

//         std::swap(beam, next_beam);
//         next_beam.clear();
//     }

//     // Check for overlaps between beam and unexpanded
//     DEBUG_PRINTF("[DEBUG] Beam search complete: beam size=%zu, unexpanded size=%zu\n",
//            beam.size(),
//            unexpanded.size());

//     std::set<int> beam_nodes;
//     std::set<int> unexpanded_nodes;

//     for (const auto& [score, node_idx] : beam) {
//         if (beam_nodes.count(node_idx) > 0) {
//             printf("[ERROR] ❌ DUPLICATE IN BEAM: Node %d appears multiple times in beam\n",
//                    node_idx);
//         }
//         beam_nodes.insert(node_idx);
//     }

//     for (const auto& [score, node_idx] : unexpanded) {
//         if (unexpanded_nodes.count(node_idx) > 0) {
//             printf("[ERROR] ❌ DUPLICATE IN UNEXPANDED: Node %d appears multiple times in unexpanded\n",
//                    node_idx);
//         }
//         unexpanded_nodes.insert(node_idx);
//     }

//     // Check for overlap between beam and unexpanded
//     std::set<int> overlap;
//     std::set_intersection(
//             beam_nodes.begin(),
//             beam_nodes.end(),
//             unexpanded_nodes.begin(),
//             unexpanded_nodes.end(),
//             std::inserter(overlap, overlap.begin()));

//     if (!overlap.empty()) {
//         printf("[ERROR] ❌ BEAM/UNEXPANDED OVERLAP: %zu nodes appear in both beam and unexpanded: ",
//                overlap.size());
//         for (int node_idx : overlap) {
//             DEBUG_PRINTF("%d ", node_idx);
//         }
//         DEBUG_PRINTF("\n");
//     } else {
//         DEBUG_PRINTF("[DEBUG] ✅ No overlap between beam and unexpanded\n");
//     }

//     /* STAGE 2: BEST-FIRST SEARCH */

//     // Distance computer to compute distances between query and indexed vectors
//     DistanceComputer* disc = index->storage->get_distance_computer();
//     disc->set_query(x);

//     // Limited size result set. The final top-k candidates will be returned
//     RunningList cand_vectors(index->search_ef);

//     // Temporary array holding vectors in a buffer to be merged into the result
//     // set
//     std::vector<std::pair<float, int_vid_t>> sorted_cands;
//     sorted_cands.reserve(index->max_sl_size);

//     // Initialize search frontier with the frontier from beam search
//     // i.e., nodes in the beam of the last step and all visited-but-not-expanded
//     // nodes
//     MinHeap<Candidate> frontier;
//     std::set<int> nodes_in_frontier; // Track nodes added to frontier

//     for (auto& cand : unexpanded) {
//         frontier.push(cand);
//         int node_idx = cand.second;
//         if (nodes_in_frontier.count(node_idx) > 0) {
//             DEBUG_PRINTF("[DEBUG] ❌ FRONTIER DUPLICATE (unexpanded): Node %d already in frontier\n",
//                    node_idx);
//         }
//         nodes_in_frontier.insert(node_idx);
//     }
//     for (auto& cand : beam) {
//         frontier.push(cand);
//         int node_idx = cand.second;
//         if (nodes_in_frontier.count(node_idx) > 0) {
//             DEBUG_PRINTF("[DEBUG] ❌ FRONTIER DUPLICATE (beam): Node %d already in frontier\n",
//                    node_idx);
//         }
//         nodes_in_frontier.insert(node_idx);
//     }

//     // Track which nodes we've visited and which vector IDs we've processed
//     std::set<int> visited_nodes;
//     std::map<int_vid_t, int> vid_count;

//     DEBUG_PRINTF("[DEBUG] Starting best-first search with frontier size: %zu\n",
//            frontier.size());

//     // Main loop of the search algorithm
//     int loop_count = 0;
//     while (!frontier.empty()) {
//         auto [score, node_idx] = frontier.top();
//         frontier.pop();
//         loop_count++;

//         // Check if we've visited this node before - this should NEVER happen
//         if (visited_nodes.count(node_idx) > 0) {
//             DEBUG_PRINTF("[DEBUG] ❌ BUG: DUPLICATE NODE VISIT: Node %d visited again at loop %d\n",
//                    node_idx,
//                    loop_count);
//             // Continue processing to gather more debug info, but this is a bug
//         }
//         visited_nodes.insert(node_idx);

//         if (loop_count <= 10) { // Print first 10 iterations
//             DEBUG_PRINTF("[DEBUG] Loop %d: processing node %d (score=%.4f)\n",
//                    loop_count,
//                    node_idx,
//                    score);
//         }

//         const TempIndexNode& node = nodes[node_idx];

//         if (loop_count <= 10) {
//             DEBUG_PRINTF("[DEBUG] Node %d range: [%d, %d), children: %zu\n",
//                    node_idx,
//                    node.start,
//                    node.end,
//                    node.children.size());
//         }

//         // Only process vectors if this is a leaf node (no children)
//         if (node.children.empty()) {
//             // Process vectors in current node's range
//             for (int i = node.start; i < node.end; i++) {
//                 int_vid_t vid = qualified_vecs[i];

//                 // Track vector ID processing
//                 vid_count[vid]++;
//                 if (vid_count[vid] > 1) {
//                     printf("[ERROR] ❌ DUPLICATE VID: vid=%d processed %d times (loop %d, node %d, index %d)\n",
//                            vid,
//                            vid_count[vid],
//                            loop_count,
//                            node_idx,
//                            i);
//                 }

//                 ext_vid_t lbl = index->id_allocator.get_label(vid);
//                 float dist = disc->operator()(lbl);
//                 sorted_cands.emplace_back(dist, vid);

//                 if (loop_count <= 5 &&
//                     i < node.start + 3) { // Print first few vectors
//                     DEBUG_PRINTF("[DEBUG] Processing vid=%d -> label=%u, dist=%.4f\n",
//                            vid,
//                            lbl,
//                            dist);
//                 }
//             }

//             if (!sorted_cands.empty()) {
//                 if (loop_count <= 10) {
//                     DEBUG_PRINTF("[DEBUG] Found %zu candidates in this LEAF node\n",
//                            sorted_cands.size());
//                 }

//                 std::sort(sorted_cands.begin(), sorted_cands.end());

//                 // Break if the result set cannot be improved by any vector in
//                 // the buffer
//                 bool updated = cand_vectors.batch_insert(sorted_cands);
//                 sorted_cands.clear();
//                 if (!updated) {
//                     DEBUG_PRINTF("[DEBUG] Candidate list not updated, breaking at loop %d\n",
//                            loop_count);
//                     break;
//                 } else if (loop_count <= 10) {
//                     DEBUG_PRINTF("[DEBUG] Candidate list updated, current size: %zu\n",
//                            cand_vectors.vids.size());
//                 }
//             }
//         } else {
//             if (loop_count <= 10) {
//                 DEBUG_PRINTF("[DEBUG] Node %d is non-leaf, skipping vector processing\n",
//                        node_idx);
//             }

//             // Add child nodes to frontier with duplicate checking
//             for (int child_idx : node.children) {
//                 if (nodes_in_frontier.count(child_idx) > 0) {
//                     printf("[ERROR] ❌ CHILD ALREADY IN FRONTIER: Child node %d already in frontier (parent: %d)\n",
//                         child_idx,
//                         node_idx);
//                 } else if (visited_nodes.count(child_idx) > 0) {
//                     printf("[ERROR] ❌ CHILD ALREADY VISITED: Child node %d already visited (parent: %d)\n",
//                         child_idx,
//                         node_idx);
//                 } else {
//                     float child_score =
//                             fvec_L2sqr(x, nodes[child_idx].centroid, index->d);
//                     frontier.emplace(child_score, child_idx);
//                     nodes_in_frontier.insert(child_idx);
//                 }
//             }
//         }
//     }

//     DEBUG_PRINTF("[DEBUG] Search loop completed after %d iterations, found %zu candidates\n",
//            loop_count,
//            cand_vectors.vids.size());

//     // Print duplicate statistics
//     DEBUG_PRINTF("[DEBUG] Duplicate analysis:\n");
//     DEBUG_PRINTF("[DEBUG] - Total unique nodes visited: %zu\n", visited_nodes.size());

//     int duplicate_vid_count = 0;
//     for (const auto& [vid, count] : vid_count) {
//         if (count > 1) {
//             DEBUG_PRINTF("[DEBUG] - VID %d appeared %d times\n", vid, count);
//             duplicate_vid_count++;
//         }
//     }
//     DEBUG_PRINTF("[DEBUG] - Total VIDs with duplicates: %d\n", duplicate_vid_count);

//     // Extract the top-k candidates directly from the sorted candidate list
//     auto n_results = std::min(static_cast<size_t>(k), cand_vectors.vids.size());

//     // Initialize unused slots with sentinel values
//     for (size_t i = 0; i < k; i++) {
//         distances[i] = std::numeric_limits<float>::max();
//         labels[i] = -1;
//     }

//     // Copy results in sorted order (best distances first)
//     for (size_t i = 0; i < n_results; i++) {
//         labels[i] = index->id_allocator.get_label(cand_vectors.vids[i]);
//         distances[i] = cand_vectors.dists[i];
//     }
// }

void search_temp_index(
        const MultiTenantIndexIVFHierarchical* index,
        const std::vector<int_vid_t>& qualified_vecs,
        const std::vector<TempIndexNode>& nodes,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) {
    
    DEBUG_PRINTF("[DEBUG] search_temp_index (simple): qualified_vecs.size()=%zu, k=%ld\n",
           qualified_vecs.size(),
           k);

    // Return if no qualified vectors
    if (qualified_vecs.empty()) {
        DEBUG_PRINTF("[DEBUG] search_temp_index: no qualified vectors, returning\n");
        // Initialize result arrays with sentinel values
        for (size_t i = 0; i < k; i++) {
            distances[i] = std::numeric_limits<float>::max();
            labels[i] = -1;
        }
        return;
    }

    // Vector to store all (distance, vector_id) pairs
    std::vector<std::pair<float, int_vid_t>> cand_pairs;
    cand_pairs.reserve(qualified_vecs.size());

    // Iterate over all qualified vectors and compute distances using fvec_L2sqr
    for (size_t i = 0; i < qualified_vecs.size(); i++) {
        int_vid_t vid = qualified_vecs[i];
        
        // Get the external label for this vector ID
        ext_vid_t lbl = index->id_allocator.get_label(vid);
        
        // Get the storage index for this vector
        auto storage_idx_it = index->vid_to_storage_idx.find(vid);
        if (storage_idx_it == index->vid_to_storage_idx.end()) {
            DEBUG_PRINTF("[WARNING] Vector ID %llu not found in storage mapping, skipping\n", vid);
            continue;
        }
        idx_t storage_idx = storage_idx_it->second;
        
        // Get pointer to the stored vector data
        const float* stored_vec = index->storage->get_xb() + storage_idx * index->d;
        
        // Compute L2 squared distance directly
        float dist = fvec_L2sqr(x, stored_vec, index->d);
        
        // Store distance and vector ID
        cand_pairs.emplace_back(dist, vid);
        
        if (i < 5) { // Debug: print first few vectors
            DEBUG_PRINTF("[DEBUG] Vector %zu: vid=%llu -> label=%u, storage_idx=%ld, dist=%.4f\n",
                   i, vid, lbl, storage_idx, dist);
        }
    }

    // Sort candidates by distance (ascending order - smallest distance first)
    std::sort(cand_pairs.begin(), cand_pairs.end());

    DEBUG_PRINTF("[DEBUG] Sorted %zu candidates, returning top %ld\n",
           cand_pairs.size(), k);

    // Initialize result arrays with sentinel values
    for (size_t i = 0; i < k; i++) {
        distances[i] = std::numeric_limits<float>::max();
        labels[i] = -1;
    }

    // Copy the top-k results
    size_t n_results = std::min(static_cast<size_t>(k), cand_pairs.size());
    for (size_t i = 0; i < n_results; i++) {
        distances[i] = cand_pairs[i].first;
        labels[i] = index->id_allocator.get_label(cand_pairs[i].second);
        
        if (i < 3) { // Debug: print first few results
            DEBUG_PRINTF("[DEBUG] Result %zu: label=%ld, dist=%.4f\n",
                   i, labels[i], distances[i]);
            printf("[DEBUG] Result %zu: label=%ld, dist=%.4f\n",
                   i, labels[i], distances[i]);
        }
    }
    
    DEBUG_PRINTF("[DEBUG] search_temp_index completed, returned %zu results\n", n_results);
}

} // namespace complex_predicate

void MultiTenantIndexIVFHierarchical::search(
        idx_t n,
        const float* x,
        idx_t k,
        const std::string& filter,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    DEBUG_PRINTF("[DEBUG] Starting complex predicate search with filter: '%s'\n",
           filter.c_str());
    DEBUG_PRINTF("[DEBUG] n=%ld, k=%ld\n", n, k);

    // Initialize timing variables
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_find_label = std::chrono::high_resolution_clock::now();

    // if the filter is already indexed, perform simple filtered search
    // here we only do exact match, but it's possible to perform subexpression
    // matching in the future
    auto filter_label = filter_to_label.find(filter);
    auto end_find_label = std::chrono::high_resolution_clock::now();

    if (filter_label != filter_to_label.end()) {
        DEBUG_PRINTF("[DEBUG] Filter found in cache, using cached search with tid=%u\n",
               filter_label->second);
        ext_lid_t tid = filter_label->second;
        search(n, x, k, tid, distances, labels, params);
        return;
    }

    // Convert the filter to use internal tenant IDs
    auto start_convert = std::chrono::high_resolution_clock::now();
    auto converted_filter = convert_complex_predicate(filter);
    auto end_convert = std::chrono::high_resolution_clock::now();
    DEBUG_PRINTF("[DEBUG] Converted filter: '%s' -> '%s'\n",
           filter.c_str(),
           converted_filter.c_str());

    // Find all qualified vectors
    auto start_find_qual = std::chrono::high_resolution_clock::now();
    auto qualified_vecs = find_all_qualified_vecs(converted_filter);
    auto end_find_qual = std::chrono::high_resolution_clock::now();
    DEBUG_PRINTF("[DEBUG] Found %zu qualified vectors\n", qualified_vecs.size());

    // Write qualified vectors to file for comparison with reference implementation
    {
        static std::mutex file_mutex;
        std::lock_guard<std::mutex> lock(file_mutex);
        
        std::string filename = "/tmp/qualified_vecs.txt";
        std::ofstream outfile(filename);
        
        if (outfile.is_open()) {
            // Convert VIDs to external labels and sort
            std::vector<ext_vid_t> qualified_labels;
            qualified_labels.reserve(qualified_vecs.size());
            
            for (int_vid_t vid : qualified_vecs) {
                ext_vid_t label = id_allocator.get_label(vid);
                qualified_labels.push_back(label);
            }
            
            std::sort(qualified_labels.begin(), qualified_labels.end());
            
            for (ext_vid_t label : qualified_labels) {
                outfile << label << "\n";
            }
            
            outfile.close();
            printf("[INFO] Wrote %zu qualified vector labels to file: %s\n", qualified_labels.size(), filename.c_str());
        } else {
            printf("[WARNING] Failed to open file for writing: %s\n", filename.c_str());
        }
    }

    if (qualified_vecs.empty()) {
        DEBUG_PRINTF("[DEBUG] No qualified vectors found, returning empty results\n");
        heap_heapify<HeapForL2>(k, distances, labels);
        return;
    }

    // Sort qualified vectors' IDs in ascending order
    auto start_sort = std::chrono::high_resolution_clock::now();
    std::sort(qualified_vecs.begin(), qualified_vecs.end());
    auto end_sort = std::chrono::high_resolution_clock::now();
    DEBUG_PRINTF("[DEBUG] Sorted qualified vectors. First 10 vids: ");
    for (size_t i = 0; i < std::min(qualified_vecs.size(), (size_t)10); i++) {
        DEBUG_PRINTF("%d ", qualified_vecs[i]);
    }
    DEBUG_PRINTF("\n");

    // Build temporary index
    auto start_build = std::chrono::high_resolution_clock::now();
    std::vector<complex_predicate::TempIndexNode> nodes;
    complex_predicate::build_temp_index_for_filter(this, qualified_vecs, nodes);
    auto end_build = std::chrono::high_resolution_clock::now();
    DEBUG_PRINTF("[DEBUG] Built temporary index with %zu nodes\n", nodes.size());

    // Search the temporary index
    auto start_search = std::chrono::high_resolution_clock::now();
    for (idx_t i = 0; i < n; i++) {
        DEBUG_PRINTF("[DEBUG] Searching query %ld...\n", i);
        complex_predicate::search_temp_index(
                this,
                qualified_vecs,
                nodes,
                x + i * d,
                k,
                distances + i * k,
                labels + i * k,
                params);

        // Print results for this query
        DEBUG_PRINTF("[DEBUG] Results for query %ld: ", i);
        for (idx_t j = 0; j < k; j++) {
            DEBUG_PRINTF("(id=%ld, dist=%.4f) ",
                   labels[i * k + j],
                   distances[i * k + j]);
        }
        DEBUG_PRINTF("\n");
    }
    auto end_search = std::chrono::high_resolution_clock::now();
    auto end_total = std::chrono::high_resolution_clock::now();

    // Calculate latencies in microseconds
    auto find_label_latency =
            std::chrono::duration_cast<std::chrono::microseconds>(
                    end_find_label - start_find_label)
                    .count();
    auto convert_latency =
            std::chrono::duration_cast<std::chrono::microseconds>(
                    end_convert - start_convert)
                    .count();
    auto find_qual_latency =
            std::chrono::duration_cast<std::chrono::microseconds>(
                    end_find_qual - start_find_qual)
                    .count();
    auto sort_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                                end_sort - start_sort)
                                .count();
    auto build_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                                 end_build - start_build)
                                 .count();
    auto search_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                                  end_search - start_search)
                                  .count();
    auto total_latency = std::chrono::duration_cast<std::chrono::microseconds>(
                                 end_total - start_total)
                                 .count();

    // Write to CSV file
    static std::mutex csv_mutex;
    std::lock_guard<std::mutex> lock(csv_mutex);

    static bool file_initialized = false;
    static std::ofstream csv_file("/tmp/search_latency.csv", std::ios::app);

    if (!file_initialized) {
        csv_file
                << "filter,find_label_latency,convert_latency,find_qual_latency,sort_latency,build_latency,search_latency,total_latency\n";
        file_initialized = true;
    }

    // Escape any commas in the filter string
    std::string escaped_filter = filter;
    size_t pos = 0;
    while ((pos = escaped_filter.find(",", pos)) != std::string::npos) {
        escaped_filter.replace(pos, 1, "\\,");
        pos += 2;
    }

    csv_file << escaped_filter << "," << find_label_latency << ","
             << convert_latency << "," << find_qual_latency << ","
             << sort_latency << "," << build_latency << "," << search_latency
             << "," << total_latency << "\n";
    csv_file.flush();
}

} // namespace faiss