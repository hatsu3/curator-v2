// -*- c++ -*-

#include <faiss/MultiTenantIndexIVFHierarchical.h>

#include <omp.h>
#include <cinttypes>
#include <cstdio>
#include <deque>
#include <queue>

#include <faiss/Clustering.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/prefetch.h>
#include <faiss/utils/utils.h>

namespace faiss {

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
        Index* quantizer,
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
        size_t beam_size,
        bool approx_dists,
        size_t search_frontier_capacity,
        bool two_stage)
        : MultiTenantIndexIVFFlat(quantizer, d, n_clusters, metric),
          n_clusters(n_clusters),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos),
          max_sl_size(max_sl_size),
          vec_store(d),
          clus_niter(clus_niter),
          max_leaf_size(max_leaf_size),
          nprobe(nprobe),
          prune_thres(prune_thres),
          variance_boost(variance_boost),
          search_ef(search_ef),
          beam_size(beam_size),
          approx_dists(approx_dists),
          search_frontier_capacity(search_frontier_capacity),
          two_stage(two_stage) {
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
        vec_store.add_vector(xi, vid);
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
    int_vid_t vid = id_allocator.get_id(label);
    auto leaf = find_assigned_leaf(label);

    // update the variance of the tree nodes along the path
    TreeNode* curr = leaf;
    auto vec = vec_store.get_vec(vid);
    while (curr != nullptr) {
        float dist = fvec_L2sqr(vec, curr->centroid, d);
        curr->variance.remove(dist);
        curr = curr->parent;
    }

    id_allocator.remove_mapping(label);
    vec_store.remove_vector(vid);
    leaf->vector_indices.erase(vid);

    return true;
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
        const std::string& filter,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    // if the filter is already indexed, perform simple filtered search
    // here we only do exact match, but it's possible to perform subexpression
    // matching in the future
    auto filter_label = filter_to_label.find(filter);
    if (filter_label != filter_to_label.end()) {
        ext_lid_t tid = filter_label->second;
        search(n, x, k, tid, distances, labels, params);
        return;
    }

    auto converted_filter = convert_complex_predicate(filter);

    for (idx_t i = 0; i < n; i++) {
        search_one(
                x + i * d,
                k,
                converted_filter,
                distances + i * k,
                labels + i * k,
                params);
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

struct SearchFrontier {
    std::multiset<Candidate> nodes;
    int capacity;

    SearchFrontier(int capacity) : capacity(capacity) {}

    bool empty() {
        return nodes.empty();
    }

    void push(const Candidate& cand) {
        auto [score, node] = cand;
        if (nodes.size() < capacity) {
            nodes.emplace(score, node);
        } else {
            auto back_it = std::prev(nodes.end());
            if (score < back_it->first) {
                nodes.erase(back_it);
                nodes.emplace(score, node);
            }
        }
    }

    void emplace(float score, const TreeNode* node) {
        push({score, node});
    }

    Candidate top() {
        return *nodes.begin();
    }

    void pop() {
        if (!nodes.empty()) {
            nodes.erase(nodes.begin());
        }
    }

    void reset() {
        nodes.clear();
    }
};

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

inline void compute_approx_dists_with_prefetch(
        const TreeNode* node,
        const std::vector<int_vid_t>& buffer,
        const float* x,
        int d,
        std::vector<float>& child_dists,
        std::vector<std::pair<float, int_vid_t>>& output) {
    FAISS_ASSERT_MSG(
            !node->children.empty(),
            "calc_approx_dists should only be called on non-leaf nodes");

    child_dists.resize(node->children.size());

    auto centroid = node->children[0]->centroid;
    for (size_t i = 0; i < node->children.size() - 1; i++) {
        auto next_centroid = node->children[i + 1]->centroid;
        prefetch_L1(next_centroid);
        child_dists[i] = fvec_L2sqr(x, centroid, d);
        centroid = next_centroid;
    }
    child_dists.back() = fvec_L2sqr(x, centroid, d);

    auto offset = sizeof(int_vid_t) * 8 -
            (node->level + 1) * CURATOR_MAX_BRANCH_FACTOR_LOG2;
    for (auto vid : buffer) {
        auto child_id = (vid >> offset) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        output.emplace_back(child_dists[child_id], vid);
    }
}

inline void compute_dists_with_prefetch(
        const MultiTenantIndexIVFHierarchical& index,
        const std::vector<int_vid_t>& vids,
        const float* x,
        std::vector<std::pair<float, int_vid_t>>& output) {
    if (vids.empty()) {
        return;
    }

    auto vec = index.vec_store.get_vec(vids[0]);
    for (size_t i = 0; i < vids.size() - 1; i++) {
        auto next_vec = index.vec_store.get_vec(vids[i + 1]);
        prefetch_L1(next_vec);
        auto dist = fvec_L2sqr(x, vec, index.d);
        output.emplace_back(dist, vids[i]);
        vec = next_vec;
    }

    auto dist = fvec_L2sqr(x, vec, index.d);
    output.emplace_back(dist, vids.back());
}

inline void compute_child_scores_with_prefetch(
        const MultiTenantIndexIVFHierarchical& index,
        const TreeNode* node,
        const float* x,
        SearchFrontier& output) {
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
        int beam_size = 0,
        bool approx_dists = false,
        int search_frontier_capacity = 16) {
    SearchFrontier frontier(search_frontier_capacity);

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
    child_dists.reserve(index.nlist);

    while (!frontier.empty()) {
        auto [score, node] = frontier.top();
        frontier.pop();

        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            auto& buffer = it->second.data;
            if (approx_dists && !node->children.empty()) {
                compute_approx_dists_with_prefetch(
                        node, buffer, x, index.d, child_dists, sorted_cands);
            } else {
                compute_dists_with_prefetch(index, buffer, x, sorted_cands);
            }
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
    if (approx_dists) {
        sorted_cands.reserve(cand_vectors.vids.size());
        compute_dists_with_prefetch(index, cand_vectors.vids, x, sorted_cands);
        std::sort(sorted_cands.begin(), sorted_cands.end());

        for (size_t i = 0; i < n_results; i++) {
            labels[i] = index.id_allocator.get_label(sorted_cands[i].second);
            distances[i] = sorted_cands[i].first;
        }
    } else {
        for (size_t i = 0; i < n_results; i++) {
            labels[i] = index.id_allocator.get_label(cand_vectors.vids[i]);
            distances[i] = cand_vectors.dists[i];
        }
    }
}

void filtered_search_experimental_two_stage(
        const MultiTenantIndexIVFHierarchical& index,
        const float* x,
        idx_t k,
        int_lid_t tid,
        float* distances,
        idx_t* labels,
        int search_ef,
        int beam_size = 0,
        int search_frontier_capacity = 16) {
    SearchFrontier frontier(search_frontier_capacity);

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

    std::vector<Candidate> buckets;

    while (!frontier.empty()) {
        auto [score, node] = frontier.top();
        frontier.pop();

        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            buckets.emplace_back(score, node);
        } else if (node->bf.contains(tid)) {
            compute_child_scores_with_prefetch(index, node, x, frontier);
        }
    }

    std::sort(buckets.begin(), buckets.end());

    RunningList cand_vectors(search_ef);
    std::vector<std::pair<float, int_vid_t>> sorted_cands;
    sorted_cands.reserve(index.max_sl_size);
    std::vector<float> child_dists;
    child_dists.reserve(index.nlist);

    for (auto [score, node] : buckets) {
        auto& buffer = node->shortlists.at(tid).data;
        compute_dists_with_prefetch(index, buffer, x, sorted_cands);
        std::sort(sorted_cands.begin(), sorted_cands.end());
        bool updated = cand_vectors.batch_insert(sorted_cands);
        sorted_cands.clear();
        if (!updated) {
            break;
        }
    }

    heap_heapify<HeapForL2>(k, distances, labels);
    auto n_results = std::min(static_cast<size_t>(k), cand_vectors.vids.size());
    for (size_t i = 0; i < n_results; i++) {
        labels[i] = index.id_allocator.get_label(cand_vectors.vids[i]);
        distances[i] = cand_vectors.dists[i];
    }
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
        if (two_stage) {
            filtered_search_experimental_two_stage(
                    *this,
                    x,
                    k,
                    tid,
                    distances,
                    labels,
                    search_ef,
                    beam_size,
                    search_frontier_capacity);
            return;
        } else {
            filtered_search_experimental(
                    *this,
                    x,
                    k,
                    tid,
                    distances,
                    labels,
                    search_ef,
                    beam_size,
                    approx_dists,
                    search_frontier_capacity);
            return;
        }
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

        float dist = fvec_L2sqr(x, node->centroid, d);
        float var = node->variance.get_mean();
        float score = dist - this->variance_boost * var;
        return score;
    };

    MinHeap<Candidate> pq;
    pq.emplace(node_priority(tree_root), tree_root);

    size_t n_cand_vecs = 0;
    std::vector<Candidate> buckets;

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
        if (dist > this->prune_thres * min_buck_dist) {
            break;
        }

        n_bucks_pruned++;
        for (auto vid : node->shortlists.at(tid)) {
            n_dists++;
            n_vecs_visited++;
            auto vec = vec_store.get_vec(vid);
            auto lbl = id_allocator.get_label(vid);
            auto dist = fvec_L2sqr(x, vec, d);
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
            auto label = id_allocator.get_label(vid);
            auto vec = vec_store.get_vec(vid);
            auto dist = fvec_L2sqr(x, vec, d);
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
            auto vec = vec_store.get_vec(vid);
            auto lbl = id_allocator.get_label(vid);
            auto dist = fvec_L2sqr(x, vec, d);
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
                printf("False positive in bloom filter\n");
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

    size_t vec_store_total_size = unordered_map_memory_usage(vec_store.vecs);
    size_t aligned_vec_size =
            (sizeof(float) * vec_store.d + vec_store.alignment - 1) /
            vec_store.alignment * vec_store.alignment;
    vec_store_total_size += aligned_vec_size * vec_store.vecs.size();

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
    total_memory_usage += vec_store_total_size;
    total_memory_usage += id_allocator_total_size;
    total_memory_usage += bloom_filter_total_size;
    total_memory_usage += short_lists_total_size;
    total_memory_usage += vector_indices_total_size;
    total_memory_usage += centroid_total_size;
    total_memory_usage += node_attrs_total_size;

    printf("Memory usage breakdown:\n");
    printf("Number of tree nodes: %lu\n", num_tree_nodes);
    printf("Total memory usage: %lu bytes\n", total_memory_usage);
    printf("Vector store: %lu bytes\n", vec_store_total_size);
    printf("ID allocator: %lu bytes\n", id_allocator_total_size);
    printf("Bloom filters: %lu bytes\n", bloom_filter_total_size);
    printf("Short lists: %lu bytes\n", short_lists_total_size);
    printf("Short lists payload: %lu bytes\n", short_lists_payload_total_size);
    printf("Vector indices: %lu bytes\n", vector_indices_total_size);
    printf("Centroids: %lu bytes\n", centroid_total_size);
    printf("Node attributes: %lu bytes\n", node_attrs_total_size);
}

} // namespace faiss