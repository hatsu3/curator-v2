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
const tid_t TenantIdAllocator::INVALID_ID = -1;

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
          quantizer(d),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos) {
    if (parent != nullptr) {
        auto offset =
                sizeof(vid_t) * 8 - level * CURATOR_MAX_BRANCH_FACTOR_LOG2;
        this->node_id = parent->node_id | (sibling_id << offset);
    } else {
        this->node_id = 0;
    }

    if (centroid == nullptr) {
        this->centroid = nullptr;
    } else {
        this->centroid = new float[d];
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
        float variance_boost)
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
          variance_boost(variance_boost) {
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
        tid_t tid) {
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
    node->quantizer.reset();
    ClusteringParameters cp;
    cp.niter = clus_niter;
    cp.min_points_per_centroid = 1;
    Clustering clus(d, n_clusters, cp);
    clus.train(n, x, node->quantizer);
    node->quantizer.is_trained = true;

    std::vector<idx_t> cluster_ids(n);
    node->quantizer.assign(n, x, cluster_ids.data());

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
        label_t label = labels[i];
        const float* xi = x + i * d;

        // add the vector to the leaf node
        TreeNode* leaf = assign_vec_to_leaf(xi);
        auto offset = sizeof(vid_t) * 8 -
                leaf->level * CURATOR_MAX_BRANCH_FACTOR_LOG2 -
                CURATOR_MAX_LEAF_SIZE_LOG2;
        auto local_vid = static_cast<vid_t>(leaf->vector_indices.size());
        auto vid = leaf->node_id | (local_vid << offset);

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

void MultiTenantIndexIVFHierarchical::grant_access(idx_t label, tid_t tid) {
    vid_t vid = id_allocator.get_id(label);
    tid = tid_allocator.get_or_create_id(tid);
    grant_access_helper(tree_root, vid, tid);
}

void MultiTenantIndexIVFHierarchical::grant_access_helper(
        TreeNode* node,
        vid_t vid,
        tid_t tid) {
    if (node->children.empty()) {
        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            it->second.insert(vid);
        } else {
            node->shortlists.emplace(tid, std::vector<vid_t>{vid});
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
            node->shortlists.emplace(tid, std::vector<vid_t>{vid});
        } else {
            auto offset = sizeof(vid_t) * 8 -
                    (node->level + 1) * CURATOR_MAX_BRANCH_FACTOR_LOG2;
            auto child_id = (vid >> offset) & (CURATOR_MAX_BRANCH_FACTOR - 1);
            grant_access_helper(node->children[child_id], vid, tid);
        }

        node->bf.insert(tid);
    }
}

bool MultiTenantIndexIVFHierarchical::remove_vector(idx_t label) {
    vid_t vid = id_allocator.get_id(label);
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

bool MultiTenantIndexIVFHierarchical::revoke_access(idx_t label, tid_t tid) {
    tid = tid_allocator.get_id(tid);
    vid_t vid = id_allocator.get_id(label);
    auto leaf = find_assigned_leaf(label);

    // phase 1: find the node that contain shortlist
    auto curr = leaf;
    while (curr != nullptr) {
        if (curr->shortlists.find(tid) != curr->shortlists.end()) {
            break;
        }
        curr = curr->parent;
    }

    FAISS_THROW_IF_NOT_MSG(
            curr != nullptr,
            "Cannot find the node that contains the shortlist of the tenant");

    // phase 2: remove the vector from the shortlist
    auto& shortlist = curr->shortlists.at(tid);
    shortlist.erase(vid);
    if (shortlist.size() == 0) {
        curr->shortlists.erase(tid);
        curr->bf = curr->recompute_bloom_filter();
    }

    // phase 3: recursively merge shortlists and update bloom filters
    while (curr && merge_short_list(curr, tid)) {
        curr = curr->parent;
    }

    return true;
}

void MultiTenantIndexIVFHierarchical::search(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (tid < 0) {
        // perform unfiltered search
        search(n, x, k, distances, labels, params);
        return;
    }

    tid = tid_allocator.get_id(tid);

    bool inter_query_parallel = getenv("BATCH_QUERY") != nullptr;
    if (inter_query_parallel) {
#pragma omp parallel for schedule(dynamic) if (n > 1)
        for (idx_t i = 0; i < n; i++) {
            search_one(
                    x + i * d,
                    k,
                    tid,
                    distances + i * k,
                    labels + i * k,
                    params);
        }
    } else {
        for (idx_t i = 0; i < n; i++) {
            search_one(
                    x + i * d,
                    k,
                    tid,
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
        auto tid = filter_label->second;
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

void MultiTenantIndexIVFHierarchical::search_one(
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    using Candidate = std::pair<float, const TreeNode*>;

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
            tid_t tid = std::stoi(var);
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
        auto shift = sizeof(vid_t) * 8 -
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
    idx_t cluster_id;

    while (!curr->children.empty()) {
        curr->quantizer.assign(1, x, &cluster_id);
        curr = curr->children[cluster_id];
    }
    return curr;
}

std::vector<idx_t> MultiTenantIndexIVFHierarchical::get_vector_path(
        label_t label) const {
    std::vector<idx_t> path;
    auto vid = id_allocator.get_id(label);
    auto curr = tree_root;
    auto shift = sizeof(vid_t) * 8 - CURATOR_MAX_BRANCH_FACTOR_LOG2;
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
        tid_t tid) {
    if (node->children.empty() ||
        node->shortlists.at(tid).size() <= max_sl_size) {
        return;
    }

    std::vector<std::vector<vid_t>> child_sls(node->children.size());
    for (vid_t vid : node->shortlists.at(tid)) {
        auto offset = sizeof(vid_t) * 8 -
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
        tid_t tid) {
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

void MultiTenantIndexIVFHierarchical::locate_vector(label_t label) const {
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
            auto tenant_id = tid_allocator.get_id(std::stol(token));
            token = std::to_string(tenant_id);
        }

        converted_filter = converted_filter + token;
        if (i < tokens.size() - 1) {
            converted_filter = converted_filter + " ";
        }
    }

    return converted_filter;
}

std::vector<vid_t> MultiTenantIndexIVFHierarchical::find_all_qualified_vecs(
        const std::string& filter) const {
    using namespace complex_predicate;
    using Candidate = std::tuple<const TreeNode*, VarMap, State>;

    auto update_var_map = [&](const TreeNode* node, VarMap var_map) -> VarMap {
        auto new_var_map = std::unordered_map<std::string, State>();

        for (const auto& var : var_map->unresolved_vars()) {
            tid_t tid = std::stoi(var);
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
        auto shift = sizeof(vid_t) * 8 -
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

    std::vector<vid_t> qual_vecs;
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
        const std::vector<vid_t>& vids,
        tid_t tid) {
    tid = tid_allocator.get_or_create_id(tid);
    tree_root->shortlists.emplace(tid, vids);
    tree_root->bf.insert(tid);
    split_short_list(tree_root, tid);
}

void MultiTenantIndexIVFHierarchical::build_index_for_filter(
        const std::string& filter) {
    auto converted_filter = convert_complex_predicate(filter);
    auto qualified_vecs = find_all_qualified_vecs(converted_filter);
    if (qualified_vecs.empty()) {
        return;
    }

    auto reserved_label = tid_allocator.allocate_reserved_label();
    auto reserved_tid = tid_allocator.allocate_id(reserved_label);
    filter_to_label.emplace(filter, reserved_label);
    batch_grant_access(qualified_vecs, reserved_tid);
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

std::pair<bool, std::set<tid_t>> check_shortlists(
        const MultiTenantIndexIVFHierarchical& index,
        const TreeNode& node) {
    // recursively check the shortlists in the descendant nodes

    std::set<tid_t> shortlists_in_desc;
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
        std::set<tid_t> all_tenants;
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

    std::set<tid_t> shortlists_in_node;
    for (const auto& [tenant, shortlist] : node.shortlists) {
        shortlists_in_node.insert(tenant);
    }

    std::set<tid_t> intersect;
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

    std::set<tid_t> all_shortlists;
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
        label_t label) const {
    auto vid = id_allocator.get_id(label);
    auto curr = tree_root;
    auto shift = sizeof(vid_t) * 8 - CURATOR_MAX_BRANCH_FACTOR_LOG2;
    while (!curr->children.empty()) {
        auto child_id = (vid >> shift) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        curr = curr->children[child_id];
        shift -= CURATOR_MAX_BRANCH_FACTOR_LOG2;
    }
    return curr;
}

} // namespace faiss