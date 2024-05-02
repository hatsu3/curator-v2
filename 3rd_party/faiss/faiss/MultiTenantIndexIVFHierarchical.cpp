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

vid_t IdAllocator::allocate_id(label_t label) {
    // return the existing id if the label already exists
    auto it = label_to_id.find(label);
    if (it != label_to_id.end()) {
        return it->second;
    }

    vid_t id;
    if (free_list.empty()) {
        id = id_to_label.size();
        id_to_label.push_back(-1);
    } else {
        id = *free_list.begin();
        free_list.erase(free_list.begin());
    }

    label_to_id.emplace(label, id);
    id_to_label[id] = label;
    return id;
}

void IdAllocator::free_id(label_t label) {
    auto it = label_to_id.find(label);

    // silently ignore if the label does not exist
    if (it == label_to_id.end()) {
        return;
    }

    label_to_id.erase(label);

    vid_t id = it->second;
    if (id == id_to_label.size() - 1) {
        id_to_label.pop_back();
    } else {
        id_to_label[id] = -1;
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
        : level(level), sibling_id(sibling_id), parent(parent), quantizer(d) {
    if (centroid == nullptr) {
        this->centroid = nullptr;
    } else {
        this->centroid = new float[d];
        std::memcpy(this->centroid, centroid, sizeof(float) * d);
    }

    bloom_parameters bf_params;
    bf_params.projected_element_count = bf_capacity;
    bf_params.false_positive_probability = bf_false_pos;
    bf_params.random_seed = 0xA5A5A5A5;
    FAISS_THROW_IF_NOT_MSG(bf_params, "Invalid bloom filter parameters");
    bf_params.compute_optimal_parameters();
    bf = bloom_filter(bf_params);
}

TreeNode::~TreeNode() {
    delete[] centroid;
}

MultiTenantIndexIVFHierarchical::MultiTenantIndexIVFHierarchical(
        Index* quantizer,
        size_t d,
        size_t n_clusters,
        MetricType metric,
        size_t bf_capacity,
        float bf_false_pos,
        size_t max_sl_size,
        size_t update_bf_interval,
        size_t clus_niter,
        size_t max_leaf_size,
        size_t nprobe,
        float prune_thres)
        : MultiTenantIndexIVFFlat(quantizer, d, n_clusters, metric),
          tree_root(0, 0, nullptr, nullptr, d, bf_capacity, bf_false_pos),
          n_clusters(n_clusters),
          bf_capacity(bf_capacity),
          bf_false_pos(bf_false_pos),
          max_sl_size(max_sl_size),
          update_bf_interval(update_bf_interval),
          update_bf_after(update_bf_interval),
          vec_store(d),
          clus_niter(clus_niter),
          max_leaf_size(max_leaf_size),
          nprobe(nprobe),
          prune_thres(prune_thres) {}

void MultiTenantIndexIVFHierarchical::train(
        idx_t n,
        const float* x,
        tid_t tid) {
    train_helper(&tree_root, n, x);
}

void MultiTenantIndexIVFHierarchical::train_helper(
        TreeNode* node,
        idx_t n,
        const float* x) {
    // stop if there are too few samples to cluster
    if (n <= max_leaf_size || node->level >= 8) {
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
        const idx_t* labels,
        tid_t tid) {
    for (size_t i = 0; i < n; i++) {
        label_t label = labels[i];
        const float* xi = x + i * d;

        // add the vector to the vector store and access matrix
        vid_t vid = id_allocator.allocate_id(label);
        vec_store.add_vector(xi, vid);
        access_matrix.add_vector(vid, tid);
        vector_owners.emplace(label, tid);

        // add the vector to the leaf node
        TreeNode* leaf = assign_vec_to_leaf(xi);
        label_to_leaf.emplace(label, leaf);
        leaf->n_vectors_per_tenant[tid]++;
        leaf->vector_indices.push_back(vid);

        // grant access to the creator (update bloom filter and short lists)
        std::vector<idx_t> path = get_vector_path(label);
        grant_access_helper(&tree_root, label, tid, path);
    }
}

void MultiTenantIndexIVFHierarchical::grant_access(idx_t label, tid_t tid) {
    // update the access information in the leaf node
    TreeNode* leaf = label_to_leaf.at(label);
    leaf->n_vectors_per_tenant[tid]++;

    // update access information in access matrix
    vid_t vid = id_allocator.get_id(label);
    access_matrix.grant_access(vid, tid);

    // grant access to the tenant (update bloom filter and short lists)
    std::vector<idx_t> path = get_vector_path(label);
    grant_access_helper(&tree_root, label, tid, path);
}

void MultiTenantIndexIVFHierarchical::grant_access_helper(
        TreeNode* node,
        label_t label,
        tid_t tid,
        std::vector<idx_t>& path) {
    vid_t vid = id_allocator.get_id(label);
    if (node->children.empty()) {
        // if this is a leaf node, directly insert the data point
        // We maintain inverted list for some tenants to speed up scanning

        auto it = node->shortlists.find(tid);
        // case 1. this tenant has already been shortlisted
        if (it != node->shortlists.end()) {
            it->second.push_back(vid);
            // if (it->second.size() > max_sl_size) {
            //     node->shortlists.erase(it);
            // }
            // case 2. this is the first vector of the tenant
        } else if (!node->bf.contains(tid)) {
            node->shortlists.emplace(tid, std::vector<vid_t>{vid});
        }

        node->bf.insert(tid);

    } else {
        // If this is not a leaf node, for each tenant, we check if the
        // new vector will be saved in a short list. If not, we insert it
        // into the closest child node.

        auto it = node->shortlists.find(tid);
        // case 1. this tenant has already been shortlisted
        if (it != node->shortlists.end()) {
            it->second.push_back(vid);
            if (it->second.size() > max_sl_size) {
                split_short_list(node, tid);
            }
            // case 2. this is the first vector of the tenant
        } else if (!node->bf.contains(tid)) {
            node->shortlists.emplace(tid, std::vector<vid_t>{vid});
            // case 3. this tenant cannot be shortlisted
        } else {
            idx_t cluster_id = path[node->level];
            grant_access_helper(node->children[cluster_id], label, tid, path);
        }

        node->bf.insert(tid);
    }
}

bool MultiTenantIndexIVFHierarchical::remove_vector(idx_t label, tid_t tid) {
    // check if the vector is accessible by the tenant
    if (!vector_owners.at(label) == tid) {
        return false;
    }

    vid_t vid = id_allocator.get_id(label);

    // update access information in the leaf node
    TreeNode* leaf = label_to_leaf.at(label);
    bool should_update_bf = false;
    auto& tenants = access_matrix.access_matrix[vid];
    for (tid_t tid : tenants) {
        if (--leaf->n_vectors_per_tenant.at(tid) == 0) {
            leaf->n_vectors_per_tenant.erase(tid);
            should_update_bf = true;
        }
    }

    // update bloom filters
    if (should_update_bf) {
        update_bf_helper(leaf);
    }

    // update shortlists
    update_shortlists_helper(leaf, vid, tenants);

    // remove the vector from the vector store and access matrix
    id_allocator.free_id(label);
    vec_store.remove_vector(vid);
    access_matrix.remove_vector(vid, tid);
    vector_owners.erase(label);

    // remove the vector from the leaf node
    label_to_leaf.erase(label);
    auto it = std::find(
            leaf->vector_indices.begin(), leaf->vector_indices.end(), vid);
    FAISS_THROW_IF_NOT_MSG(
            it != leaf->vector_indices.end(), "vector not found in leaf node");
    leaf->vector_indices.erase(it);

    return true;
}

bool MultiTenantIndexIVFHierarchical::revoke_access(idx_t label, tid_t tid) {
    // update the access information in access matrix
    vid_t vid = id_allocator.get_id(label);
    access_matrix.revoke_access(vid, tid);

    // update the access information in the leaf node
    TreeNode* leaf = label_to_leaf.at(label);
    bool should_update_bf = false;
    if (--leaf->n_vectors_per_tenant.at(tid) == 0) {
        leaf->n_vectors_per_tenant.erase(tid);
        should_update_bf = true;
    }

    // update bloom filters
    if (should_update_bf) {
        update_bf_helper(leaf);
    }

    // update shortlists
    std::vector<tid_t> tenants{tid};
    update_shortlists_helper(leaf, vid, tenants);

    return true;
}

void MultiTenantIndexIVFHierarchical::update_shortlists_helper(
        TreeNode* leaf,
        vid_t vid,
        std::vector<tid_t>& tenants) {
    for (TreeNode* curr = leaf; curr != nullptr; curr = curr->parent) {
        std::vector<tid_t> intersection;

        if (curr->shortlists.size() < tenants.size()) {
            for (const auto& pair : curr->shortlists) {
                if (std::find(tenants.begin(), tenants.end(), pair.first) !=
                    tenants.end()) {
                    intersection.push_back(pair.first);
                }
            }
        } else {
            for (const int& elem : tenants) {
                if (curr->shortlists.find(elem) != curr->shortlists.end()) {
                    intersection.push_back(elem);
                }
            }
        }

        for (tid_t tenant : intersection) {
            auto& shortlist = curr->shortlists.at(tenant);
            auto vec_it = std::find(shortlist.begin(), shortlist.end(), vid);

            // this shortlist is generated by recursive merging
            if (vec_it == shortlist.end())
                break;

            *vec_it = shortlist.back();
            shortlist.pop_back();

            if (shortlist.empty()) {
                curr->shortlists.erase(tenant);
            } else {
                merge_short_list_recursively(curr, tenant);
            }
        }
    }
}

void MultiTenantIndexIVFHierarchical::update_bf_helper(TreeNode* leaf) {
    if (update_bf_after == 0) {
        leaf->bf.clear();
        for (auto it : leaf->n_vectors_per_tenant) {
            leaf->bf.insert(it.first);
        }
        for (TreeNode* curr = leaf->parent; curr != nullptr;
             curr = curr->parent) {
            curr->bf.clear();
            for (TreeNode* child : curr->children) {
                if (child != nullptr) {
                    curr->bf |= child->bf;
                }
            }
        }
    }

    if (update_bf_after == 0) {
        update_bf_after = update_bf_interval;
    } else {
        update_bf_after--;
    }
}

void MultiTenantIndexIVFHierarchical::search(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
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

using CandBucket =
        std::tuple<float /*dist*/, const TreeNode* /*node*/, size_t /*nvecs*/>;
using Candidate = std::pair<float /*dist*/, const TreeNode* /*node*/>;
using MinHeap = std::priority_queue<
        Candidate,
        std::vector<Candidate>,
        std::greater<Candidate>>;
using HeapForL2 = CMax<float, idx_t>;

void MultiTenantIndexIVFHierarchical::search_one(
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    MinHeap pq;
    pq.emplace(0.0, &tree_root);

    std::vector<CandBucket> bucket_dists;

    while (!pq.empty() && bucket_dists.size() < this->nprobe) {
        auto [dist, node] = pq.top();
        pq.pop();

        auto it = node->shortlists.find(tid);
        if (it != node->shortlists.end()) {
            bucket_dists.emplace_back(dist, node, it->second.size());
            continue;
        }

        if (!node->bf.contains(tid)) {
            continue;
        }

        if (!node->children.empty()) {
            for (auto child : node->children) {
                float dist = fvec_L2sqr(x, child->centroid, d);
                pq.emplace(dist, child);
            }
            continue;
        }

        // We should only reach here due to false positives in the bloom filter
    }

    std::sort(bucket_dists.begin(), bucket_dists.end());

    heap_heapify<HeapForL2>(k, distances, labels);

    if (bucket_dists.empty()) {
        return;
    }

    float min_buck_dist = std::get<0>(bucket_dists[0]);

    for (auto [dist, node, nvecs] : bucket_dists) {
        if (dist > this->prune_thres * min_buck_dist) {
            break;
        }

        for (auto vid : node->shortlists.at(tid)) {
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
    TreeNode* curr = &tree_root;
    idx_t cluster_id;

    while (!curr->children.empty()) {
        curr->quantizer.assign(1, x, &cluster_id);
        curr = curr->children[cluster_id];
    }
    return curr;
}

std::vector<idx_t> MultiTenantIndexIVFHierarchical::get_vector_path(
        label_t label) const {
    TreeNode* leaf = label_to_leaf.at(label);
    std::vector<idx_t> path(leaf->level);

    for (int i = leaf->level - 1; i >= 0; i--) {
        path[i] = leaf->sibling_id;
        leaf = leaf->parent;
    }

    return path;
}

void MultiTenantIndexIVFHierarchical::split_short_list(
        TreeNode* node,
        tid_t tid) {
    for (vid_t vid : node->shortlists.at(tid)) {
        label_t label = id_allocator.get_label(vid);
        std::vector<idx_t> path = get_vector_path(label);
        TreeNode* child = node->children[path[node->level]];
        grant_access_helper(child, label, tid, path);
    }
    node->shortlists.erase(tid);
}

bool MultiTenantIndexIVFHierarchical::merge_short_list(
        TreeNode* node,
        tid_t tid) {
    /* We perform merging only when all sibling nodes have short lists
     * for the tenant and the total size of the short lists is smaller
     * than the threshold.
     */
    if (node->shortlists.at(tid).size() > max_sl_size) {
        return false;
    }

    if (node->parent == nullptr) {
        return false;
    }

    size_t total_sl_size = 0;
    for (TreeNode* sibling : node->parent->children) {
        if (sibling == nullptr) {
            continue;
        } else if (sibling->shortlists.find(tid) == sibling->shortlists.end()) {
            return false;
        } else {
            total_sl_size += sibling->shortlists.at(tid).size();
        }
    }

    if (total_sl_size > max_sl_size) {
        return false;
    }

    /* During the merging process, we combine all short lists from sibling
     * nodes and add the combined short list to the parent node. We also
     * remove the short lists from sibling nodes.
     */
    std::vector<vid_t> combined_sl;
    for (TreeNode* sibling : node->parent->children) {
        if (sibling == nullptr) {
            continue;
        } else {
            combined_sl.insert(
                    combined_sl.end(),
                    sibling->shortlists.at(tid).begin(),
                    sibling->shortlists.at(tid).end());

            sibling->shortlists.erase(sibling->shortlists.find(tid));
        }
    }
    node->parent->shortlists.emplace(tid, combined_sl);
    return true;
}

bool MultiTenantIndexIVFHierarchical::merge_short_list_recursively(
        TreeNode* node,
        tid_t tid) {
    if (!merge_short_list(node, tid)) {
        return false;
    }

    for (TreeNode* curr = node->parent; curr != nullptr; curr = curr->parent) {
        if (!merge_short_list(curr, tid)) {
            break;
        }
    }
    return true;
}

} // namespace faiss