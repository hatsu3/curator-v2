#include <array>
#include <iostream>

#include <faiss/HybridCurator.h>

#include <faiss/Clustering.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/prefetch.h>

namespace faiss {

/*
 * graph index for the base level
 */

void Level0HNSW::neighbor_range(idx_t no, size_t* begin, size_t* end) const {
    *begin = no * nbNeighbors;
    *end = (no + 1) * nbNeighbors;
}

Level0HNSW::Level0HNSW(int M) {
    nbNeighbors = M;
}

namespace {
using NodeDistCloser = Level0HNSW::NodeDistCloser;
using NodeDistFarther = Level0HNSW::NodeDistFarther;

// add a link from src to dest
void add_link(Level0HNSW& hnsw, DistanceComputer& qdis, idx_t src, idx_t dest) {
    size_t begin, end;
    hnsw.neighbor_range(src, &begin, &end);

    // if there is room in the neighbors list, find a slot to add it
    // we assume there is no gap in the neighbors list
    if (hnsw.neighbors[end - 1] == -1) {
        for (size_t i = begin; i < end; i++) {
            if (hnsw.neighbors[i] == -1) {
                hnsw.neighbors[i] = dest;
                return;
            }
        }
    }

    // otherwise, we may need to replace an existing neighbor
    // top elements are the farthest from the query
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) {
        auto neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
        if (resultSet.size() > hnsw.nbNeighbors) {
            resultSet.pop();
        }
    }

    size_t i = begin;
    while (!resultSet.empty()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }

    while (i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

void search_neighbors_to_add(
        Level0HNSW& hnsw,
        DistanceComputer& qdis,
        std::priority_queue<NodeDistCloser>& results,
        int entry_point,
        float d_entry_point,
        VisitedTable& vt) {
    // top elements are the closest to the query
    std::priority_queue<NodeDistFarther> candidates;

    // start with the entry point
    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get the closest vector to the query
        const NodeDistFarther& currEv = candidates.top();

        if (currEv.d > results.top().d) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // explore the neighbors of the current node
        size_t begin, end;
        hnsw.neighbor_range(currNode, &begin, &end);
        for (size_t i = begin; i < end; i++) {
            idx_t nodeId = hnsw.neighbors[i];
            if (nodeId < 0) { // no more neighbors
                break;
            }
            if (vt.get(nodeId)) { // already visited
                continue;
            }
            vt.set(nodeId);

            float dis = qdis(nodeId);

            if (results.size() < hnsw.efConstruction || results.top().d > dis) {
                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }
        }
    }
    vt.advance();
}

} // namespace

void Level0HNSW::add_links_starting_from(
        DistanceComputer& ptdis,
        idx_t pt_id,
        idx_t nearest,
        float d_nearest,
        omp_lock_t* locks,
        VisitedTable& vt) {
    // search for closest vectors to the new point
    std::priority_queue<NodeDistCloser> link_targets;
    search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest, vt);

    // create links to neighbors
    std::vector<idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
        auto other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id);
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    // create back links
    omp_unset_lock(&locks[pt_id]);
    for (auto other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

void Level0HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_id,
        idx_t entry_pt,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {
    neighbors.resize(neighbors.size() + nbNeighbors, -1);

    // do nothing if this is the first point
    if (entry_pt >= 0) {
        auto nearest = entry_pt;
        float d_nearest = ptdis(nearest);

        omp_set_lock(&locks[pt_id]);
        add_links_starting_from(
                ptdis, pt_id, nearest, d_nearest, locks.data(), vt);
        omp_unset_lock(&locks[pt_id]);
    }
}

namespace {

using MinimaxHeap = HNSW::MinimaxHeap;
using Node = Level0HNSW::Node;

int search_from_candidates(
        const Level0HNSW& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt) {
    int nres = 0; // number of results so far

    // add all initial candidates to the heap
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (nres < k) {
            faiss::maxheap_push(++nres, D, I, d, v1);
        } else if (d < D[0]) {
            faiss::maxheap_replace_top(nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    int nstep = 0; // number of search steps so far

    while (candidates.size() > 0) {
        // pop the candidate closest to the query from the search frontier
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (hnsw.check_relative_distance) {
            // tricky stopping condition: there are more than ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= hnsw.efSearch) {
                break;
            }
        }

        // explore the neighbors of the current node
        size_t begin, end;
        hnsw.neighbor_range(v0, &begin, &end);

        size_t jmax = begin;
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;

            prefetch_L2(vt.visited.data() + v1);
            jmax += 1;
        }

        int counter = 0;
        size_t saved_j[4];

        auto add_to_heap = [&](const size_t idx, const float dis) {
            if (nres < k) {
                faiss::maxheap_push(++nres, D, I, dis, idx);
            } else if (dis < D[0]) {
                faiss::maxheap_replace_top(nres, D, I, dis, idx);
            }
            candidates.push(idx, dis);
        };

        // process 4 neighbors at a time
        for (size_t j = begin; j < jmax; j++) {
            int v1 = hnsw.neighbors[j];

            bool vget = vt.get(v1);
            vt.set(v1);
            saved_j[counter] = v1;
            counter += vget ? 0 : 1;

            if (counter == 4) {
                float dis[4];
                qdis.distances_batch_4(
                        saved_j[0],
                        saved_j[1],
                        saved_j[2],
                        saved_j[3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t id4 = 0; id4 < 4; id4++) {
                    add_to_heap(saved_j[id4], dis[id4]);
                }

                counter = 0;
            }
        }

        // process the remaining neighbors
        for (size_t icnt = 0; icnt < counter; icnt++) {
            float dis = qdis(saved_j[icnt]);
            add_to_heap(saved_j[icnt], dis);
        }

        nstep++;
        if (!hnsw.check_relative_distance && nstep > hnsw.efSearch) {
            break;
        }
    }

    return nres;
}

} // namespace

void Level0HNSW::search(
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        idx_t entry_pt,
        VisitedTable& vt) const {
    auto nearest = entry_pt;
    float d_nearest = qdis(nearest);

    MinimaxHeap candidates(std::max(efSearch, k));
    candidates.push(nearest, d_nearest);

    // ``search_from_candidates`` returns unsorted results
    // so we need to reorder the results afterwards
    maxheap_heapify(k, D, I);
    search_from_candidates(*this, qdis, k, I, D, candidates, vt);
    maxheap_reorder(k, D, I);

    vt.advance();
}

/*
 * hierarchical zone map implementation
 */

HierarchicalZoneMap::TreeNode::TreeNode(
        size_t d,
        std::vector<int8_t>&& node_id,
        TreeNode* parent,
        size_t bf_capacity,
        float bf_error_rate,
        size_t buf_capacity)
        : node_id(std::move(node_id)),
          parent(parent),
          is_leaf(true),
          bf_capacity(bf_capacity),
          bf_error_rate(bf_error_rate),
          buf_capacity(buf_capacity),
          quantizer(d) {
    // initialize the bloom filter
    bloom_parameters bf_params;
    bf_params.projected_element_count = bf_capacity;
    bf_params.false_positive_probability = bf_error_rate;
    bf_params.compute_optimal_parameters();
    this->bf = std::make_unique<bloom_filter>(bf_params);
}

HierarchicalZoneMap::HierarchicalZoneMap(
        size_t d,
        size_t branch_factor,
        size_t bf_capacity,
        float bf_error_rate,
        size_t buf_capacity)
        : d(d),
          branch_factor(branch_factor),
          bf_capacity(bf_capacity),
          bf_error_rate(bf_error_rate),
          buf_capacity(buf_capacity) {
    tree_root = std::make_unique<TreeNode>(
            d,
            std::vector<int8_t>(),
            nullptr,
            bf_capacity,
            bf_error_rate,
            buf_capacity);
}

namespace {
using TreeNode = HierarchicalZoneMap::TreeNode;

// perform k-NN clustering on the vectors (train the quantizer)
void cluster_vectors(
        size_t n,
        size_t d,
        const float* vecs,
        size_t k,
        IndexFlatL2& quantizer) {
    ClusteringParameters cp;
    Clustering clus(d, k, cp);
    clus.train(n, vecs, quantizer);
}

// group the vectors according to the cluster assignments
// returns the offsets of each cluster in the vector array
std::vector<size_t> group_vectors_by_cluster(
        size_t n,
        size_t d,
        float* vecs,
        IndexFlatL2& quantizer) {
    // assign the vectors to clusters
    std::vector<idx_t> cluster_ids(n);
    quantizer.assign(n, vecs, cluster_ids.data());

    // determine the order of the vectors
    std::vector<idx_t> vec_indices(n);
    for (idx_t i = 0; i < n; i++) {
        vec_indices[i] = i;
    }
    std::sort(vec_indices.begin(), vec_indices.end(), [&](idx_t i, idx_t j) {
        return cluster_ids[i] < cluster_ids[j];
    });

    // reorder the vectors
    std::vector<float> vecs_copy(n * d);
    memcpy(vecs_copy.data(), vecs, n * d * sizeof(float));
    for (idx_t i = 0; i < n; i++) {
        memcpy(vecs + i * d,
               vecs_copy.data() + vec_indices[i] * d,
               d * sizeof(float));
    }

    // count the number of vectors in each cluster
    size_t n_clusters = quantizer.ntotal;
    std::vector<size_t> cluster_sizes(n_clusters, 0);
    for (auto cluster_id : cluster_ids) {
        cluster_sizes[cluster_id]++;
    }

    // calculate the offsets of each cluster in the vector array
    std::vector<size_t> cluster_offsets(n_clusters + 1);
    cluster_offsets[0] = 0;
    for (size_t i = 1; i < n_clusters + 1; i++) {
        cluster_offsets[i] = cluster_offsets[i - 1] + cluster_sizes[i - 1];
    }

    return cluster_offsets;
}

// recursively train the tree structure
void train_tree_node(
        TreeNode* node,
        size_t n,
        size_t d,
        float* vecs,
        size_t branch_factor,
        size_t level,
        size_t& tree_size,
        size_t split_thres,
        size_t max_depth) {
    // stop if there are not enough vectors to split
    if (n < split_thres || level > max_depth) {
        return;
    }

    node->is_leaf = false;

    // cluster the vectors
    cluster_vectors(n, d, vecs, branch_factor, node->quantizer);
    auto cluster_offsets =
            group_vectors_by_cluster(n, d, vecs, node->quantizer);

    for (size_t i = 0; i < branch_factor; i++) {
        // create a child node
        std::vector<int8_t> child_node_id = node->node_id;
        child_node_id.push_back(i);
        auto child_node = std::make_unique<TreeNode>(
                d,
                std::move(child_node_id),
                node,
                node->bf_capacity,
                node->bf_error_rate,
                node->buf_capacity);
        tree_size++;

        // recursively train the child node
        auto cluster_size = cluster_offsets[i + 1] - cluster_offsets[i];
        auto cluster_offset = cluster_offsets[i] * d;
        train_tree_node(
                child_node.get(),
                cluster_size,
                d,
                vecs + cluster_offset,
                branch_factor,
                level + 1,
                tree_size,
                split_thres,
                max_depth);

        node->children.push_back(std::move(child_node));
    }
}
} // namespace

void HierarchicalZoneMap::train(float* x, size_t n, size_t split_thres) {
    this->size = 1; // root node

    if (split_thres < branch_factor * 8) {
        std::cerr << "Warning: split threshold is too small (" << split_thres
                  << "), setting it to " << branch_factor * 8 << std::endl;
        split_thres = branch_factor * 8;
    }

    train_tree_node(
            tree_root.get(),
            n,
            this->d,
            x,
            branch_factor,
            /*level*/ 0,
            this->size,
            split_thres,
            /*max_depth*/ std::numeric_limits<size_t>::max());
}

void HierarchicalZoneMap::train_with_depth_limit(
        float* x,
        size_t n,
        size_t max_depth) {
    this->size = 1; // root node

    // the split threshold is set to a small but non-zero value to ensure that
    // we always have enough vectors to split at each level
    train_tree_node(
            tree_root.get(),
            n,
            this->d,
            x,
            branch_factor,
            /*level*/ 0,
            this->size,
            /*split_thres*/ branch_factor * 8,
            max_depth);
}

namespace {
using TreeNode = HierarchicalZoneMap::TreeNode;
using node_id_t = HierarchicalZoneMap::node_id_t;
using buffer_t = HierarchicalZoneMap::buffer_t;

TreeNode* find_nearest_leaf(TreeNode* node, const float* x, size_t d) {
    while (!node->is_leaf) {
        idx_t branch_idx = 0;
        node->quantizer.assign(1, x, &branch_idx);
        node = node->children[branch_idx].get();
    }
    return node;
}

void insert_tree_node(
        TreeNode* node,
        int level,
        node_id_t& leaf_id,
        idx_t label,
        tid_t tenant,
        HierarchicalZoneMap& zone_map) {
    auto it = node->buffers.find(tenant);
    if (it != node->buffers.end()) {
        it->second.push_back(label);
        if (it->second.size() > node->buf_capacity && !node->is_leaf) {
            // split the buffer and push it down to the children nodes
            for (auto& lbl : it->second) {
                auto leaf = zone_map.vec2leaf.at(lbl);
                auto& leaf_id1 = leaf->node_id;
                auto child = node->children[leaf_id1.at(level)].get();
                insert_tree_node(
                        child, level + 1, leaf_id1, lbl, tenant, zone_map);
            }
            node->buffers.erase(it);
        }
    } else if (!node->bf->contains(tenant)) {
        node->buffers.emplace(tenant, buffer_t{label});
        node->bf->insert(tenant);
    } else if (!node->is_leaf) {
        auto child = node->children[leaf_id.at(level)].get();
        insert_tree_node(child, level + 1, leaf_id, label, tenant, zone_map);
    } else {
        // detected false positive at the leaf node, create a buffer anyway
        node->buffers.emplace(tenant, buffer_t{label});
    }
}
} // namespace

void HierarchicalZoneMap::insert(const float* x, idx_t label, tid_t tenant) {
    if (vec2leaf.find(label) != vec2leaf.end()) {
        FAISS_THROW_FMT("Vector with label %ld already exists", label);
    }
    auto root = tree_root.get();
    auto leaf = find_nearest_leaf(root, x, this->d);
    vec2leaf.emplace(label, leaf);
    insert_tree_node(root, 0, leaf->node_id, label, tenant, *this);
}

void HierarchicalZoneMap::grant_access(idx_t label, tid_t tenant) {
    auto it = vec2leaf.find(label);
    if (it == vec2leaf.end()) {
        FAISS_THROW_FMT("Vector with label %ld does not exist", label);
    }
    auto leaf = it->second;
    insert_tree_node(tree_root.get(), 0, leaf->node_id, label, tenant, *this);
}

namespace {
const TreeNode* seek_from_node(
        const TreeNode* node,
        const float* qv,
        tid_t tid,
        std::vector<idx_t> corder) {
    if (node->is_leaf) {
        if (node->buffers.find(tid) != node->buffers.end()) {
            return node;
        }
        return nullptr;
    }

    auto buffer_it = node->buffers.find(tid);
    if (buffer_it != node->buffers.end()) {
        // if this node contains a buffer of the tenant, return it
        return node;
    } else if (node->bf->contains(tid)) {
        // examine children in the order of increasing distance
        auto n_children = node->children.size();
        node->quantizer.assign(1, qv, corder.data(), n_children);
        for (auto i = 0; i < n_children; i++) {
            auto child = node->children[corder[i]].get();
            if (auto res = seek_from_node(child, qv, tid, corder)) {
                return res;
            }
        }
    }

    // if this node does not contain any vector accessible to the tenant
    return nullptr;
}
} // namespace

const TreeNode* HierarchicalZoneMap::seek(const float* qv, tid_t tid) const {
    // allocate a buffer to store the order of the children in order to
    // avoid incurring repeated memory allocation in the recursive calls
    auto corder = std::vector<idx_t>(branch_factor, 0);
    return seek_from_node(tree_root.get(), qv, tid, corder);
}

const TreeNode* HierarchicalZoneMap::seek(const node_id_t& leaf_id, tid_t tid)
        const {
    auto node = tree_root.get();
    if (node->buffers.find(tid) != node->buffers.end()) {
        return node;
    }

    for (auto id : leaf_id) {
        node = node->children.at(id).get();
        if (node->buffers.find(tid) != node->buffers.end()) {
            return node;
        }
    }
    return nullptr;
}

const TreeNode* HierarchicalZoneMap::seek_empty(const TreeNode* leaf, tid_t tid)
        const {
    auto node = leaf;
    while (node && node->bf->contains(tid)) {
        node = node->parent;
    }

    if (!node) {
        // check for false positives in the bloom filter
        if (leaf->buffers.find(tid) == leaf->buffers.end()) {
            return leaf;
        }
    }
    return node;
}

/*
 * visited subtree table implementation
 */

template <typename T>
void VisitedSubtreeTable<T>::set(const node_id_t& node_id) {
    TrieNode* curr = &trie[0];
    for (auto& id : node_id) {
        auto it = curr->children.find(id);
        if (it == curr->children.end()) {
            curr->children.emplace(id, this->size++);
        }
        curr = &trie[curr->children.at(id)];
    }
    curr->is_end_of_prefix = true;
}

template <typename T>
bool VisitedSubtreeTable<T>::get(const node_id_t& node_id) const {
    const TrieNode* curr = &trie[0];
    if (curr->is_end_of_prefix) {
        return true;
    }

    for (auto& id : node_id) {
        auto it = curr->children.find(id);
        if (it == curr->children.end()) {
            return false;
        }
        curr = &trie[curr->children.at(id)];
        if (curr->is_end_of_prefix) {
            return true;
        }
    }
    return curr->is_end_of_prefix;
}

template <typename T>
void VisitedSubtreeTable<T>::clear() {
    for (auto i = 0; i < this->size; i++) {
        // the capacity (memory consumption) of the unordered_map should
        // remain unchanged after erasure
        trie[i].children.clear();
        trie[i].is_end_of_prefix = false;
    }

    this->size = 1;
}

template struct VisitedSubtreeTable<int8_t>;

/*
 * hybrid curator implementation
 */

HybridCurator::HybridCurator(
        size_t d,
        size_t M,
        size_t branch_factor,
        size_t bf_capacity,
        float bf_error_rate,
        size_t buf_capacity)
        : d(d),
          storage(d),
          base_level(M),
          zone_map(d, branch_factor, bf_capacity, bf_error_rate, buf_capacity) {
}

HybridCurator::~HybridCurator() {}

void HybridCurator::train(idx_t n, const float* x, tid_t tid) {
    std::vector<float> x_copy(x, x + n * d);
    zone_map.train(x_copy.data(), n, zone_map.branch_factor * 8);
    is_trained = true;
}

void HybridCurator::add_vector(idx_t n, const float* x, tid_t tid) {
    FAISS_ASSERT_MSG(is_trained, "Curator is not trained");

    auto ntotal_old = storage.ntotal;
    for (idx_t i = 0; i < n; i++) {
        access_matrix.emplace(ntotal_old + i, AccessList{tid});
    }

    // add the vectors to the storage
    auto ntotal_new = ntotal_old + n;
    storage.add(n, x);

    // add the vectors to the base-level graph index and the zone map
    auto dis = storage.get_distance_computer();
    auto vt = faiss::VisitedTable(ntotal_new);

    auto locks = std::vector<omp_lock_t>(ntotal_new);
    for (auto& lock : locks) {
        omp_init_lock(&lock);
    }

    for (idx_t i = 0; i < n; i++) {
        dis->set_query(x + i * d);
        base_level.add_with_locks(*dis, ntotal_old + i, entry_pt, locks, vt);
        zone_map.insert(x + i * d, ntotal_old + i, tid);

        if (entry_pt < 0) {
            entry_pt = ntotal_old + i;
        }
    }

    for (auto& lock : locks) {
        omp_destroy_lock(&lock);
    }
}

void HybridCurator::grant_access(idx_t xid, tid_t tid) {
    // add the tenant to the access list of the vector
    auto it = access_matrix.find(xid);
    if (it == access_matrix.end()) {
        FAISS_THROW_FMT("Vector with id %ld does not exist", xid);
    }

    auto access_list = it->second;
    auto it2 = std::find(access_list.begin(), access_list.end(), tid);
    if (it2 == access_list.end()) {
        access_list.insert(tid);
        zone_map.grant_access(xid, tid);
    }
}

namespace {
using buffer_t = HierarchicalZoneMap::buffer_t;

void add_buffer_to_heap(
        const buffer_t& buffer,
        MinimaxHeap& candidates,
        float* distances,
        idx_t* labels,
        int& nres,
        int k,
        DistanceComputer& qdis) {
    auto add_to_heap = [&](idx_t label, float d) {
        // add the vector to the result heap
        if (nres < k) {
            faiss::maxheap_push(++nres, distances, labels, d, label);
        } else if (d < distances[0]) {
            faiss::maxheap_replace_top(nres, distances, labels, d, label);
        }

        // add the vector to the search frontier
        candidates.push(label, d);
    };

    std::array<float, 4> dis;
    size_t i = 0;

    if (buffer.size() >= 4) {
        for (; i < buffer.size() - 4 + 1; i += 4) {
            // calculate distances in batches of 4
            qdis.distances_batch_4(
                    buffer[i],
                    buffer[i + 1],
                    buffer[i + 2],
                    buffer[i + 3],
                    dis[0],
                    dis[1],
                    dis[2],
                    dis[3]);

            for (size_t j = 0; j < 4; j++) {
                add_to_heap(buffer[i + j], dis[j]);
            }
        }
    }

    // process the remaining vectors
    for (; i < buffer.size(); i++) {
        add_to_heap(buffer[i], qdis(buffer[i]));
    }
}
} // namespace

// TODO: this version suffers from loss of connectivity during graph traversal
// because it does not further explore the inaccessible vectors
void HybridCurator::search(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels) const {
    FAISS_ASSERT_MSG(is_trained, "Curator is not trained");

    // initialize the result heap
    auto nres = static_cast<int>(0); // number of results so far
    maxheap_heapify(
            k,
            distances,
            labels); // fill the label array with invalid labels (-1)

    // find the node that is closest to the query vector and contains a buffer
    auto init_node = zone_map.seek(x, tid);
    if (!init_node) {
        // seek returning nullptr indicates that the tenant has no access to
        // any vector in the index
        return;
    }

    // maintain the search frontier in a priority queue
    auto efSearch = base_level.efSearch;
    auto candidates = MinimaxHeap(std::max(efSearch, static_cast<int>(k)));

    // initialize the visited set and the distance computer for the query
    auto vt = VisitedSubtreeTable<int8_t>(zone_map.size);
    vt.set(init_node->node_id); // mark vectors in the initial node as visited
    auto qdis = storage.get_distance_computer();
    qdis->set_query(x);

    // helper functions used in the search
    auto add_buffer_to_heap1 =
            [&](const HierarchicalZoneMap::buffer_t& buffer) {
                add_buffer_to_heap(
                        buffer, candidates, distances, labels, nres, k, *qdis);
            };

    auto get_neighbors = [&](idx_t label) -> std::pair<size_t, size_t> {
        size_t begin, end;
        base_level.neighbor_range(label, &begin, &end);

        for (auto j = begin; j < end; j++) {
            if (base_level.neighbors[j] < 0) {
                return {begin, j};
            }
        }

        return {begin, end};
    };

    // initialize the result set and search frontier with the vectors in
    // the buffer of the initial node
    auto& init_buffer = init_node->buffers.at(tid);
    add_buffer_to_heap1(init_buffer);

    float d0;
    size_t begin, end;

    while (candidates.size() > 0) {
        auto v0 = candidates.pop_min(&d0);
        std::tie(begin, end) = get_neighbors(v0);

        // stop search if no candidate can improve the result
        if (d0 > distances[0]) {
            break;
        }

        for (auto j = begin; j < end; j++) {
            auto v1 = base_level.neighbors[j];
            auto leaf = zone_map.vec2leaf.at(v1);
            auto& leaf_id = leaf->node_id;
            auto visited = vt.get(leaf_id);

            if (!visited) {
                auto cnode = zone_map.seek(leaf_id, tid);
                if (!cnode) {
                    // if we cannot find an ancestor node that contains a buffer
                    // look for an ancestor node that can be skipped instead
                    cnode = zone_map.seek_empty(leaf, tid);
                } else {
                    add_buffer_to_heap1(cnode->buffers.at(tid));
                }
                vt.set(cnode->node_id);
            }
        }
    }

    maxheap_reorder(k, distances, labels);
}

HybridCuratorV2::HybridCuratorV2(
        size_t d,
        size_t M,
        size_t tree_depth,
        size_t branch_factor,
        float alpha,
        size_t bf_capacity,
        float bf_error_rate,
        size_t buf_capacity)
        : d(d),
          tree_depth(tree_depth),
          alpha(alpha),
          storage(d),
          zone_map(d, branch_factor, bf_capacity, bf_error_rate, buf_capacity) {
    // initialize graph index for the lowest ``index_levels``-levels of the tree
    for (size_t level = 1; level <= tree_depth; level++) {
        level_indexes.emplace(level, Level0HNSW(M));
        level_storages.emplace(level, IndexFlatL2(d));
        auto dis = level_storages.at(level).get_distance_computer();
        idx2node.emplace(level, std::unordered_map<idx_t, TreeNode*>());
    }
}

void HybridCuratorV2::train(idx_t n, const float* x, tid_t tid) {
    std::vector<float> x_copy(x, x + n * d);
    zone_map.train_with_depth_limit(x_copy.data(), n, tree_depth);

    // insert tree nodes into their corresponding indexes
    std::queue<std::pair<TreeNode*, size_t>> q;
    q.emplace(zone_map.tree_root.get(), 0);

    // collect nodes at each level of the tree
    std::unordered_map<int, std::vector<TreeNode*>> level_nodes;
    for (size_t level = 1; level <= tree_depth; level++) {
        level_nodes.emplace(level, std::vector<TreeNode*>());
    }

    while (!q.empty()) {
        auto [node, level] = q.front();
        q.pop();

        if (level > 0) {
            level_nodes.at(level).push_back(node);
        }

        for (auto& child : node->children) {
            q.emplace(child.get(), level + 1);
        }
    }

    // store the centroids of the nodes at each level
    std::vector<float> centroid(d);
    for (size_t level = 1; level <= tree_depth; level++) {
        auto& nodes = level_nodes.at(level);
        auto& storage = level_storages.at(level);

        for (auto node : nodes) {
            auto branch_idx = node->node_id.back();
            node->parent->quantizer.reconstruct(branch_idx, centroid.data());
            idx2node.at(level).emplace(storage.ntotal, node);
            node2idx.emplace(node, std::make_pair(level, storage.ntotal));
            storage.add(1, centroid.data());
        }
    }

    // insert centroids at each level into the corresponding index
    for (size_t level = 1; level <= tree_depth; level++) {
        auto num_nodes = level_nodes.at(level).size();
        auto& index = level_indexes.at(level);
        auto& storage = level_storages.at(level);

        auto dis = storage.get_distance_computer();
        auto vt = faiss::VisitedTable(num_nodes);

        // TODO: consider removing locks
        auto locks = std::vector<omp_lock_t>(num_nodes);
        for (auto& lock : locks) {
            omp_init_lock(&lock);
        }

        for (size_t j = 0; j < num_nodes; j++) {
            storage.reconstruct(j, centroid.data());
            dis->set_query(centroid.data());
            index.add_with_locks(*dis, j, j == 0 ? -1 : 0, locks, vt);
        }

        for (auto& lock : locks) {
            omp_destroy_lock(&lock);
        }

        level_dist_comps.emplace(level, std::unique_ptr<DistanceComputer>(dis));
    }
}

void HybridCuratorV2::add_vector(idx_t n, const float* x, tid_t tid) {
    // update the access matrix
    auto ntotal_old = storage.ntotal;
    for (idx_t i = 0; i < n; i++) {
        access_matrix.emplace(ntotal_old + i, AccessList{tid});
    }

    // add the vectors to the storage
    storage.add(n, x);

    // update the zone map
    for (idx_t i = 0; i < n; i++) {
        zone_map.insert(x + i * d, ntotal_old + i, tid);
    }
}

void HybridCuratorV2::grant_access(idx_t xid, tid_t tid) {
    // add the tenant to the access list of the vector
    auto it = access_matrix.find(xid);
    if (it == access_matrix.end()) {
        FAISS_THROW_FMT("Vector with id %ld does not exist", xid);
    }

    auto access_list = it->second;
    auto it2 = std::find(access_list.begin(), access_list.end(), tid);
    if (it2 == access_list.end()) {
        access_list.insert(tid);
        zone_map.grant_access(xid, tid);
    }
}

void HybridCuratorV2::search(
        idx_t n,
        const float* x,
        idx_t k,
        tid_t tid,
        float* distances,
        idx_t* labels) const {
    // initialize the result heap
    auto nres = static_cast<int>(0);
    maxheap_heapify(k, distances, labels);

    // find the node that is closest to the query vector and contains a buffer
    auto init_node = zone_map.seek(x, tid);
    if (!init_node) {
        // seek returning nullptr indicates that the tenant has no access to
        // any vector in the index
        return;
    }

    // maintain the search frontier in a priority queue
    using Node = std::pair<float /*distance*/, const TreeNode*>;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> candidates;

    // initialize the visited set and the distance computer
    // TODO: replace vt with a more efficient data structure
    auto vt = std::unordered_set<const TreeNode*>();
    vt.insert(init_node);

    auto qdis = storage.get_distance_computer();
    qdis->set_query(x);

    // helper functions used in the search
    auto add_vec_to_heap = [&](idx_t label, float d) {
        if (nres < k) {
            faiss::maxheap_push(++nres, distances, labels, d, label);
        } else if (d < distances[0]) {
            faiss::maxheap_replace_top(nres, distances, labels, d, label);
        }
    };

    auto dis_to_qv = [&](const TreeNode* node) {
        auto [level, lidx] = node2idx.at(node);
        auto& ctrd_qdis = level_dist_comps.at(level);
        ctrd_qdis->set_query(x);
        return (*ctrd_qdis)(lidx);
    };

    auto add_buffer_to_heap = [&](const buffer_t& buffer) {
        std::array<float, 4> dis;
        size_t i = 0;

        if (buffer.size() >= 4) {
            for (; i < buffer.size() - 4 + 1; i += 4) {
                // calculate distances in batches of 4
                qdis->distances_batch_4(
                        buffer[i],
                        buffer[i + 1],
                        buffer[i + 2],
                        buffer[i + 3],
                        dis[0],
                        dis[1],
                        dis[2],
                        dis[3]);

                for (size_t j = 0; j < 4; j++) {
                    add_vec_to_heap(buffer[i + j], dis[j]);
                }
            }
        }

        // process the remaining vectors
        for (; i < buffer.size(); i++) {
            add_vec_to_heap(buffer[i], (*qdis)(buffer[i]));
        }
    };

    // initialize the result set and search frontier with the vectors in
    // the buffer of the initial node
    auto& init_buffer = init_node->buffers.at(tid);
    add_buffer_to_heap(init_buffer);
    candidates.emplace(dis_to_qv(init_node), init_node);

    size_t begin, end;

    // TODO: perform A/B testing to determine whether the optimization of
    // skipping external nodes is beneficial (which assumes that the cost
    // of distance computation dominates the cost of bloom filter lookup)
    while (candidates.size() > 0) {
        auto [dis, node] = candidates.top();
        candidates.pop();

        if (dis > alpha * distances[0]) {
            break;
        }

        auto buffer_it = node->buffers.find(tid);
        if (buffer_it != node->buffers.end()) {
            // if node is a leaf node, add its buffer to the result heap
            // and add its neighbors to the search frontier
            add_buffer_to_heap(buffer_it->second);

            // root node has no neighbors
            if (node == zone_map.tree_root.get()) {
                continue;
            }

            auto [level, lidx] = node2idx.at(node);
            auto& index = level_indexes.at(level);
            index.neighbor_range(lidx, &begin, &end);

            for (auto j = begin; j < end; j++) {
                auto neigh = idx2node.at(level).at(index.neighbors[j]);
                // optimization: skip to the ancestor node that is not an
                // external node to reduce the number of distance computations
                while (neigh && !neigh->bf->contains(tid)) {
                    neigh = neigh->parent;
                }
                if (vt.find(neigh) == vt.end()) {
                    candidates.emplace(dis_to_qv(neigh), neigh);
                    vt.insert(neigh);
                }
            }
        } else if (!node->bf->contains(tid)) {
            // if node is an external node, add its parent node to the search
            // frontier
            auto parent = node->parent;
            // optimization: skip to the ancestor node that is not an
            // external node to reduce the number of distance computations
            while (parent && !parent->bf->contains(tid)) {
                parent = parent->parent;
            }
            candidates.emplace(dis_to_qv(parent), parent);
            vt.insert(parent);
        } else {
            // if node is an internal node, add its children to the search
            // frontier
            for (auto& child : node->children) {
                // optimization: skip children that are external nodes
                if (child->bf->contains(tid)) {
                    auto c = child.get();
                    candidates.emplace(dis_to_qv(c), c);
                    vt.insert(c);
                }
            }
        }
    }

    maxheap_reorder(k, distances, labels);
}

} // namespace faiss