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

using storage_idx_t = Level0HNSW::storage_idx_t;
using NodeDistCloser = Level0HNSW::NodeDistCloser;
using NodeDistFarther = Level0HNSW::NodeDistFarther;

// add a link from src to dest
void add_link(
        Level0HNSW& hnsw,
        DistanceComputer& qdis,
        storage_idx_t src,
        storage_idx_t dest) {
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
        storage_idx_t neigh = hnsw.neighbors[i];
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
            storage_idx_t nodeId = hnsw.neighbors[i];
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
        storage_idx_t pt_id,
        storage_idx_t nearest,
        float d_nearest,
        omp_lock_t* locks,
        VisitedTable& vt) {
    // search for closest vectors to the new point
    std::priority_queue<NodeDistCloser> link_targets;
    search_neighbors_to_add(*this, ptdis, link_targets, nearest, d_nearest, vt);

    // create links to neighbors
    std::vector<storage_idx_t> neighbors;
    neighbors.reserve(link_targets.size());
    while (!link_targets.empty()) {
        storage_idx_t other_id = link_targets.top().id;
        add_link(*this, ptdis, pt_id, other_id);
        neighbors.push_back(other_id);
        link_targets.pop();
    }

    // create back links
    omp_unset_lock(&locks[pt_id]);
    for (storage_idx_t other_id : neighbors) {
        omp_set_lock(&locks[other_id]);
        add_link(*this, ptdis, other_id, pt_id);
        omp_unset_lock(&locks[other_id]);
    }
    omp_set_lock(&locks[pt_id]);
}

void Level0HNSW::add_with_locks(
        DistanceComputer& ptdis,
        int pt_id,
        storage_idx_t entry_pt,
        std::vector<omp_lock_t>& locks,
        VisitedTable& vt) {
    neighbors.resize(neighbors.size() + nbNeighbors, -1);

    // do nothing if this is the first point
    if (entry_pt >= 0) {
        storage_idx_t nearest = entry_pt;
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
        storage_idx_t entry_pt,
        VisitedTable& vt) const {
    storage_idx_t nearest = entry_pt;
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
 * hierarchical zone map
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
    this->tree_root = std::make_unique<TreeNode>(
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
        size_t branch_factor) {
    // stop if there are not enough vectors to split
    if (n < branch_factor * 8) {
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

        // recursively train the child node
        auto cluster_size = cluster_offsets[i + 1] - cluster_offsets[i];
        auto cluster_offset = cluster_offsets[i] * d;
        train_tree_node(
                child_node.get(),
                cluster_size,
                d,
                vecs + cluster_offset,
                branch_factor);

        node->children.push_back(std::move(child_node));
    }
}
} // namespace

void HierarchicalZoneMap::train(float* x, size_t n) {
    train_tree_node(this->tree_root.get(), n, this->d, x, this->branch_factor);
}

namespace {
using TreeNode = HierarchicalZoneMap::TreeNode;

void insert_tree_node(
        TreeNode* node,
        const float* x,
        size_t d,
        storage_idx_t label,
        tid_t tenant) {
    auto it = node->buffers.find(tenant);
    if (it != node->buffers.end()) {
        it->second.push_back(label);
        if (it->second.size() > node->buf_capacity) {
            if (!node->is_leaf) {
                // split the buffer and push it down to the children nodes
                for (auto& child : node->children) {
                    insert_tree_node(child.get(), x, d, label, tenant);
                }
            }
            node->buffers.erase(it);
        }
    } else if (!node->bf->contains(tenant)) {
        node->buffers.emplace(tenant, std::vector<storage_idx_t>{label});
        node->bf->insert(tenant);
    } else if (!node->is_leaf) {
        idx_t branch_idx = 0;
        node->quantizer.assign(1, x, &branch_idx);
        auto child_node = node->children[branch_idx].get();
        insert_tree_node(child_node, x, d, label, tenant);
    }
}
} // namespace

void HierarchicalZoneMap::insert(
        const float* x,
        storage_idx_t label,
        tid_t tenant) {
    insert_tree_node(this->tree_root.get(), x, this->d, label, tenant);
}

const TreeNode* HierarchicalZoneMap::seek(const float* qv) const {
    // recursively traverse the tree to find the leaf node that is closest to
    // the query vector
    const TreeNode* curr = tree_root.get();
    idx_t branch_idx = 0;
    while (!curr->is_leaf) {
        curr->quantizer.assign(1, qv, &branch_idx);
        curr = curr->children[branch_idx].get();
    }
    return curr;
}
} // namespace faiss