#include <iostream>

#include <faiss/HybridCurator.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/utils/prefetch.h>

namespace faiss {

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
            if (nodeId < 0) {  // no more neighbors
                break;
            }
            if (vt.get(nodeId)) {  // already visited
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
        add_links_starting_from(ptdis, pt_id, nearest, d_nearest, locks.data(), vt);
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
    int nres = 0;  // number of results so far

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

    int nstep = 0;  // number of search steps so far

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

} // namespace faiss