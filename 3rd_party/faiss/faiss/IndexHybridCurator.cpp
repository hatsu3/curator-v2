#include <faiss/IndexHybridCurator.h>

namespace faiss {

HybridCurator::HybridCurator(
        int d,
        int M,
        int gamma,
        int M_beta,
        int n_branches,
        int leaf_size,
        int n_uniq_labels,
        float sel_threshold)
        : MultiTenantIndex(d, METRIC_L2),
          index_built(false),
          own_fields(true),
          sel_threshold(sel_threshold) {
    IndexFlat* storage = new IndexFlat(d, METRIC_L2);

    this->curator = new MultiTenantIndexIVFHierarchical(
            /*storage=*/storage,
            /*n_clusters=*/n_branches,
            /*bf_capacity=*/n_uniq_labels,
            /*bf_false_pos=*/0.01,
            /*max_sl_size=*/leaf_size,
            /*clus_niter=*/20,
            /*max_leaf_size=*/leaf_size,
            /*nprobe=*/0,      // Not used
            /*prune_thres=*/0, // Not used
            /*variance_boost=*/0.4,
            /*search_ef=*/1, // Should be set manually later
            /*beam_size=*/4);

    this->acorn = new IndexACORN(
            /*storage=*/storage,
            /*M=*/M,
            /*gamma=*/gamma,
            /*metadata=*/acorn_metadata,
            /*M_beta=*/M_beta);
}

HybridCurator::HybridCurator(
        IndexACORN* acorn,
        MultiTenantIndexIVFHierarchical* curator,
        float sel_threshold)
        : MultiTenantIndex(curator->d, curator->metric_type),
          acorn(acorn),
          curator(curator),
          storage(acorn->storage),
          own_fields(false),
          index_built(false),
          sel_threshold(sel_threshold) {
    // Check if ACORN and Curator shares the same storage
    assert(acorn->storage == curator->storage &&
           "ACORN and Curator must share the same storage");
}

HybridCurator::~HybridCurator() {
    if (own_fields) {
        delete storage;
        delete acorn;
        delete curator;
    }
}

void HybridCurator::train(idx_t n, const float* x, ext_lid_t tid) {
    curator->train(n, x, tid);
}

void HybridCurator::add_vector_with_ids(
        idx_t n,
        const float* x,
        const idx_t* labels) {
    // we do not support non-sequential labels because ACORN does not
    // support it otherwise we can maintain a mapping from labels to indices
    for (idx_t i = 0; i < n; i++) {
        assert(labels[i] == i && "Labels must be sequential");
    }

    if (!index_built) {
        storage->add(n, x);
        curator->add_vector_with_ids(n, x, labels);
        acorn->add(n, x);

        ntotal = storage->ntotal;
        acorn_metadata.resize(ntotal);
        std::fill(acorn_metadata.begin(), acorn_metadata.end(), 0);

        index_built = true;
    } else {
        FAISS_THROW_MSG("ACORN does not support adding vectors");
    }
}

void HybridCurator::grant_access(idx_t label, ext_lid_t tid) {
    assert(label >= 0 && "Label must be non-negative");
    curator->grant_access(label, tid);
    bitmaps[tid].add(static_cast<uint64_t>(label));
}

bool HybridCurator::revoke_access(idx_t label, ext_lid_t tid) {
    assert(label >= 0 && "Label must be non-negative");
    bool retv = curator->revoke_access(label, tid);
    if (retv) {
        bitmaps[tid].remove(static_cast<uint64_t>(label));
    }
    return retv;
}

float HybridCurator::get_global_selectivity(ext_lid_t tid) const {
    auto it = bitmaps.find(tid);
    if (it != bitmaps.end()) {
        return it->second.cardinality() / (float)ntotal;
    } else {
        return 0.0f;
    }
}

// TODO: measure the overhead of this function
// TODO: if too high, we will store char vectors directly
char* HybridCurator::populate_filter_id_map(ext_lid_t tid) const {
    filter_id_map.resize(acorn->ntotal);
    std::fill(filter_id_map.begin(), filter_id_map.end(), 0);
    auto it = bitmaps.find(tid);
    if (it != bitmaps.end()) {
        const auto& bitmap = it->second;
        for (auto it = bitmap.begin(); it != bitmap.end(); ++it) {
            uint64_t label = *it;
            if (label < acorn->ntotal) {
                filter_id_map[label] = 1;
            }
        }
    }
    return filter_id_map.data();
}

void HybridCurator::search(
        idx_t n,
        const float* x,
        idx_t k,
        ext_lid_t tid,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    float global_sel = get_global_selectivity(tid);
    if (global_sel > sel_threshold) {
        char* filter = populate_filter_id_map(tid);
        acorn->search(n, x, k, distances, labels, filter, params);
    } else {
        curator->search(n, x, k, tid, distances, labels, params);
    }
}

} // namespace faiss