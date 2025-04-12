#pragma once

#include <faiss/roaring/roaring.hh>

#include <faiss/IndexACORN.h>
#include <faiss/MultiTenantIndexIVFHierarchical.h>

namespace faiss {

struct HybridCurator : MultiTenantIndex {
   private:
    std::unordered_map<ext_lid_t, roaring::Roaring64Map> bitmaps;
    mutable std::vector<char> filter_id_map; // input to acorn search

    float get_global_selectivity(ext_lid_t tid) const;

    char* populate_filter_id_map(ext_lid_t tid) const;

   public:
    IndexACORN* acorn;
    MultiTenantIndexIVFHierarchical* curator;
    float sel_threshold; // selectivity threshold for index selection

    bool index_built; // to prevent incr update since ACORN does not support it
    bool own_fields;  // true if we own the storage
    Index* storage;   // storage shared by ACORN and Curator
    std::vector<int> acorn_metadata;  // not used, make ACORN happy

    HybridCurator(
            int d,
            int M,
            int gamma,
            int M_beta,
            int n_branches,
            int leaf_size,
            int n_uniq_labels, 
            float sel_threshold);

    HybridCurator(IndexACORN* acorn, MultiTenantIndexIVFHierarchical* curator, float sel_threshold);

    ~HybridCurator() override;

    void train(idx_t n, const float* x, ext_lid_t tid) override;

    void add_vector_with_ids(idx_t n, const float* x, const idx_t* labels)
            override;

    void grant_access(idx_t label, ext_lid_t tid) override;

    bool revoke_access(idx_t label, ext_lid_t tid) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            ext_lid_t tid,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void add_vector(idx_t n, const float* x) override {
        FAISS_THROW_MSG("add_vector not supported");
    }

    bool remove_vector(idx_t label) override {
        FAISS_THROW_MSG("remove_vector not supported");
    }

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            ext_lid_t tid,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override {
        FAISS_THROW_MSG("range_search not supported");
    }

    void assign(
            idx_t n,
            const float* x,
            ext_lid_t tid,
            idx_t* labels,
            idx_t k = 1) const override {
        FAISS_THROW_MSG("assign not supported");
    }

    void reset() override {
        FAISS_THROW_MSG("reset not supported");
    }
};

} // namespace faiss