// -*- c++ -*-

#ifndef FAISS_MULTI_TENANT_INDEX_IVF_FLAT_H
#define FAISS_MULTI_TENANT_INDEX_IVF_FLAT_H

#include <stdint.h>
#include <unordered_map>

#include <faiss/MultiTenantIndexIVF.h>
#include <faiss/complex_predicate.h>

namespace faiss {

/** Inverted file with stored vectors. Here the inverted file
 * pre-selects the vectors to be searched, but they are not otherwise
 * encoded, the code array just contains the raw float entries.
 */
struct MultiTenantIndexIVFFlat : MultiTenantIndexIVF {
    MultiTenantIndexIVFFlat(
            Index* quantizer,
            size_t d,
            size_t nlist_,
            MetricType = METRIC_L2);

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx) override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons)
            const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    /** Search with complex predicate filter
     *
     * @param n      number of vectors to search
     * @param x      query vectors, size n * d
     * @param k      number of nearest neighbors to return
     * @param filter predicate filter in Polish notation (e.g., "OR 1 2")
     * @param distances output distances, size n * k
     * @param labels output labels, size n * k
     * @param params search parameters
     */
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            const std::string& filter,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    /** Search vectors that are pre-quantized by the IVF quantizer with complex predicate filter
     *
     * @param n      number of vectors to search
     * @param x      query vectors, size n * d
     * @param k      number of nearest neighbors to return
     * @param filter predicate filter in Polish notation (e.g., "OR 1 2")
     * @param keys   coarse quantization indices, size n * nprobe
     * @param coarse_dis distances to coarse centroids, size n * nprobe
     * @param distances output distances, size n * k
     * @param labels output labels, size n * k
     * @param store_pairs store inv list index + inv list offset instead of ids
     * @param params search parameters
     * @param ivf_stats search stats to be updated (can be null)
     */
    void search_preassigned(
            idx_t n,
            const float* x,
            idx_t k,
            const std::string& filter,
            const idx_t* keys,
            const float* coarse_dis,
            float* distances,
            idx_t* labels,
            bool store_pairs,
            const IVFSearchParameters* params = nullptr,
            IndexIVFStats* ivf_stats = nullptr) const;

    MultiTenantIndexIVFFlat();
};

} // namespace faiss

#endif
