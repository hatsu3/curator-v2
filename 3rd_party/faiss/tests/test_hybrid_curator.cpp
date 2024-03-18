#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/HybridCurator.h>
#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>

TEST(HybridCurator, Test_level0hnsw) {
    faiss::Level0HNSW hnsw(32);

    int nv = 10;   // number of vectors
    int nd = 32;   // dimension of vectors

    // generate 10 random 32-dim vectors
    std::vector<float> vecs;
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> u(0, 1);
    for (int i = 0; i < nv; i++) {
        for (int j = 0; j < nd; j++) {
            vecs.push_back(u(rng));
        }
    }

    // generate a query vector
    std::vector<float> qv;
    for (int i = 0; i < nd; i++) {
        qv.push_back(u(rng));
    }

    // add vectors to vector storage
    faiss::IndexFlatL2 index(nd);
    index.add(nv, vecs.data());
    
    faiss::DistanceComputer* dis = index.get_distance_computer();
    faiss::VisitedTable vt(nv);

    std::vector<omp_lock_t> locks(nv);
    for (int i = 0; i < nv; i++) {
        omp_init_lock(&locks[i]);
    }

    // add vectors to hnsw
    for (int i = 0; i < nv; i++) {
        faiss::Level0HNSW::storage_idx_t entry_pt = i - 1;
        dis->set_query(vecs.data() + i * nd);
        hnsw.add_with_locks(*dis, i, entry_pt, locks, vt);
    }

    int k = 5;

    // search with hnsw
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);
    dis->set_query(qv.data());
    hnsw.search(*dis, k, labels.data(), distances.data(), 0, vt);

    // compute ground truth
    std::vector<float> gt_distances(k);
    std::vector<faiss::idx_t> gt_labels(k);
    index.search(1, qv.data(), k, gt_distances.data(), gt_labels.data());

    for (int i = 0; i < k; i++) {
        EXPECT_EQ(labels[i], gt_labels[i]);
        EXPECT_NEAR(distances[i], gt_distances[i], 1e-6);
    }

    for (int i = 0; i < nv; i++) {
        omp_destroy_lock(&locks[i]);
    }
}