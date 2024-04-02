#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <faiss/HybridCurator.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>

TEST(HybridCurator, Test_level0hnsw) {
    faiss::Level0HNSW hnsw(32);

    int nv = 10; // number of vectors
    int nd = 32; // dimension of vectors

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
        faiss::idx_t entry_pt = i - 1;
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

TEST(HybridCurator, Test_hierarchical_zone_map) {
    // generate 1000 random 32-dim vectors
    int nv = 1000;
    int nd = 32;
    std::vector<float> vecs;
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> u(0, 1);
    for (int i = 0; i < nv; i++) {
        for (int j = 0; j < nd; j++) {
            vecs.push_back(u(rng));
        }
    }

    // generate a random tenant id for each vector
    int nt = 10; // number of tenants
    std::vector<faiss::tid_t> tenant_ids;
    std::uniform_int_distribution<int> t(0, nt - 1);
    for (int i = 0; i < nv; i++) {
        tenant_ids.push_back(t(rng));
    }

    // initialize hierarchical zone map
    int buf_capacity = 8;
    auto hzm = faiss::HierarchicalZoneMap(
            nd, /*branch_factor=*/4, nt, /*bf_error_rate=*/0.01, buf_capacity);

    // train hierarchical zone map
    hzm.train(vecs.data(), nv);

    // insert vectors into hierarchical zone map
    for (int i = 0; i < nv; i++) {
        hzm.insert(vecs.data() + i * nd, /*label=*/i, tenant_ids[i]);
    }

    // randomly sample 20% of the vectors as query vectors
    std::vector<faiss::idx_t> sampled_vids;
    std::uniform_int_distribution<int> v(0, nv - 1);
    for (int i = 0; i < nv / 5; i++) {
        sampled_vids.push_back(v(rng));
    }

    // search with hierarchical zone map
    for (auto qv_idx : sampled_vids) {
        auto tid = tenant_ids[qv_idx];
        auto vec = vecs.data() + qv_idx * nd;
        auto leaf = hzm.vec2leaf.at(qv_idx);
        auto closest_node = hzm.seek(vec, tid);
        auto cnode = hzm.seek(leaf->node_id, tid);
        EXPECT_EQ(closest_node, cnode);
        auto& buffers = closest_node->buffers;
        EXPECT_NE(buffers.find(tid), buffers.end());
    }
}

TEST(HybridCurator, Test_visited_subtree_table) {
    using node_id_t = faiss::HierarchicalZoneMap::node_id_t;

    auto vst_size = 256;
    auto vt = faiss::VisitedSubtreeTable<int8_t>(vst_size);

    vt.set({1, 2, 3});
    EXPECT_EQ(vt.get({1, 2}), false);
    EXPECT_EQ(vt.get({1, 2, 3}), true);
    EXPECT_EQ(vt.get({1, 2, 4}), false);
    EXPECT_EQ(vt.get({1, 2, 3, 4}), true);

    vt.set({1, 2});
    EXPECT_EQ(vt.get({1}), false);
    EXPECT_EQ(vt.get({1, 2}), true);
    EXPECT_EQ(vt.get({1, 2, 4}), true);
}

TEST(HybridCurator, Test_insert_search) {
    using idx_t = faiss::idx_t;

    // generate 1000 random 32-dim vectors
    int nv = 1000;
    int nd = 32;
    std::vector<float> vecs;
    std::default_random_engine rng(42);
    std::uniform_real_distribution<float> u(0, 1);
    for (int i = 0; i < nv; i++) {
        for (int j = 0; j < nd; j++) {
            vecs.push_back(u(rng));
        }
    }

    // generate a random tenant id for each vector
    int nt = 10; // number of tenants
    std::vector<faiss::tid_t> tenant_ids;
    std::uniform_int_distribution<int> t(0, nt - 1);
    for (int i = 0; i < nv; i++) {
        tenant_ids.push_back(t(rng));
    }

    // initialize index
    auto index = faiss::HybridCuratorV2(
            /*d*/ nd,
            /*M*/ 8,
            /*tree-depth*/ 5,
            /*br-fact*/ 4,
            /*alpha*/ 1.0,
            /*bf-capa*/ nt,
            /*bf-error*/ 0.01,
            /*buf-capa*/ 8);

    index.train(nv, vecs.data(), /*tid (unused)*/ 0);

    // insert vectors into index
    for (idx_t i = 0; i < nv; i++) {
        index.add_vector(1, vecs.data() + i * nd, tenant_ids[i]);
    }

    // ground truth per-tenant indexes
    std::vector<std::unique_ptr<faiss::IndexFlatL2>> tenant_indexes;
    std::vector<std::unordered_map<idx_t, idx_t>> lidx_to_gidx(nt);

    for (int i = 0; i < nt; i++) {
        auto index = std::make_unique<faiss::IndexFlatL2>(nd);
        tenant_indexes.emplace_back(std::move(index));
    }
    for (idx_t i = 0; i < nv; i++) {
        auto tid = tenant_ids[i];
        tenant_indexes[tid]->add(1, vecs.data() + i * nd);
        lidx_to_gidx[tid].emplace(lidx_to_gidx[tid].size(), i);
    }

    // randomly sample 20% of the vectors as query vectors
    std::vector<idx_t> sampled_vids;
    std::uniform_int_distribution<int> v(0, nv - 1);
    for (int i = 0; i < nv / 5; i++) {
        sampled_vids.push_back(v(rng));
    }

    auto topk_recall = [](int k,
                          const std::vector<idx_t>& labels,
                          const std::vector<idx_t>& gt_labels) {
        float filtered_k = 0;
        for (auto gt_label : gt_labels) {
            if (gt_label != -1) {
                filtered_k++;
            }
        }

        float correct = 0;
        for (auto label : labels) {
            if (label == -1) {
                continue;
            }
            if (std::find(gt_labels.begin(), gt_labels.end(), label) !=
                gt_labels.end()) {
                correct++;
            }
        }

        return correct / filtered_k;
    };

    // search with index
    int k = 10;
    std::vector<float> distances(k);
    std::vector<idx_t> labels(k);
    std::vector<float> gt_distances(k);
    std::vector<idx_t> gt_labels(k);
    std::vector<float> recall;

    for (auto qv_idx : sampled_vids) {
        auto tid = tenant_ids[qv_idx];
        auto vec = vecs.data() + qv_idx * nd;
        index.search(1, vec, k, tid, distances.data(), labels.data());

        // sanity check: all returned vectors are accessible to the tenant
        for (auto label : labels) {
            if (label != -1) {
                EXPECT_EQ(tenant_ids[label], tid);
            }
        }

        tenant_indexes[tid]->search(
                1, vec, k, gt_distances.data(), gt_labels.data());
        for (int i = 0; i < k; i++) {
            if (gt_labels[i] != -1) {
                gt_labels[i] = lidx_to_gidx[tid][gt_labels[i]];
            }
        }

        recall.push_back(topk_recall(k, labels, gt_labels));
    }

    // compute average recall
    float avg_recall = 0;
    for (auto r : recall) {
        avg_recall += r;
    }
    avg_recall /= recall.size();

    std::cout << "Average recall: " << avg_recall << std::endl;
    EXPECT_GE(avg_recall, 0.9);
}
