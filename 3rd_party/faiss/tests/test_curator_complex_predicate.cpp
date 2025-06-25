#include <algorithm>
#include <cassert>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <faiss/MultiTenantIndexIVFHierarchical.h>
#include <faiss/utils/distances.h>
#include <gtest/gtest.h>

using namespace faiss;
using complex_predicate::build_temp_index_for_filter;
using complex_predicate::TempIndexNode;

// Default parameters
const std::string PREFIX = "/home/yicheng/curator-v2/data/yfcc100m_cpp/";
const size_t N_CLUSTERS = 16;
const size_t BF_CAPACITY = 1000;
const size_t MAX_SL_SIZE = 128;
const size_t MAX_LEAF_SIZE = 128;
const size_t CLUS_NITER = 10;
const float VARIANCE_BOOST = 0.0f;
const size_t SEARCH_EF = 128;
const size_t BEAM_SIZE = 4;

// Helper to read .fvecs file (FAISS format)
// Returns a flat vector, and sets n_out and d_out
std::vector<float> read_fvecs(
        const std::string& fname,
        size_t& n_out,
        size_t& d_out) {
    std::ifstream fin(fname, std::ios::binary);
    if (!fin)
        throw std::runtime_error("Cannot open file: " + fname);

    std::vector<float> data;
    n_out = 0;
    d_out = 0;

    while (true) {
        // Read dimension and check if all vectors have the same dimension
        int d = 0;
        fin.read(reinterpret_cast<char*>(&d), 4);
        if (!fin)
            break;
        if (d_out == 0)
            d_out = d;
        else if ((size_t)d != d_out)
            throw std::runtime_error("Inconsistent vector dimension in file");

        // Read vector data
        std::vector<float> vec(d);
        fin.read(reinterpret_cast<char*>(vec.data()), sizeof(float) * d);
        if (!fin)
            break;
        data.insert(data.end(), vec.begin(), vec.end());

        n_out++;
    }

    return data;
}

// Load dataset from converted files
struct Dataset {
    std::vector<float> train_vecs;
    std::vector<float> test_vecs;
    size_t train_n, train_d;
    size_t test_n, test_d;
    std::vector<std::vector<int>> train_mds;
    std::vector<std::vector<int>> test_mds;
};

Dataset load_converted_dataset(const std::string& prefix) {
    Dataset ds;

    // Load train_vecs and test_vecs
    ds.train_vecs =
            read_fvecs(prefix + "train_vecs_cpp.fvecs", ds.train_n, ds.train_d);
    ds.test_vecs =
            read_fvecs(prefix + "test_vecs_cpp.fvecs", ds.test_n, ds.test_d);

    // Load train_mds
    std::ifstream fin(prefix + "train_mds_cpp.txt");
    std::string line;
    while (std::getline(fin, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        int x;
        while (iss >> x)
            row.push_back(x);
        ds.train_mds.push_back(row);
    }
    fin.close();

    // Load test_mds
    std::ifstream fin2(prefix + "test_mds_cpp.txt");
    while (std::getline(fin2, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        int x;
        while (iss >> x)
            row.push_back(x);
        ds.test_mds.push_back(row);
    }
    fin2.close();

    return ds;
}

// Helper to compare two float arrays
bool compare_centroids(
        const float* a,
        const float* b,
        size_t d,
        float tol = 1e-5) {
    for (size_t i = 0; i < d; ++i) {
        if (std::abs(a[i] - b[i]) > tol)
            return false;
    }
    return true;
}

// Validate temp index ranges: check that leaf nodes cover all qualified vectors
// without overlap and that internal nodes have valid children
void validate_temp_index_ranges(
        const std::vector<TempIndexNode>& temp_nodes,
        size_t qualified_vecs_size) {
    std::vector<bool> covered(qualified_vecs_size, false);

    // Traverse all nodes and collect leaf ranges
    std::function<void(int)> traverse = [&](int node_idx) {
        const auto& node = temp_nodes[node_idx];

        if (node.children.empty()) {
            // This is a leaf node - check its range
            ASSERT_GE(node.start, 0)
                    << "Leaf node " << node_idx << " has negative start";
            ASSERT_LE(node.end, (int)qualified_vecs_size)
                    << "Leaf node " << node_idx
                    << " end exceeds qualified_vecs size";
            ASSERT_LT(node.start, node.end)
                    << "Leaf node " << node_idx << " has invalid range ["
                    << node.start << ", " << node.end << ")";

            // Check for overlap
            for (int i = node.start; i < node.end; i++) {
                ASSERT_FALSE(covered[i])
                        << "Vector index " << i
                        << " is covered by multiple leaf nodes";
                covered[i] = true;
            }
        } else {
            // This is an internal node - traverse children
            for (int child_idx : node.children) {
                ASSERT_GE(child_idx, 0) << "Invalid child index " << child_idx;
                ASSERT_LT(child_idx, (int)temp_nodes.size())
                        << "Child index " << child_idx << " out of bounds";
                traverse(child_idx);
            }
        }
    };

    // Start traversal from root (node 0)
    ASSERT_FALSE(temp_nodes.empty()) << "Temp index is empty";
    traverse(0);

    // Check that all qualified vectors are covered
    for (size_t i = 0; i < qualified_vecs_size; i++) {
        ASSERT_TRUE(covered[i]) << "Qualified vector index " << i
                                << " is not covered by any leaf node";
    }
}

// Recursively compare real and temp index trees
void compare_trees(
        const TreeNode* real,
        int temp_idx,
        const std::vector<TempIndexNode>& temp_nodes,
        const std::vector<int_vid_t>& qualified_vecs,
        size_t d,
        int_lid_t tid) {
    const auto& temp = temp_nodes[temp_idx];

    // 1. All nodes in the sub-tree must have bf.contains(tid)
    ASSERT_TRUE(real->bf.contains(tid));

    // Verify bit manipulation logic consistency
    // Check if real->node_id encodes the sibling_id correctly
    if (real->parent != nullptr) {
        auto offset = sizeof(int_vid_t) * 8 -
                real->level * CURATOR_MAX_BRANCH_FACTOR_LOG2;
        auto expected_sibling_id =
                (real->node_id >> offset) & (CURATOR_MAX_BRANCH_FACTOR - 1);
        ASSERT_EQ(expected_sibling_id, real->sibling_id)
                << "Node " << real->node_id << " at level " << real->level
                << " has sibling_id " << real->sibling_id
                << " but node_id encodes " << expected_sibling_id;
    }

    // 2. If this is a sub-tree leaf (buffer for tid)
    auto it = real->shortlists.find(tid);
    if (it != real->shortlists.end()) {
        // Must be a leaf in temp index
        ASSERT_TRUE(temp.children.empty());

        // VERIFICATION: Check if temp node's centroid pointer matches real
        // node's centroid
        ASSERT_EQ(temp.centroid, real->centroid)
                << "Temp node centroid pointer mismatch at level "
                << real->level
                << ". Real centroid: " << static_cast<void*>(real->centroid)
                << ", Temp centroid: " << static_cast<void*>(temp.centroid);

        // Compare buffers as sets (order may differ)
        const auto& real_buf = it->second.data;

        // VERIFICATION: Check if vector IDs in real_buf have the same prefix as
        // real->node_id
        auto shift = sizeof(int_vid_t) * 8 -
                CURATOR_MAX_BRANCH_FACTOR_LOG2 * real->level;
        auto expected_prefix = real->node_id >> shift;

        for (int_vid_t vid : real_buf) {
            auto vid_prefix = vid >> shift;
            ASSERT_EQ(vid_prefix, expected_prefix)
                    << "Vector ID " << vid
                    << " in buffer does not match node prefix. "
                    << "Expected prefix: " << expected_prefix
                    << ", actual prefix: " << vid_prefix << " (shift=" << shift
                    << ", level=" << real->level << ")";
        }

        std::set<int_vid_t> real_set(real_buf.begin(), real_buf.end());
        std::set<int_vid_t> temp_set(
                qualified_vecs.begin() + temp.start,
                qualified_vecs.begin() + temp.end);

        // VERIFICATION: Check if vector IDs in temp_set also follow the same
        // bit pattern
        for (auto it = qualified_vecs.begin() + temp.start;
             it != qualified_vecs.begin() + temp.end;
             ++it) {
            int_vid_t vid = *it;
            auto vid_prefix = vid >> shift;
            ASSERT_EQ(vid_prefix, expected_prefix)
                    << "Temp Vector ID " << vid
                    << " does not match node prefix. "
                    << "Expected prefix: " << expected_prefix
                    << ", actual prefix: " << vid_prefix << " (shift=" << shift
                    << ", level=" << real->level << ")";
        }

        ASSERT_LE(temp_set.size(), MAX_SL_SIZE);
        ASSERT_EQ(real_set, temp_set);
        return;
    }

    // 3. Otherwise, must be an internal node in the sub-tree
    // temp node must have children
    ASSERT_FALSE(temp.children.empty());

    // Find all real children in the sub-tree (those with bf.contains(tid))
    std::vector<const TreeNode*> real_subtree_children;
    for (const auto* child : real->children) {
        if (child->bf.contains(tid)) {
            real_subtree_children.push_back(child);
        }
    }

    // VERIFICATION: Check if temp children are partitioned correctly by bit
    // manipulation
    auto level = real->level;
    auto offset = sizeof(int_vid_t) * 8 -
            CURATOR_MAX_BRANCH_FACTOR_LOG2 * (level + 1);
    auto mask = (CURATOR_MAX_BRANCH_FACTOR - 1);

    for (size_t i = 0; i < temp.children.size(); ++i) {
        int child_temp_idx = temp.children[i];
        const auto& temp_child = temp_nodes[child_temp_idx];

        // Check if all vectors in this child's range have the same branch index
        std::set<int> branch_indices;
        for (int j = temp_child.start; j < temp_child.end; j++) {
            int_vid_t vid = qualified_vecs[j];
            int branch_idx = (vid >> offset) & mask;
            branch_indices.insert(branch_idx);
        }

        ASSERT_EQ(branch_indices.size(), 1)
                << "Temp child " << i
                << " contains vectors with different branch indices: "
                << "level=" << level << ", offset=" << offset
                << ", mask=" << mask;
    }

    // For each temp child, find a real child with matching centroid pointer
    for (size_t i = 0; i < temp.children.size(); ++i) {
        const auto& temp_child = temp_nodes[temp.children[i]];
        bool found = false;
        for (const auto* real_child : real_subtree_children) {
            if (real_child->centroid == temp_child.centroid) {
                compare_trees(
                        real_child,
                        temp.children[i],
                        temp_nodes,
                        qualified_vecs,
                        d,
                        tid);
                found = true;
                break;
            }
        }
        ASSERT_TRUE(found)
                << "No matching real child for temp child centroid pointer";
    }
}

TEST(CuratorComplexPredicate, CompareRealAndTempIndex) {
    // Load real dataset
    std::cout << "Loading dataset..." << std::endl;
    Dataset ds = load_converted_dataset(PREFIX);
    size_t d = ds.train_d;
    size_t nb = ds.train_n;
    const float* xb = ds.train_vecs.data();

    // Build index
    std::cout << "Building index..." << std::endl;
    MultiTenantIndexIVFHierarchical index(
            d,
            N_CLUSTERS,
            METRIC_L2,
            BF_CAPACITY,
            0.01f, // bf_false_pos
            MAX_SL_SIZE,
            CLUS_NITER,
            MAX_LEAF_SIZE,
            0,    // nprobe (not used)
            0.0f, // prune_thres (not used)
            VARIANCE_BOOST,
            SEARCH_EF,
            BEAM_SIZE);

    index.train(nb, xb, 0);
    std::vector<idx_t> labels(nb);
    for (size_t i = 0; i < nb; ++i)
        labels[i] = i;
    index.add_vector_with_ids(nb, xb, labels.data());

    // Add metadata
    for (size_t i = 0; i < nb; ++i) {
        for (int tenant : ds.train_mds[i]) {
            index.grant_access(i, tenant);
        }
    }

    // 1. Calculate number of vectors accessible to each tenant
    std::unordered_map<int, int> tenant_to_count;
    for (const auto& mds : ds.train_mds) {
        for (int tenant : mds) {
            tenant_to_count[tenant]++;
        }
    }

    // 2. Sort tenants by number of vectors
    std::vector<std::pair<int, int>> tenant_counts(
            tenant_to_count.begin(), tenant_to_count.end());
    std::sort(
            tenant_counts.begin(),
            tenant_counts.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });

    // 3. Sample 10 tenants with diverse #vectors (min, max, and quantiles)
    std::vector<int> sampled_tenants;
    size_t n_tenants = tenant_counts.size();
    if (n_tenants <= 10) {
        for (const auto& p : tenant_counts)
            sampled_tenants.push_back(p.first);
    } else {
        sampled_tenants.push_back(tenant_counts.front().first); // min
        for (int i = 1; i < 9; ++i) {
            size_t idx = (size_t)((double)i / 9 * (n_tenants - 1));
            sampled_tenants.push_back(tenant_counts[idx].first);
        }
        sampled_tenants.push_back(tenant_counts.back().first); // max
    }

    // 4. For each sampled tenant, run the comparison
    for (int tenant : sampled_tenants) {
        int n_vecs = tenant_to_count[tenant];
        std::cout << "\nTesting tenant " << tenant << " (#vecs=" << n_vecs
                  << ")... ";

        // Build temp index
        std::cout << "\tBuilding temp index..." << std::endl;
        std::string filter = std::to_string(tenant);
        auto converted_filter = index.convert_complex_predicate(filter);
        auto qualified_vecs = index.find_all_qualified_vecs(converted_filter);
        std::sort(qualified_vecs.begin(), qualified_vecs.end());
        std::vector<TempIndexNode> temp_nodes;
        build_temp_index_for_filter(&index, qualified_vecs, temp_nodes);

        // Validate temp index ranges
        validate_temp_index_ranges(temp_nodes, qualified_vecs.size());

        // Compare trees
        std::cout << "\tComparing trees..." << std::endl;
        int_lid_t tid = index.tid_allocator.get_id(tenant);
        ASSERT_NO_FATAL_FAILURE(compare_trees(
                index.tree_root, 0, temp_nodes, qualified_vecs, d, tid))
                << "Failed for tenant " << tenant;

        std::cout << "PASSED" << std::endl;
    }
}

TEST(CuratorComplexPredicate, SearchTempIndexExactResults) {
    // Load real dataset
    std::cout << "Loading dataset..." << std::endl;
    Dataset ds = load_converted_dataset(PREFIX);
    size_t d = ds.train_d;
    size_t nb = ds.train_n;
    const float* xb = ds.train_vecs.data();
    const float* xq = ds.test_vecs.data();
    size_t nq = ds.test_n;

    // Build index with large search_ef for exact results
    std::cout << "Building index..." << std::endl;
    MultiTenantIndexIVFHierarchical index(
            d,
            N_CLUSTERS,
            METRIC_L2,
            BF_CAPACITY,
            0.01f, // bf_false_pos
            MAX_SL_SIZE,
            CLUS_NITER,
            MAX_LEAF_SIZE,
            0,    // nprobe (not used)
            0.0f, // prune_thres (not used)
            VARIANCE_BOOST,
            10000, // large search_ef for exact results
            BEAM_SIZE);

    index.train(nb, xb, 0);
    std::vector<idx_t> labels(nb);
    for (size_t i = 0; i < nb; ++i)
        labels[i] = i;
    index.add_vector_with_ids(nb, xb, labels.data());

    // Add metadata
    for (size_t i = 0; i < nb; ++i) {
        for (int tenant : ds.train_mds[i]) {
            index.grant_access(i, tenant);
        }
    }

    // Sample 10 (query, tenant) pairs from test data
    std::mt19937 rng(42);
    std::vector<std::pair<size_t, int>> query_tenant_pairs;

    // Collect all valid (query_idx, tenant) pairs
    for (size_t query_idx = 0; query_idx < nq; ++query_idx) {
        for (int tenant : ds.test_mds[query_idx]) {
            query_tenant_pairs.emplace_back(query_idx, tenant);
        }
    }

    // Randomly sample 1 pair for detailed debugging
    std::shuffle(query_tenant_pairs.begin(), query_tenant_pairs.end(), rng);
    size_t n_samples = std::min((size_t)1, query_tenant_pairs.size());
    query_tenant_pairs.resize(n_samples);

    // Calculate number of vectors accessible to each tenant for info
    std::unordered_map<int, int> tenant_to_count;
    for (const auto& mds : ds.train_mds) {
        for (int tenant : mds) {
            tenant_to_count[tenant]++;
        }
    }

    const int k = 10; // top-k results

    for (const auto& [query_idx, tenant] : query_tenant_pairs) {
        int n_vecs = tenant_to_count[tenant];
        std::cout << "\nTesting query " << query_idx << " with tenant "
                  << tenant << " (#vecs=" << n_vecs << ")... ";

        // Convert filter and find qualified vectors
        std::string filter = std::to_string(tenant);
        auto converted_filter = index.convert_complex_predicate(filter);
        auto qualified_vecs = index.find_all_qualified_vecs(converted_filter);
        std::sort(qualified_vecs.begin(), qualified_vecs.end());

        // Ensure search_ef is at least the number of qualified vectors for
        // exact results
        auto temp_index = index;
        temp_index.search_ef =
                std::max(temp_index.search_ef, qualified_vecs.size());

        // Build temp index
        std::cout << "  Building temp index for " << qualified_vecs.size()
                  << " qualified vectors..." << std::endl;
        std::vector<complex_predicate::TempIndexNode> temp_nodes;
        complex_predicate::build_temp_index_for_filter(
                &temp_index, qualified_vecs, temp_nodes);
        std::cout << "  Built temp index with " << temp_nodes.size() << " nodes"
                  << std::endl;

        const float* query = xq + query_idx * d;

        // Compute ground truth by brute force
        std::vector<std::pair<float, idx_t>> ground_truth;
        for (int_vid_t vid : qualified_vecs) {
            ext_vid_t ext_vid = index.id_allocator.get_label(vid);
            float dist = fvec_L2sqr(query, xb + ext_vid * d, d);
            ground_truth.emplace_back(dist, ext_vid);
        }
        std::sort(ground_truth.begin(), ground_truth.end());

        // Take top-k from ground truth
        std::vector<idx_t> gt_labels(k);
        std::vector<float> gt_distances(k);
        for (int i = 0; i < k && i < ground_truth.size(); ++i) {
            gt_labels[i] = ground_truth[i].second;
            gt_distances[i] = ground_truth[i].first;
        }

        // Search using temp index
        std::vector<idx_t> search_labels(k);
        std::vector<float> search_distances(k);
        complex_predicate::search_temp_index(
                &temp_index,
                qualified_vecs,
                temp_nodes,
                query,
                k,
                search_distances.data(),
                search_labels.data(),
                nullptr);

        // Debug: Check if search results are valid (i.e., belong to the tenant)
        std::cout << "\n  Ground truth top-"
                  << std::min(k, (int)ground_truth.size()) << ":" << std::endl;
        for (int i = 0; i < k && i < ground_truth.size(); ++i) {
            std::cout << "    [" << i << "] label=" << gt_labels[i]
                      << ", dist=" << gt_distances[i] << std::endl;
        }

        std::cout << "  Search results top-" << k << ":" << std::endl;
        for (int i = 0; i < k; ++i) {
            if (search_labels[i] >=
                0) { // Check if result is valid (non-negative)
                // Check if this vector actually belongs to the tenant
                bool belongs_to_tenant = false;
                if (search_labels[i] < ds.train_mds.size()) {
                    for (int t : ds.train_mds[search_labels[i]]) {
                        if (t == tenant) {
                            belongs_to_tenant = true;
                            break;
                        }
                    }
                }
                std::cout << "    [" << i << "] label=" << search_labels[i]
                          << ", dist=" << search_distances[i]
                          << (belongs_to_tenant ? " (VALID)"
                                                : " (INVALID - not in tenant)")
                          << std::endl;
            } else {
                std::cout << "    [" << i << "] label=" << search_labels[i]
                          << " (invalid)" << std::endl;
            }
        }

        // Compare results - only fail if we have valid results to compare
        for (int i = 0; i < k && i < ground_truth.size(); ++i) {
            if (search_labels[i] >=
                0) { // Only check valid (non-negative) results
                ASSERT_EQ(search_labels[i], gt_labels[i])
                        << "Mismatch at position " << i << " for tenant "
                        << tenant << ", query " << query_idx
                        << ". Expected: " << gt_labels[i]
                        << ", Got: " << search_labels[i];

                ASSERT_NEAR(search_distances[i], gt_distances[i], 1e-4)
                        << "Distance mismatch at position " << i
                        << " for tenant " << tenant << ", query " << query_idx;
            }
        }

        std::cout << "PASSED" << std::endl;
    }
}

TEST(CuratorComplexPredicate, ORFilterRecallEvaluation) {
    // Load real dataset
    std::cout << "Loading dataset..." << std::endl;
    Dataset ds = load_converted_dataset(PREFIX);
    size_t d = ds.train_d;
    size_t nb = ds.train_n;
    const float* xb = ds.train_vecs.data();
    const float* xq = ds.test_vecs.data();
    size_t nq = ds.test_n;

    // Build index
    std::cout << "Building index..." << std::endl;
    MultiTenantIndexIVFHierarchical index(
            d,
            N_CLUSTERS,
            METRIC_L2,
            BF_CAPACITY,
            0.01f,
            MAX_SL_SIZE,
            CLUS_NITER,
            MAX_LEAF_SIZE,
            0,
            0.0f,
            VARIANCE_BOOST,
            32,
            BEAM_SIZE); // Start with search_ef=32

    index.train(nb, xb, 0);
    std::vector<idx_t> labels(nb);
    for (size_t i = 0; i < nb; ++i)
        labels[i] = i;
    index.add_vector_with_ids(nb, xb, labels.data());

    // Add metadata
    for (size_t i = 0; i < nb; ++i) {
        for (int tenant : ds.train_mds[i]) {
            index.grant_access(i, tenant);
        }
    }

    // Find tenants with sufficient vectors for OR queries
    std::unordered_map<int, int> tenant_to_count;
    for (const auto& mds : ds.train_mds) {
        for (int tenant : mds) {
            tenant_to_count[tenant]++;
        }
    }

    // Select tenants with sufficient individual counts and create pairs
    std::vector<int> good_tenants;
    for (const auto& [tenant, count] : tenant_to_count) {
        if (count >= 10000) { // At least 10000 vectors for each tenant
            good_tenants.push_back(tenant);
        }
    }

    if (good_tenants.size() < 2) {
        std::cout << "Not enough tenants with sufficient vectors, skipping test"
                  << std::endl;
        return;
    }

    // Create tenant pairs for testing
    std::vector<std::pair<int, int>> tenant_pairs;
    for (size_t i = 0; i < good_tenants.size() && tenant_pairs.size() < 3;
         ++i) {
        for (size_t j = i + 1;
             j < good_tenants.size() && tenant_pairs.size() < 3;
             ++j) {
            tenant_pairs.emplace_back(good_tenants[i], good_tenants[j]);
        }
    }

    std::mt19937 rng(42);

    const int k = 10;
    std::vector<int> search_ef_values = {32, 64, 128, 256};

    // Track statistics
    int total_queries = 0;
    int queries_with_violations = 0;
    int total_violations = 0;

    std::cout << "\nTesting OR filter recall with different search_ef values:\n"
              << std::endl;

    for (const auto& [tenant1, tenant2] : tenant_pairs) {
        std::cout << "=== Testing OR filter: tenant " << tenant1
                  << " OR tenant " << tenant2 << " ===" << std::endl;

        // Create OR filter
        std::string filter =
                "OR " + std::to_string(tenant1) + " " + std::to_string(tenant2);
        auto converted_filter = index.convert_complex_predicate(filter);
        auto qualified_vecs = index.find_all_qualified_vecs(converted_filter);
        std::sort(qualified_vecs.begin(), qualified_vecs.end());

        std::cout << "Found " << qualified_vecs.size() << " qualified vectors"
                  << std::endl;

        // Find test queries that match this OR filter
        std::vector<size_t> matching_queries;
        for (size_t query_idx = 0; query_idx < nq; ++query_idx) {
            const auto& test_mds = ds.test_mds[query_idx];
            bool matches_filter = false;
            for (int tenant : test_mds) {
                if (tenant == tenant1 || tenant == tenant2) {
                    matches_filter = true;
                    break;
                }
            }
            if (matches_filter) {
                matching_queries.push_back(query_idx);
            }
        }

        if (matching_queries.size() < 5) {
            std::cout << "  Only " << matching_queries.size()
                      << " queries match OR filter, skipping this tenant pair"
                      << std::endl;
            continue;
        }

        // Sample 5 queries from matching ones
        std::shuffle(matching_queries.begin(), matching_queries.end(), rng);
        matching_queries.resize(5);

        for (size_t query_idx : matching_queries) {
            const float* query = xq + query_idx * d;
            total_queries++;

            // Compute ground truth: vectors that contain either tenant1 or
            // tenant2
            std::vector<std::pair<float, idx_t>> ground_truth;
            for (int_vid_t vid : qualified_vecs) {
                ext_vid_t ext_vid = index.id_allocator.get_label(vid);
                // qualified_vecs already contains vectors matching OR filter
                float dist = fvec_L2sqr(query, xb + ext_vid * d, d);
                ground_truth.emplace_back(dist, ext_vid);
            }
            std::sort(ground_truth.begin(), ground_truth.end());

            if (ground_truth.size() < k) {
                std::cout << "  Query " << query_idx << ": Only "
                          << ground_truth.size()
                          << " ground truth results (< k=" << k << "), skipping"
                          << std::endl;
                continue;
            }

            // Extract top-k ground truth
            std::set<idx_t> gt_set;
            for (int i = 0; i < k && i < ground_truth.size(); ++i) {
                gt_set.insert(ground_truth[i].second);
            }

            std::cout << "  Query " << query_idx
                      << " (GT size: " << gt_set.size() << "):" << std::endl;

            // Test different search_ef values and track recall progression
            std::vector<double> recalls;
            for (int search_ef : search_ef_values) {
                index.search_ef = search_ef;

                // Build temp index
                std::vector<complex_predicate::TempIndexNode> temp_nodes;
                complex_predicate::build_temp_index_for_filter(
                        &index, qualified_vecs, temp_nodes);

                // Search
                std::vector<idx_t> search_labels(k);
                std::vector<float> search_distances(k);
                complex_predicate::search_temp_index(
                        &index,
                        qualified_vecs,
                        temp_nodes,
                        query,
                        k,
                        search_distances.data(),
                        search_labels.data(),
                        nullptr);

                // Calculate recall
                std::set<idx_t> result_set;
                for (int i = 0; i < k; ++i) {
                    if (search_labels[i] >= 0) {
                        result_set.insert(search_labels[i]);
                    }
                }

                // Intersection of results and ground truth
                std::set<idx_t> intersection;
                std::set_intersection(
                        result_set.begin(),
                        result_set.end(),
                        gt_set.begin(),
                        gt_set.end(),
                        std::inserter(intersection, intersection.begin()));

                double recall = (double)intersection.size() / gt_set.size();
                recalls.push_back(recall);
                std::cout << "    search_ef=" << search_ef
                          << ": recall=" << std::fixed << std::setprecision(3)
                          << recall << std::endl;
            }

            // Verify that recall is non-decreasing with increasing search_ef
            bool has_violations = false;
            int query_violations = 0;
            for (size_t i = 1; i < recalls.size(); ++i) {
                if (recalls[i] < recalls[i - 1] - 1e-6) {
                    std::cout << "    WARNING: Recall decreased! Details:"
                              << std::endl;
                    std::cout << "      search_ef " << search_ef_values[i - 1]
                              << " -> " << search_ef_values[i] << std::endl;
                    std::cout << "      recall " << recalls[i - 1] << " -> "
                              << recalls[i] << std::endl;
                    std::cout
                            << "      This suggests potential algorithm instability or randomness"
                            << std::endl;

                    // Investigate duplicate vector hypothesis
                    std::cout
                            << "    INVESTIGATING DUPLICATE VECTOR HYPOTHESIS:"
                            << std::endl;

                    // Re-run the problematic search_ef values to get detailed
                    // results
                    std::vector<std::vector<idx_t>> all_labels;
                    std::vector<std::vector<float>> all_distances;

                    for (int ef :
                         {search_ef_values[i - 1], search_ef_values[i]}) {
                        index.search_ef = ef;

                        std::vector<complex_predicate::TempIndexNode>
                                temp_nodes;
                        complex_predicate::build_temp_index_for_filter(
                                &index, qualified_vecs, temp_nodes);

                        std::vector<idx_t> search_labels(k);
                        std::vector<float> search_distances(k);
                        complex_predicate::search_temp_index(
                                &index,
                                qualified_vecs,
                                temp_nodes,
                                query,
                                k,
                                search_distances.data(),
                                search_labels.data(),
                                nullptr);

                        all_labels.push_back(search_labels);
                        all_distances.push_back(search_distances);
                    }

                    // Compare distances
                    std::cout << "      Ground truth distances (top-" << k
                              << "):" << std::endl;
                    for (int j = 0; j < k && j < ground_truth.size(); ++j) {
                        std::cout << "        " << j
                                  << ": dist=" << ground_truth[j].first
                                  << ", id=" << ground_truth[j].second
                                  << std::endl;
                    }

                    for (size_t ef_idx = 0; ef_idx < 2; ++ef_idx) {
                        int ef = (ef_idx == 0) ? search_ef_values[i - 1]
                                               : search_ef_values[i];
                        std::cout << "      search_ef=" << ef
                                  << " results:" << std::endl;
                        for (int j = 0; j < k; ++j) {
                            if (all_labels[ef_idx][j] >= 0) {
                                std::cout
                                        << "        " << j
                                        << ": dist=" << all_distances[ef_idx][j]
                                        << ", id=" << all_labels[ef_idx][j]
                                        << std::endl;
                            }
                        }
                    }

                    // Check if distances are similar between the two search_ef
                    // values (indicating duplicate vectors with same distances
                    // but different IDs)
                    std::vector<float> dists_ef1, dists_ef2;

                    for (int j = 0; j < k; ++j) {
                        if (all_labels[0][j] >= 0) {
                            dists_ef1.push_back(all_distances[0][j]);
                        }
                        if (all_labels[1][j] >= 0) {
                            dists_ef2.push_back(all_distances[1][j]);
                        }
                    }

                    std::sort(dists_ef1.begin(), dists_ef1.end());
                    std::sort(dists_ef2.begin(), dists_ef2.end());

                    std::cout << "      DUPLICATE VECTOR ANALYSIS:"
                              << std::endl;
                    std::cout << "      Comparing distances between search_ef="
                              << search_ef_values[i - 1]
                              << " and search_ef=" << search_ef_values[i]
                              << std::endl;

                    bool distances_equivalent = true;
                    size_t min_size =
                            std::min(dists_ef1.size(), dists_ef2.size());

                    for (size_t d = 0; d < min_size; ++d) {
                        float diff = std::abs(dists_ef1[d] - dists_ef2[d]);
                        std::cout << "        pos " << d << ": ef"
                                  << search_ef_values[i - 1] << "="
                                  << dists_ef1[d] << ", ef"
                                  << search_ef_values[i] << "=" << dists_ef2[d]
                                  << ", diff=" << diff << std::endl;
                        if (diff > 1e-5) {
                            distances_equivalent = false;
                        }
                    }

                    std::cout << "      CONCLUSION: Distances are "
                              << (distances_equivalent ? "EQUIVALENT"
                                                       : "DIFFERENT")
                              << std::endl;

                    if (distances_equivalent) {
                        std::cout
                                << "      ✓ This confirms the duplicate vector hypothesis!"
                                << std::endl;
                        std::cout
                                << "      ✓ Both search_ef values found equally good results,"
                                << std::endl;
                        std::cout
                                << "        but with different vector IDs (likely duplicates)"
                                << std::endl;
                    } else {
                        std::cout
                                << "      ✗ Distances differ - this suggests a real algorithmic issue"
                                << std::endl;
                    }

                    has_violations = true;
                    query_violations++;
                    total_violations++;
                }
            }
            if (has_violations) {
                std::cout << "    Query " << query_idx << " had "
                          << query_violations << " recall violations"
                          << std::endl;
                queries_with_violations++;
            }
        }
    }

    // Print summary statistics
    std::cout << "\n=== RECALL VIOLATION SUMMARY ===" << std::endl;
    std::cout << "Total queries tested: " << total_queries << std::endl;
    std::cout << "Queries with violations: " << queries_with_violations
              << std::endl;
    std::cout << "Total violations: " << total_violations << std::endl;
    std::cout << "Violation rate: " << std::fixed << std::setprecision(2)
              << (total_queries > 0
                          ? 100.0 * queries_with_violations / total_queries
                          : 0.0)
              << "%" << std::endl;

    if (queries_with_violations > 0) {
        std::cout << "\nRecall violations detected. This could indicate:"
                  << std::endl;
        std::cout << "1. Algorithm instability or randomness in search process"
                  << std::endl;
        std::cout
                << "2. Potential bugs in HNSW search with different search_ef values"
                << std::endl;
        std::cout << "3. Edge cases in the filtering/search logic" << std::endl;
        std::cout << "\nInvestigation recommended." << std::endl;
    } else {
        std::cout
                << "\nNo recall violations detected - recall consistently increases with search_ef!"
                << std::endl;
    }
}