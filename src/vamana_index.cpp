#include "vamana_index.h"
#include "distance.h"
#include "io_utils.h"
#include "timer.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <cstdlib>

// ============================================================================
// Destructor
// ============================================================================

VamanaIndex::~VamanaIndex() {
    if (owns_data_ && data_) {
        std::free(data_);
        data_ = nullptr;
    }
}

// ============================================================================
// Greedy Search
// ============================================================================
// Beam search starting from start_node_. Maintains a candidate set of at most
// L nodes, always expanding the closest unvisited node. Returns when no
// unvisited candidates remain.
//
// Uses std::set<Candidate> as an ordered container — simple, correct, and
// easy for students to understand and modify.

std::pair<std::vector<VamanaIndex::Candidate>, uint32_t>
VamanaIndex::greedy_search(const float* query, uint32_t L, uint32_t s_node) const {
    // Candidate set: ordered by (distance, id). Bounded at size L.
    std::set<Candidate> candidate_set;
    // Track which nodes we've already expanded (visited).
    std::vector<bool> visited(npts_, false);

    uint32_t dist_cmps = 0;

    uint32_t act_start = (s_node == UINT32_MAX) ? start_node_ : s_node;

    // Seed with start node
    float start_dist = compute_l2sq(query, get_vector(act_start), dim_);
    dist_cmps++;
    candidate_set.insert({start_dist, act_start});
    visited[act_start] = true;

    // Track which candidates have been expanded (their neighbors explored).
    // We iterate through candidate_set; entries before our "frontier" pointer
    // have been expanded. We use a simple approach: keep scanning from the
    // beginning of the set for the first un-expanded entry.
    std::set<uint32_t> expanded;

    while (true) {
        // Find closest candidate that hasn't been expanded yet
        uint32_t best_node = UINT32_MAX;
        for (const auto& [dist, id] : candidate_set) {
            if (expanded.find(id) == expanded.end()) {
                best_node = id;
                break;
            }
        }
        if (best_node == UINT32_MAX)
            break;  // all candidates expanded

        expanded.insert(best_node);

        // Expand: evaluate all neighbors of best_node
        // Copy neighbor list under lock to avoid data race with parallel build
        // (another thread might push_back / reallocate graph_[best_node]).
        std::vector<uint32_t> neighbors;
        {
            std::lock_guard<std::mutex> lock(locks_[best_node]);
            neighbors = graph_[best_node];
        }
        for (uint32_t nbr : neighbors) {
            if (visited[nbr])
                continue;
            visited[nbr] = true;

            float d = compute_l2sq(query, get_vector(nbr), dim_);
            dist_cmps++;

            // Insert if candidate set isn't full or this is closer than worst
            if (candidate_set.size() < L) {
                candidate_set.insert({d, nbr});
            } else {
                auto worst = std::prev(candidate_set.end());
                if (d < worst->first) {
                    candidate_set.erase(worst);
                    candidate_set.insert({d, nbr});
                }
            }
        }
    }

    // Convert to sorted vector
    std::vector<Candidate> results(candidate_set.begin(), candidate_set.end());
    return {results, dist_cmps};
}

// ============================================================================
// Robust Prune (Alpha-RNG Rule)
// ============================================================================
// Given a node and a set of candidates, greedily select neighbors that are
// "diverse" — a candidate c is added only if it's not too close to any
// already-selected neighbor (within a factor of alpha).
//
// Formally: add c if for ALL already-chosen neighbors n:
//     dist(node, c) <= alpha * dist(c, n)
//
// This ensures good graph navigability by keeping some long-range edges
// (alpha > 1 makes it easier for a candidate to survive pruning).

void VamanaIndex::robust_prune(uint32_t node, std::vector<Candidate>& candidates,
                               float alpha, uint32_t R) {
    // Remove self from candidates if present
    candidates.erase(
        std::remove_if(candidates.begin(), candidates.end(),
                       [node](const Candidate& c) { return c.second == node; }),
        candidates.end());

    // Sort by distance to node (ascending)
    std::sort(candidates.begin(), candidates.end());

    std::vector<uint32_t> new_neighbors;
    new_neighbors.reserve(R);

    for (const auto& [dist_to_node, cand_id] : candidates) {
        if (new_neighbors.size() >= R)
            break;

        // Check alpha-RNG condition against all already-selected neighbors
        bool keep = true;
        for (uint32_t selected : new_neighbors) {
            float dist_cand_to_selected =
                compute_l2sq(get_vector(cand_id), get_vector(selected), dim_);
            if (dist_to_node > alpha * dist_cand_to_selected) {
                keep = false;
                break;
            }
        }

        if (keep)
            new_neighbors.push_back(cand_id);
    }

    graph_[node] = std::move(new_neighbors);
}

// ============================================================================
// Build
// ============================================================================

void VamanaIndex::build(const std::string& data_path, uint32_t R, uint32_t L,
                        float alpha, float gamma, uint32_t L_target) {
    // --- Load data ---
    std::cout << "Loading data from " << data_path << "..." << std::endl;
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    std::cout << "  Points: " << npts_ << ", Dimensions: " << dim_ << std::endl;

    if (L < R) {
        std::cerr << "Warning: L (" << L << ") < R (" << R
                  << "). Setting L = R." << std::endl;
        L = R;
    }

    // --- Initialize empty graph and per-node locks ---
    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    // --- Pick random start node ---
    std::mt19937 rng(42);  // fixed seed for reproducibility
    start_node_ = rng() % npts_;
    std::cout << "  Start node: " << start_node_ << std::endl;

    // --- Create random insertion order ---
    std::vector<uint32_t> perm(npts_);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    // --- Build graph: parallel insertion with per-node locking ---
    uint32_t gamma_R = static_cast<uint32_t>(gamma * R);
    std::cout << "Building index (R=" << R << ", L=" << L
              << ", alpha=" << alpha << ", gamma=" << gamma
              << ", gammaR=" << gamma_R << ")..." << std::endl;

    Timer build_timer;

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t idx = 0; idx < npts_; idx++) {
        uint32_t point = perm[idx];

        // Step 1: Search for this point in the current graph to find candidates
        auto [candidates, _dist_cmps] = greedy_search(get_vector(point), L);

        // Step 2: Prune candidates to get this point's neighbors
        // We don't need to lock graph_[point] here because each point appears
        // exactly once in the permutation — only this thread writes to it now.
        robust_prune(point, candidates, alpha, R);

        // Step 3: Add backward edges from each new neighbor back to this point
        for (uint32_t nbr : graph_[point]) {
            std::lock_guard<std::mutex> lock(locks_[nbr]);

            // Add backward edge
            graph_[nbr].push_back(point);

            // Step 4: If neighbor's degree exceeds gamma*R, prune its neighborhood
            if (graph_[nbr].size() > gamma_R) {
                // Build candidate list from current neighbors of nbr
                std::vector<Candidate> nbr_candidates;
                nbr_candidates.reserve(graph_[nbr].size());
                for (uint32_t nn : graph_[nbr]) {
                    float d = compute_l2sq(get_vector(nbr), get_vector(nn), dim_);
                    nbr_candidates.push_back({d, nn});
                }
                robust_prune(nbr, nbr_candidates, alpha, R);
            }
        }

        // Progress reporting (from one thread only)
        if (idx % 10000 == 0) {
            #pragma omp critical
            {
                std::cout << "\r  Inserted " << idx << " / " << npts_
                          << " points" << std::flush;
            }
        }
    }

    double build_time = build_timer.elapsed_seconds();

    // Compute average degree
    size_t total_edges = 0;
    for (uint32_t i = 0; i < npts_; i++)
        total_edges += graph_[i].size();
    double avg_degree = (double)total_edges / npts_;

    std::cout << "\n  Build complete in " << build_time << " seconds."
              << std::endl;
    std::cout << "  Average out-degree: " << avg_degree << std::endl;

    // ========================================================================
    // Step 1.1: Data Sampling & K-Medoids for Spatial Variance L Allocation
    // ========================================================================
    std::cout << "  Starting Dynamic L Allocation via Spatial Variance (K=16)..." << std::endl;
    uint32_t K_medoids = 16;
    uint32_t sample_size = std::max(1u, (uint32_t)(npts_ * 0.05)); // 5% sample
    
    std::vector<uint32_t> dataset_ids(npts_);
    std::iota(dataset_ids.begin(), dataset_ids.end(), 0);
    std::shuffle(dataset_ids.begin(), dataset_ids.end(), rng);
    std::vector<uint32_t> sampled_ids(dataset_ids.begin(), dataset_ids.begin() + sample_size);

    // Initialize 16 medoids randomly from sample
    std::vector<uint32_t> medoid_ids(K_medoids);
    for (uint32_t i = 0; i < K_medoids; ++i) {
        medoid_ids[i] = sampled_ids[i]; // First K points as initial medoids
    }

    std::vector<std::vector<float>> cluster_means(K_medoids, std::vector<float>(dim_, 0.0f));
    std::vector<uint32_t> cluster_assignments(sample_size, 0);
    std::vector<uint32_t> cluster_sizes(K_medoids, 0);
    
    int max_iters = 20;
    for (int iter = 0; iter < max_iters; ++iter) {
        // Step 1.2: Cluster Assignment
        std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
        for(auto& mean : cluster_means) {
            std::fill(mean.begin(), mean.end(), 0.0f);
        }

        bool changed = false;
        for (uint32_t i = 0; i < sample_size; ++i) {
            uint32_t pt_id = sampled_ids[i];
            float min_dist = std::numeric_limits<float>::max();
            uint32_t best_k = 0;
            for (uint32_t k = 0; k < K_medoids; ++k) {
                float dist = compute_l2sq(get_vector(pt_id), get_vector(medoid_ids[k]), dim_);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_k = k;
                }
            }
            if (cluster_assignments[i] != best_k) {
                cluster_assignments[i] = best_k;
                changed = true;
            }
            cluster_sizes[best_k]++;
            const float* pt_vec = get_vector(pt_id);
            for(uint32_t d = 0; d < dim_; ++d) {
                cluster_means[best_k][d] += pt_vec[d];
            }
        }

        if (!changed) break;

        // Medoid update (Snap to closest sampled point to the mean)
        for (uint32_t k = 0; k < K_medoids; ++k) {
            if (cluster_sizes[k] > 0) {
                for (uint32_t d = 0; d < dim_; ++d) {
                    cluster_means[k][d] /= cluster_sizes[k];
                }
                
                float min_mean_dist = std::numeric_limits<float>::max();
                uint32_t new_medoid_id = medoid_ids[k];
                for (uint32_t i = 0; i < sample_size; ++i) {
                    if (cluster_assignments[i] == k) {
                        uint32_t pt_id = sampled_ids[i];
                        float dist = compute_l2sq(get_vector(pt_id), cluster_means[k].data(), dim_);
                        if (dist < min_mean_dist) {
                            min_mean_dist = dist;
                            new_medoid_id = pt_id;
                        }
                    }
                }
                medoid_ids[k] = new_medoid_id;
            }
        }
    }

    // Step 1.2 & 1.3: Variance and Weight calculation
    std::vector<float> V_i(K_medoids, 0.0f);
    std::vector<float> w_i(K_medoids, 0.0f);
    
    for (uint32_t i = 0; i < sample_size; ++i) {
        uint32_t k = cluster_assignments[i];
        uint32_t pt_id = sampled_ids[i];
        V_i[k] += compute_l2sq(get_vector(pt_id), get_vector(medoid_ids[k]), dim_);
    }

    float V_avg = 0.0f;
    for (uint32_t k = 0; k < K_medoids; ++k) {
        if (cluster_sizes[k] > 0) {
            w_i[k] = (float)cluster_sizes[k] / sample_size;
            V_i[k] /= cluster_sizes[k];
            V_avg += w_i[k] * V_i[k];
        }
    }

    // Step 1.4: Dynamic L_i Assignment & Clamping
    std::vector<float> L_continuous(K_medoids, 0.0f);
    std::vector<bool> is_maxed(K_medoids, false);
    float L_min = std::max(15.0f, 10.0f); // Default K_neighbors = 10 for build phase
    float L_max = (float)(L_target * 4);

    for (uint32_t k = 0; k < K_medoids; ++k) {
        L_continuous[k] = (V_avg > 0) ? (float)L_target * (V_i[k] / V_avg) : L_target;
    }

    // Normalization loop
    bool done = false;
    while (!done) {
        done = true;
        float current_E_L = 0.0f;
        float remaining_weight = 0.0f;

        for (uint32_t k = 0; k < K_medoids; ++k) {
            if (L_continuous[k] >= L_max && !is_maxed[k]) {
                L_continuous[k] = L_max;
                is_maxed[k] = true;
                done = false;
            }
            if (L_continuous[k] <= L_min && !is_maxed[k]) {
                L_continuous[k] = L_min;
                is_maxed[k] = true;
                done = false;
            }
            
            if (is_maxed[k]) {
                current_E_L += w_i[k] * L_continuous[k];
            } else {
                remaining_weight += w_i[k];
            }
        }

        float deficiency = L_target - current_E_L;
        // If deficiency < 0, we are over budget; if > 0, under budget.
        // Check if we are within 10% error
        float expected = 0.0f;
        for (uint32_t k = 0; k < K_medoids; ++k) {
            expected += w_i[k] * L_continuous[k];
        }
        
        if (std::abs(expected - L_target) <= 0.10f * L_target) {
            break; 
        }

        if (remaining_weight > 0 && !done) {
            // Re-distribute missing budget proportionally among non-maxed clusters based on their contribution relative to remaining
            // Actually, proportionally to their weight * base L (or just multiply them by a constant). Wait, multiply by constant.
            // new_sum * const = deficiency => const = deficiency / (sum w_i * L_i for non maxed)
            float non_maxed_E_L = 0.0f;
            for (uint32_t k = 0; k < K_medoids; ++k) {
                if (!is_maxed[k]) non_maxed_E_L += w_i[k] * L_continuous[k];
            }
            
            if (non_maxed_E_L > 0) {
                float scaling_factor = deficiency / non_maxed_E_L;
                for (uint32_t k = 0; k < K_medoids; ++k) {
                    if (!is_maxed[k]) {
                        L_continuous[k] *= scaling_factor;
                    }
                }
            } else {
                break; // They are all identically 0 or something weird, prevent nan
            }
        }
    }

    routing_table_.clear();
    for (uint32_t k = 0; k < K_medoids; ++k) {
        RoutingNode node;
        node.medoid_id = medoid_ids[k];
        node.assigned_L = (uint32_t)std::round(L_continuous[k]);
        if (node.assigned_L < 15) node.assigned_L = 15; // Just absolute sanity check

        const float* m_vec = get_vector(medoid_ids[k]);
        node.medoid_vector.assign(m_vec, m_vec + dim_);
        routing_table_.push_back(node);
        
        std::cout << "    Cluster " << k << " | w=" << w_i[k] << " | Var=" << V_i[k] 
                  << " | L=" << node.assigned_L << std::endl;
    }

    float final_expected = 0.0f;
    for (uint32_t k = 0; k < K_medoids; ++k) {
        final_expected += w_i[k] * routing_table_[k].assigned_L;
    }
    std::cout << "  Assigned Dynamc L_target Expected: " << final_expected << " (Target was " << L_target << ")" << std::endl;
}

// ============================================================================
// Search
// ============================================================================

SearchResult VamanaIndex::search(const float* query, uint32_t K, uint32_t L) const {
    if (L < K) L = K;

    Timer t;
    auto [candidates, dist_cmps] = greedy_search(query, L);
    double latency = t.elapsed_us();

    // Return top-K results
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

SearchResult VamanaIndex::search_dynamic(const float* query, uint32_t K) const {
    if (routing_table_.empty()) {
        std::cerr << "Routing table empty! Falling back to standard search with L = K." << std::endl;
        return search(query, K, K);
    }
    
    Timer t;
    float min_dist = std::numeric_limits<float>::max();
    uint32_t best_k = 0;
    
    // Query routing overhead
    for (uint32_t k = 0; k < routing_table_.size(); ++k) {
        float dist = compute_l2sq(query, routing_table_[k].medoid_vector.data(), dim_);
        if (dist < min_dist) {
            min_dist = dist;
            best_k = k;
        }
    }
    
    uint32_t start_node = routing_table_[best_k].medoid_id;
    uint32_t L_query = routing_table_[best_k].assigned_L;
    if (L_query < K) L_query = K;
    
    auto [candidates, dist_cmps] = greedy_search(query, L_query, start_node);
    dist_cmps += routing_table_.size(); // Count routing overhead distance comparisons
    
    double latency = t.elapsed_us();
    
    SearchResult result;
    result.dist_cmps = dist_cmps;
    result.latency_us = latency;
    result.ids.reserve(K);
    for (uint32_t i = 0; i < K && i < candidates.size(); i++) {
        result.ids.push_back(candidates[i].second);
    }
    return result;
}

// ============================================================================
// Save / Load
// ============================================================================
// Binary format:
//   [uint32] npts
//   [uint32] dim
//   [uint32] start_node
//   For each node i in [0, npts):
//     [uint32] degree
//     [uint32 * degree] neighbor IDs

void VamanaIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open())
        throw std::runtime_error("Cannot open file for writing: " + path);

    out.write(reinterpret_cast<const char*>(&npts_), 4);
    out.write(reinterpret_cast<const char*>(&dim_), 4);
    out.write(reinterpret_cast<const char*>(&start_node_), 4);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg = graph_[i].size();
        out.write(reinterpret_cast<const char*>(&deg), 4);
        if (deg > 0) {
            out.write(reinterpret_cast<const char*>(graph_[i].data()),
                      deg * sizeof(uint32_t));
        }
    }

    uint32_t K_medoids = routing_table_.size();
    out.write(reinterpret_cast<const char*>(&K_medoids), 4);
    for (const auto& node : routing_table_) {
        out.write(reinterpret_cast<const char*>(&node.medoid_id), 4);
        out.write(reinterpret_cast<const char*>(node.medoid_vector.data()), dim_ * sizeof(float));
        out.write(reinterpret_cast<const char*>(&node.assigned_L), 4);
    }

    std::cout << "Index saved to " << path << std::endl;
}

void VamanaIndex::load(const std::string& index_path,
                       const std::string& data_path) {
    // Load data vectors
    FloatMatrix mat = load_fbin(data_path);
    npts_ = mat.npts;
    dim_  = mat.dims;
    data_ = mat.data.release();
    owns_data_ = true;

    // Load graph
    std::ifstream in(index_path, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("Cannot open index file: " + index_path);

    uint32_t file_npts, file_dim;
    in.read(reinterpret_cast<char*>(&file_npts), 4);
    in.read(reinterpret_cast<char*>(&file_dim), 4);
    in.read(reinterpret_cast<char*>(&start_node_), 4);

    if (file_npts != npts_ || file_dim != dim_)
        throw std::runtime_error(
            "Index/data mismatch: index has " + std::to_string(file_npts) +
            "x" + std::to_string(file_dim) + ", data has " +
            std::to_string(npts_) + "x" + std::to_string(dim_));

    graph_.resize(npts_);
    locks_ = std::vector<std::mutex>(npts_);

    for (uint32_t i = 0; i < npts_; i++) {
        uint32_t deg;
        in.read(reinterpret_cast<char*>(&deg), 4);
        graph_[i].resize(deg);
        if (deg > 0) {
            in.read(reinterpret_cast<char*>(graph_[i].data()),
                    deg * sizeof(uint32_t));
        }
    }

    // Try to read routing table if present
    routing_table_.clear();
    uint32_t K_medoids = 0;
    if (in.read(reinterpret_cast<char*>(&K_medoids), 4)) {
        for (uint32_t k = 0; k < K_medoids; ++k) {
            RoutingNode node;
            node.medoid_vector.resize(dim_);
            in.read(reinterpret_cast<char*>(&node.medoid_id), 4);
            in.read(reinterpret_cast<char*>(node.medoid_vector.data()), dim_ * sizeof(float));
            in.read(reinterpret_cast<char*>(&node.assigned_L), 4);
            routing_table_.push_back(node);
        }
    }

    std::cout << "Index loaded: " << npts_ << " points, " << dim_
              << " dims, start=" << start_node_ << std::endl;
}
