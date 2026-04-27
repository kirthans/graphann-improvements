#include "vamana_index.h"
#include "timer.h"

#include <iostream>
#include <string>
#include <cstdlib>

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " --data <fbin_path>"
              << " --output <index_path>"
              << " [--R <max_degree=32>]"
              << " [--L <build_search_list=75>]"
              << " [--alpha <rng_alpha=1.2>]"
              << " [--gamma <degree_multiplier=1.5>]"
              << " [--L_target <dynamic_latency_target=50>]"
              << " [--pq_M <subspaces_for_pq=0>]"
              << std::endl;
}

int main(int argc, char** argv) {
    // Defaults
    std::string data_path, output_path;
    uint32_t R = 32;
    uint32_t L = 75;
    float alpha = 1.2f;
    float gamma = 1.5f;
    uint32_t L_target = 50;
    uint32_t pq_M = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc)       data_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_path = argv[++i];
        else if (arg == "--R" && i + 1 < argc)      R = std::atoi(argv[++i]);
        else if (arg == "--L" && i + 1 < argc)      L = std::atoi(argv[++i]);
        else if (arg == "--alpha" && i + 1 < argc)  alpha = std::atof(argv[++i]);
        else if (arg == "--gamma" && i + 1 < argc)  gamma = std::atof(argv[++i]);
        else if (arg == "--L_target" && i + 1 < argc) L_target = std::atoi(argv[++i]);
        else if (arg == "--pq_M" && i + 1 < argc)   pq_M = std::atoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_path.empty() || output_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "=== Vamana Index Builder ===" << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  R     = " << R << std::endl;
    std::cout << "  L     = " << L << std::endl;
    std::cout << "  alpha    = " << alpha << std::endl;
    std::cout << "  gamma    = " << gamma << std::endl;
    std::cout << "  L_target = " << L_target << std::endl;
    std::cout << "  pq_M     = " << pq_M << std::endl;

    VamanaIndex index;

    Timer total_timer;
    index.build(data_path, R, L, alpha, gamma, L_target);
    
    if (pq_M > 0) {
        index.train_pq(pq_M);
    }
    
    double total_time = total_timer.elapsed_seconds();

    std::cout << "\nTotal build time: " << total_time << " seconds" << std::endl;

    index.save(output_path);
    std::cout << "Done." << std::endl;
    return 0;
}
