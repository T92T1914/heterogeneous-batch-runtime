#include "hbr/kernels.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 || std::string(argv[1]) != "--run") {
        std::cerr << "Use --run after reserving one measurement slot.\n"; return 2;
    }
    std::cout << "workload,side,backend,threads,iteration,elapsed_ms,checksum,avx2\n" << std::setprecision(17);
    for (auto side : {32u, 256u, 1024u}) {
        const std::size_t n = static_cast<std::size_t>(side) * side;
        std::vector<double> x(n); std::vector<std::uint8_t> mask(n), bytes(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<double>(i % 1009) / 1009 - 0.5;
            mask[i] = i % 3 != 0; bytes[i] = static_cast<std::uint8_t>((i * 13) % 256);
        }
        for (auto backend : {hbr::Backend::scalar, hbr::Backend::optimized}) {
            for (auto threads : {1u, 2u}) {
                if (backend == hbr::Backend::scalar && threads != 1) continue;
                hbr::Options options{backend, threads};
                for (const auto* workload : {"masked_reduce", "tile_histogram", "stencil3x3"}) {
                    for (int iteration = 0; iteration < 6; ++iteration) {
                        const auto begin = std::chrono::steady_clock::now();
                        double checksum = 0;
                        if (std::string(workload) == "masked_reduce") checksum = hbr::masked_reduce(x, mask, options);
                        else if (std::string(workload) == "tile_histogram") {
                            const auto result = hbr::tile_histogram(bytes, side, side, 64, 64, options);
                            checksum = static_cast<double>(result[result.size() / 2]);
                        } else {
                            const auto result = hbr::stencil3x3(x, side, side, options);
                            checksum = result[result.size() / 2];
                        }
                        const auto end = std::chrono::steady_clock::now();
                        std::cout << workload << ',' << side << ',' << (backend == hbr::Backend::scalar ? "scalar" : "optimized")
                          << ',' << threads << ',' << iteration << ','
                          << std::chrono::duration<double, std::milli>(end - begin).count() << ',' << checksum << ',' << hbr::avx2_available() << '\n';
                    }
                }
            }
        }
    }
}
