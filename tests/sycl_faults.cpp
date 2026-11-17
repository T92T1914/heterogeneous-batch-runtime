// Host exceptions injected at every checkpoint of a SYCL histogram call. Each failure must
// drain the queue before device memory is freed and must leave no device allocation behind.
#include "hbr/kernels.hpp"
#include "hbr/sycl.hpp"
#include "../src/sycl_faults.hpp"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
int main() {
    try {
        std::vector<std::uint8_t> data(17 * 19);
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i * 13);
        const auto expected = hbr::tile_histogram(data, 17, 19, 8, 7, {hbr::Backend::scalar, 1});
        int points = 0;
        for (auto local : {false, true}) {
            hbr::test::fail_at(-1);
            check(hbr::sycl_tile_histogram(data, 17, 19, 8, 7, local) == expected, "histogram without injection");
            points = hbr::test::checkpoints();
            check(points > 0 && hbr::test::live_allocations() == 0, "checkpoint or allocation accounting");
            for (int point = 0; point < points; ++point) {
                const auto before = hbr::test::drains();
                hbr::test::fail_at(point);
                bool injected = false;
                try { (void)hbr::sycl_tile_histogram(data, 17, 19, 8, 7, local); }
                catch (const std::runtime_error& error) { injected = std::string(error.what()) == "injected host exception"; }
                check(injected, "injected failure missing or replaced");
                check(hbr::test::drains() > before, "failed call did not drain its queue");
                check(hbr::test::live_allocations() == 0, "failed call leaked device memory");
                hbr::test::fail_at(-1);
                check(hbr::sycl_tile_histogram(data, 17, 19, 8, 7, local) == expected, "a failed call disturbed the next one");
            }
        }
        check(hbr::test::drain_errors() == 0, "cleanup synchronization failed");
        std::cout << points << " host exception checkpoints passed for global and for local bins\n";
        std::cout << "Actual device loss, SYCL allocation failure and GPU sanitizer coverage are not established\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
