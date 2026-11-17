// Built without -fsycl. It reaches the backend only through the public header and the
// shared library, the way an installed application does.
#include "hbr/sycl.hpp"
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
// With no GPU offered, validation still runs first and an empty image needs no device.
// Any other call must fail visibly as std::runtime_error.
int without_gpu() {
    const std::vector<std::uint8_t> nine(9, 1);
    try { (void)hbr::sycl_tile_histogram(nine, 3, 3, 0, 1); return 1; } catch (const std::invalid_argument&) {}
    try { (void)hbr::sycl_tile_histogram(nine, 3, 4, 1, 1); return 1; } catch (const std::invalid_argument&) {}
    if (!hbr::sycl_tile_histogram({}, 0, 5, 2, 2).empty()) return 1;
    const std::vector<std::uint8_t> wide(2048 * 4096);
    try { (void)hbr::sycl_tile_histogram(wide, 2048, 4096, 1, 1); return 1; } catch (const std::overflow_error&) {}
    try { (void)hbr::sycl_tile_histogram(nine, 3, 3, 2, 2); return 1; } catch (const std::runtime_error&) {}
    try { (void)hbr::sycl_device(); return 1; }
    catch (const std::runtime_error& error) { std::cout << "rejected as expected: " << error.what() << '\n'; }
    return 0;
}
int main(int argc, char** argv) {
    const bool expect_no_gpu = argc == 2 && std::string(argv[1]) == "--expect-no-gpu";
    if (argc > 1 && !expect_no_gpu) { std::cerr << "usage: hbr_sycl_tests [--expect-no-gpu]\n"; return 2; }
    try {
        if (expect_no_gpu) {
            const int status = without_gpu();
            if (status) std::cerr << "a call succeeded or failed with the wrong type although no GPU was offered\n";
            return status;
        }
        const auto device = hbr::sycl_device();
        std::cout << "backend " << device.backend << '\n' << "platform " << device.platform << '\n'
                  << "device " << device.name << '\n' << "driver " << device.driver_version << '\n'
                  << "global memory bytes " << device.global_memory << '\n'
                  << "local memory bytes " << device.local_memory << '\n'
                  << "fp64 " << (device.fp64 ? "yes" : "no") << '\n'
                  << "atomic64 " << (device.atomic64 ? "yes" : "no") << '\n';
        if (device.backend.empty() || device.platform.empty() || device.name.empty() || !device.global_memory || !device.local_memory)
            throw std::runtime_error("SYCL device description incomplete");
        std::cout << "SYCL device discovery passed\n";
    } catch (const std::exception& error) {
        // Anything that escapes here is a failure, including an exception of the wrong type.
        std::cerr << error.what() << '\n';
        return 1;
    }
}
