// Built without -fsycl. It reaches the backend only through the public header and the
// shared library, the way an installed application does.
#include "hbr/sycl.hpp"
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
int main(int argc, char** argv) {
    const bool expect_no_gpu = argc == 2 && std::string(argv[1]) == "--expect-no-gpu";
    if (argc > 1 && !expect_no_gpu) { std::cerr << "usage: hbr_sycl_tests [--expect-no-gpu]\n"; return 2; }
    try {
        const auto device = hbr::sycl_device();
        if (expect_no_gpu) { std::cerr << "selected " << device.name << " although no GPU was offered\n"; return 1; }
        std::cout << "backend " << device.backend << '\n' << "platform " << device.platform << '\n'
                  << "device " << device.name << '\n' << "driver " << device.driver_version << '\n'
                  << "global memory bytes " << device.global_memory << '\n'
                  << "local memory bytes " << device.local_memory << '\n'
                  << "fp64 " << (device.fp64 ? "yes" : "no") << '\n'
                  << "atomic64 " << (device.atomic64 ? "yes" : "no") << '\n';
        if (device.backend.empty() || device.platform.empty() || device.name.empty() || !device.global_memory || !device.local_memory)
            throw std::runtime_error("SYCL device description incomplete");
        std::cout << "SYCL device discovery passed\n";
    } catch (const std::runtime_error& error) {
        if (expect_no_gpu) { std::cout << "rejected as expected: " << error.what() << '\n'; return 0; }
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        // Anything other than std::runtime_error breaks the interface promise, even when no GPU was expected.
        std::cerr << "unexpected exception type: " << error.what() << '\n';
        return 1;
    }
}
