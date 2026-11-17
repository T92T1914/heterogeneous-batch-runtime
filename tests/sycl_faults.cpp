// Failures injected at every checkpoint of a SYCL call, first as a host exception and then as
// a sycl::exception, and an asynchronous SYCL error found at completion. Each failure must
// leave the library as std::runtime_error, drain the queue before device memory is freed and
// leave no device allocation behind.
#include "hbr/kernels.hpp"
#include "hbr/sycl.hpp"
#include "../src/sycl_faults.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
// True when call throws std::runtime_error whose message starts with prefix and contains text.
bool fails_with(const std::function<bool()>& call, const std::string& prefix, const std::string& text) {
    try { (void)call(); } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        return message.rfind(prefix, 0) == 0 && message.find(text) != std::string::npos;
    }
    return false;
}
void after_failure(int drains_before, const std::function<bool()>& correct) {
    hbr::test::fail_at(-1);
    hbr::test::fail_as_sycl(false);
    hbr::test::fail_async(false);
    check(hbr::test::drains() > drains_before, "failed call did not drain its queue");
    check(hbr::test::live_allocations() == 0, "failed call leaked device memory");
    check(correct(), "a failed call disturbed the next one");
}
// correct runs the call once and reports whether it returned the expected result.
void inject_everywhere(int expected_points, const std::function<bool()>& correct) {
    hbr::test::fail_at(-1);
    check(correct(), "call without injection");
    check(hbr::test::checkpoints() == expected_points, "checkpoint count changed");
    check(hbr::test::live_allocations() == 0, "allocation accounting");
    for (bool as_sycl : {false, true}) for (int point = 0; point < expected_points; ++point) {
        const auto before = hbr::test::drains();
        hbr::test::fail_as_sycl(as_sycl);
        hbr::test::fail_at(point);
        check(as_sycl ? fails_with(correct, "SYCL: ", "injected SYCL exception")
                      : fails_with(correct, "injected host exception", ""), "injected failure missing, replaced or not converted");
        after_failure(before, correct);
    }
    const auto before = hbr::test::drains();
    hbr::test::fail_async(true);
    check(fails_with(correct, "SYCL: ", "injected asynchronous SYCL error"), "kept asynchronous error not thrown");
    after_failure(before, correct);
}
int main() {
    try {
        std::vector<std::uint8_t> data(17 * 19);
        for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<std::uint8_t>(i * 13);
        const auto expected = hbr::tile_histogram(data, 17, 19, 8, 7, {hbr::Backend::scalar, 1});
        for (auto local : {false, true})
            inject_everywhere(6, [&] { return hbr::sycl_tile_histogram(data, 17, 19, 8, 7, local) == expected; });
        std::cout << "6 checkpoints passed for global and for local bins, as host and as SYCL exceptions\n";

        std::vector<double> values(70000);
        std::vector<std::uint8_t> mask(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<double>(static_cast<long long>(i % 61) - 30);
            mask[i] = i % 5 ? 1 : 0;
        }
        const double sum = hbr::masked_reduce(values, mask, {hbr::Backend::scalar, 1});
        inject_everywhere(8, [&] { return hbr::sycl_masked_reduce(values, mask) == sum; });
        std::cout << "8 checkpoints passed for the masked reduction, as host and as SYCL exceptions\n";

        std::vector<double> image(13 * 11);
        for (std::size_t i = 0; i < image.size(); ++i) image[i] = static_cast<double>(i % 17) / 4 - 2;
        const auto filtered = hbr::stencil3x3(image, 13, 11, {hbr::Backend::scalar, 1});
        inject_everywhere(6, [&] { return hbr::sycl_stencil3x3(image, 13, 11) == filtered; });
        std::cout << "6 checkpoints passed for the stencil, as host and as SYCL exceptions\n";

        check(hbr::test::drain_errors() == 0, "cleanup synchronization failed");
        std::cout << "A kept asynchronous SYCL error was thrown after the drain\n";
        std::cout << "The real asynchronous handler, device loss, SYCL allocation failure and GPU sanitizer coverage are not established\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
