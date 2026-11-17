// Executed SYCL stencil checks. Built without -fsycl, like an installed application.
#include "hbr/kernels.hpp"
#include "hbr/runtime.hpp"
#include "hbr/sycl.hpp"
#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class E, class F> void rejects(F f, const char* message) {
    bool caught = false;
    try { f(); } catch (const E&) { caught = true; }
    check(caught, message);
}
std::vector<double> reference(const std::vector<double>& image, std::size_t rows, std::size_t cols) {
    return hbr::stencil3x3(image, rows, cols, {hbr::Backend::scalar, 1});
}
// The kernel keeps the scalar order of the nine divided terms and contracts nothing, so each
// output must equal the scalar one bit for bit, sign of zero included, which is stricter than the
// 1e-12 of the CUDA test.
void same(const std::vector<double>& actual, const std::vector<double>& expected, const char* message) {
    check(actual.size() == expected.size(), message);
    for (std::size_t i = 0; i < actual.size(); ++i)
        check(std::isfinite(actual[i]) && std::bit_cast<std::uint64_t>(actual[i]) == std::bit_cast<std::uint64_t>(expected[i]), message);
}
std::vector<double> noise(std::size_t n, std::uint64_t seed) {
    std::vector<double> x(n);
    for (auto& value : x) {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        value = static_cast<double>(seed >> 11) / 9007199254740992.0 * 2 - 1;
    }
    return x;
}
int main() {
    try {
        const auto device = hbr::sycl_device();
        std::cout << device.name << " through " << device.backend << '\n';
        std::size_t cases = 0;
        // The shapes and inputs of the CUDA stencil checks in tests/cuda.cpp, including the
        // images under 3 in either dimension that are copied.
        for (auto rows : {0u, 1u, 2u, 3u, 17u, 257u}) for (auto cols : {0u, 1u, 7u, 259u}) {
            std::vector<double> x(static_cast<std::size_t>(rows) * cols);
            for (std::size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<double>(i % 103) - 51.5) / 103;
            same(hbr::sycl_stencil3x3(x, rows, cols), reference(x, rows, cols), "SYCL stencil mismatch");
            ++cases;
        }
        // Larger images with values spread in [-1, 1). 1024 by 1024 gives each of the 4096 by 256 work
        // items one pixel, and the other two make each work item stride over two or three.
        const std::pair<std::size_t, std::size_t> shapes[] = {{1024, 1024}, {2049, 1023}, {3, 1000003}};
        for (const auto& [rows, cols] : shapes) {
            const auto x = noise(rows * cols, rows * 31 + cols);
            same(hbr::sycl_stencil3x3(x, rows, cols), reference(x, rows, cols), "SYCL large stencil mismatch");
            ++cases;
        }
        // At the representable limits the last addition overflows and the output is bounded.
        const auto largest = std::numeric_limits<double>::max();
        for (const auto value : {largest, -largest}) {
            const std::vector<double> input(25, value);
            same(hbr::sycl_stencil3x3(input, 5, 5), reference(input, 5, 5), "SYCL extreme stencil mismatch");
            ++cases;
        }
        // Subnormal inputs, whose ninths underflow, and tiny negative ones, whose interior is -0.0.
        {
            std::vector<double> tiny(7 * 9);
            for (std::size_t i = 0; i < tiny.size(); ++i)
                tiny[i] = (static_cast<double>(i % 7) - 3) * 1e-310;
            same(hbr::sycl_stencil3x3(tiny, 7, 9), reference(tiny, 7, 9), "SYCL subnormal stencil mismatch");
            ++cases;
            const std::vector<double> negative(25, -std::numeric_limits<double>::denorm_min());
            const auto zeros = reference(negative, 5, 5);
            check(std::signbit(zeros[12]) && zeros[12] == 0.0, "scalar reference no longer gives -0.0");
            same(hbr::sycl_stencil3x3(negative, 5, 5), zeros, "SYCL signed zero stencil mismatch");
            ++cases;
        }
        std::cout << cases << " stencil cases equal the scalar reference bit for bit\n";

        // Rejections, each before any device work.
        rejects<std::invalid_argument>([] { (void)hbr::sycl_stencil3x3(std::vector<double>(9), 3, 4); }, "shape mismatch accepted");
        rejects<std::overflow_error>([] { (void)hbr::sycl_stencil3x3({}, std::numeric_limits<std::size_t>::max(), 2); },
            "shape overflow accepted");
        for (double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
            std::vector<double> bad(9, 1.0);
            bad[4] = value;
            rejects<std::invalid_argument>([&bad] { (void)hbr::sycl_stencil3x3(bad, 3, 3); }, "nonfinite input accepted");
        }
        check(hbr::sycl_stencil3x3({}, 0, 4).empty(), "empty image");

        // The borrowed input is never modified, the result owns its storage, and a second call
        // sees new input.
        auto image = noise(64 * 64, 7);
        const auto copy = image;
        auto first = hbr::sycl_stencil3x3(image, 64, 64);
        check(image == copy, "input modified");
        const auto original = first;
        std::fill(image.begin(), image.end(), 0.5);
        auto second = hbr::sycl_stencil3x3(image, 64, 64);
        check(first == original, "result aliased");
        same(second, reference(image, 64, 64), "new input not uploaded");

        // Through the runtime: concurrent calls on two workers, each owning its input.
        {
            hbr::Runtime runtime(2, 16);
            std::vector<hbr::Ticket> tickets;
            auto results = std::make_shared<std::vector<std::vector<double>>>(8);
            std::vector<std::vector<double>> expected;
            for (std::size_t job = 0; job < 8; ++job) {
                auto input = std::make_shared<std::vector<double>>(noise(257 * 259, job + 1));
                expected.push_back(reference(*input, 257, 259));
                tickets.push_back(runtime.submit([input, results, job] {
                    (*results)[job] = hbr::sycl_stencil3x3(*input, 257, 259);
                }));
            }
            for (std::size_t job = 0; job < tickets.size(); ++job) {
                tickets[job].wait();
                check(tickets[job].state() == hbr::State::completed, "runtime job not completed");
                same((*results)[job], expected[job], "concurrent SYCL stencil mismatch");
            }
        }

        // A running cancellation is only a request, and a queued one prevents the start.
        {
            hbr::Runtime runtime(1, 4);
            std::promise<void> started, release;
            auto gate = release.get_future().share();
            auto output = std::make_shared<std::vector<double>>();
            auto input = std::make_shared<std::vector<double>>(copy);
            auto running = runtime.submit([&started, gate, input, output] {
                started.set_value();
                gate.wait();
                *output = hbr::sycl_stencil3x3(*input, 64, 64);
            });
            auto ran = std::make_shared<std::atomic<bool>>(false);
            auto queued = runtime.submit([ran] { ran->store(true); });
            started.get_future().wait();
            check(!running.cancel() && running.cancellation_requested(), "running cancellation not recorded as a request");
            check(running.state() == hbr::State::running, "running cancellation changed the state");
            check(queued.cancel() && queued.state() == hbr::State::cancelled, "queued cancellation not confirmed");
            release.set_value();
            running.wait();
            check(running.state() == hbr::State::completed, "requested cancellation stopped running work");
            same(*output, reference(*input, 64, 64), "work after a cancellation request was wrong");
            check(!ran->load(), "cancelled job ran");
        }
        std::cout << "SYCL stencil: exact outputs, rejections, ownership, runtime concurrency and cancellation passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
