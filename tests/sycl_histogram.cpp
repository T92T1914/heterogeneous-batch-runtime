// Executed SYCL histogram checks. Built without -fsycl, like an installed application.
#include "hbr/kernels.hpp"
#include "hbr/runtime.hpp"
#include "hbr/sycl.hpp"
#include <algorithm>
#include <atomic>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class E, class F> void rejects(F f, const char* message) {
    bool caught = false;
    try { f(); } catch (const E&) { caught = true; }
    check(caught, message);
}
std::vector<std::uint64_t> reference(const std::vector<std::uint8_t>& image, std::size_t rows, std::size_t cols,
                                     std::size_t tr, std::size_t tc) {
    return hbr::tile_histogram(image, rows, cols, tr, tc, {hbr::Backend::scalar, 1});
}
int main() {
    try {
        const auto device = hbr::sycl_device();
        std::cout << device.name << " through " << device.backend << '\n';
        std::size_t cases = 0;
        // The shapes and inputs of the CUDA histogram checks in tests/cuda.cpp.
        for (auto rows : {0u, 1u, 2u, 3u, 17u, 257u}) for (auto cols : {0u, 1u, 7u, 259u}) {
            std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * cols);
            for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i);
            const auto expected = reference(bytes, rows, cols, 31, 33);
            for (auto local : {false, true}) {
                check(hbr::sycl_tile_histogram(bytes, rows, cols, 31, 33, local) == expected, "SYCL histogram mismatch");
                ++cases;
            }
        }
        for (auto side : {32u, 256u, 1024u}) for (auto skewed : {false, true}) {
            std::vector<std::uint8_t> input(static_cast<std::size_t>(side) * side);
            for (std::size_t i = 0; i < input.size(); ++i) input[i] = skewed ? 7 : static_cast<std::uint8_t>(i * 13);
            const auto expected = reference(input, side, side, 64, 64);
            for (auto local : {false, true}) {
                const auto actual = hbr::sycl_tile_histogram(input, side, side, 64, 64, local);
                check(actual == expected, "SYCL benchmark histogram mismatch");
                check(std::accumulate(actual.begin(), actual.end(), std::uint64_t{}) == input.size(), "conservation");
                ++cases;
            }
        }
        std::cout << cases << " histogram cases equal the scalar reference\n";

        // Rejections, each before any device work.
        const std::vector<std::uint8_t> nine(9, 1);
        rejects<std::invalid_argument>([&] { (void)hbr::sycl_tile_histogram(nine, 3, 3, 0, 1); }, "zero tile accepted");
        rejects<std::invalid_argument>([&] { (void)hbr::sycl_tile_histogram(nine, 3, 4, 1, 1); }, "shape mismatch accepted");
        rejects<std::overflow_error>([&] {
            (void)hbr::sycl_tile_histogram(nine, std::numeric_limits<std::size_t>::max(), 2, 1, 1); }, "shape overflow accepted");

        // The result owns its storage, and the borrowed input is never modified.
        std::vector<std::uint8_t> image(64 * 64);
        for (std::size_t i = 0; i < image.size(); ++i) image[i] = static_cast<std::uint8_t>(i * 7);
        const auto copy = image;
        auto first = hbr::sycl_tile_histogram(image, 64, 64, 16, 16);
        check(image == copy, "input modified");
        const auto original = first;
        std::fill(image.begin(), image.end(), std::uint8_t{3});
        auto second = hbr::sycl_tile_histogram(image, 64, 64, 16, 16);
        check(first == original && second == reference(image, 64, 64, 16, 16), "new input not uploaded or result aliased");

        // Through the runtime: concurrent calls on two workers, each owning its input.
        {
            hbr::Runtime runtime(2, 16);
            std::vector<hbr::Ticket> tickets;
            auto results = std::make_shared<std::vector<std::vector<std::uint64_t>>>(8);
            std::vector<std::vector<std::uint8_t>> inputs;
            for (std::size_t job = 0; job < 8; ++job) {
                auto input = std::make_shared<std::vector<std::uint8_t>>(257 * 259);
                for (std::size_t i = 0; i < input->size(); ++i) input->at(i) = static_cast<std::uint8_t>(i * (job + 1));
                inputs.push_back(*input);
                tickets.push_back(runtime.submit([input, results, job] {
                    (*results)[job] = hbr::sycl_tile_histogram(*input, 257, 259, 31, 33, job % 2 == 0);
                }));
            }
            for (std::size_t job = 0; job < tickets.size(); ++job) {
                tickets[job].wait();
                check(tickets[job].state() == hbr::State::completed, "runtime job not completed");
                check((*results)[job] == reference(inputs[job], 257, 259, 31, 33), "concurrent SYCL histogram mismatch");
            }
        }

        // A running cancellation is only a request, and a queued one prevents the start.
        {
            hbr::Runtime runtime(1, 4);
            std::promise<void> started, release;
            auto gate = release.get_future().share();
            auto output = std::make_shared<std::vector<std::uint64_t>>();
            auto input = std::make_shared<std::vector<std::uint8_t>>(image);
            auto running = runtime.submit([&started, gate, input, output] {
                started.set_value();
                gate.wait();
                *output = hbr::sycl_tile_histogram(*input, 64, 64, 16, 16);
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
            check(*output == reference(*input, 64, 64, 16, 16), "work after a cancellation request was wrong");
            check(!ran->load(), "cancelled job ran");
        }
        std::cout << "SYCL histogram: exact bins, rejections, ownership, runtime concurrency and cancellation passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
