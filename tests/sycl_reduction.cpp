// Executed SYCL masked reduction checks. Built without -fsycl, like an installed application.
#include "hbr/kernels.hpp"
#include "hbr/runtime.hpp"
#include "hbr/sycl.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class E, class F> void rejects(F f, const char* message) {
    bool caught = false;
    try { f(); } catch (const E&) { caught = true; }
    check(caught, message);
}
double reference(const std::vector<double>& values, const std::vector<std::uint8_t>& mask) {
    return hbr::masked_reduce(values, mask, {hbr::Backend::scalar, 1});
}
int main() {
    try {
        const auto device = hbr::sycl_device();
        std::cout << device.name << " through " << device.backend << '\n';
        std::size_t cases = 0;
        // The shapes, inputs and tolerance of the CUDA reduction checks in tests/cuda.cpp.
        for (auto rows : {0u, 1u, 2u, 3u, 17u, 257u}) for (auto cols : {0u, 1u, 7u, 259u}) {
            std::vector<double> x(static_cast<std::size_t>(rows) * cols);
            std::vector<std::uint8_t> mask(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) {
                x[i] = (static_cast<double>(i % 103) - 51.5) / 103;
                mask[i] = i % 3 ? 255 : 0;
            }
            const double actual = hbr::sycl_masked_reduce(x, mask);
            check(std::isfinite(actual) && std::abs(actual - reference(x, mask)) <= 1e-9, "SYCL reduction mismatch");
            ++cases;
        }
        // Integer values whose every partial sum is exact in binary64, so any summation order
        // must give the reference bit for bit. 3,000,001 values exceed the 4096 work groups of
        // the first pass, so each work item strides over the input several times.
        for (std::size_t n : {std::size_t{255}, std::size_t{257}, std::size_t{1} << 20, std::size_t{3000001}}) {
            std::vector<double> x(n);
            for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<double>(static_cast<long long>(i % 2001) - 1000);
            for (int pattern = 0; pattern < 4; ++pattern) {
                std::vector<std::uint8_t> mask(n);
                for (std::size_t i = 0; i < n; ++i)
                    mask[i] = pattern == 0 ? 1 : pattern == 1 ? 0 : pattern == 2 ? (i % 2 ? 0 : 7) : (i + 1 == n ? 1 : 0);
                check(hbr::sycl_masked_reduce(x, mask) == reference(x, mask), "SYCL exact reduction mismatch");
                ++cases;
            }
        }
        std::cout << cases << " reduction cases agree with the scalar reference\n";

        // Rejections, each before any device work, and the range check on selected values only.
        const auto largest = std::numeric_limits<double>::max();
        rejects<std::invalid_argument>([] { (void)hbr::sycl_masked_reduce(std::vector<double>(3), std::vector<std::uint8_t>(2)); },
            "size mismatch accepted");
        for (double value : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
            rejects<std::invalid_argument>([value] {
                (void)hbr::sycl_masked_reduce(std::vector<double>{value}, std::vector<std::uint8_t>{0}); }, "nonfinite input accepted");
        rejects<std::overflow_error>([largest] {
            (void)hbr::sycl_masked_reduce(std::vector<double>{largest, 1.0}, std::vector<std::uint8_t>{1, 1}); }, "overflow guard missing");
        check(hbr::sycl_masked_reduce(std::vector<double>{largest, largest}, std::vector<std::uint8_t>{1, 0}) == largest,
            "an unselected value counted toward the range");
        check(hbr::sycl_masked_reduce({}, {}) == 0.0, "empty input");

        // The borrowed input is never modified, and a second call sees new input.
        std::vector<double> values(4096);
        std::vector<std::uint8_t> mask(values.size(), 1);
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = static_cast<double>(i);
        const auto copy = values;
        check(hbr::sycl_masked_reduce(values, mask) == reference(values, mask), "ownership case");
        check(values == copy, "input modified");
        std::fill(values.begin(), values.end(), 2.0);
        check(hbr::sycl_masked_reduce(values, mask) == 8192.0, "new input not uploaded");

        // Through the runtime: concurrent calls on two workers, each owning its input.
        {
            hbr::Runtime runtime(2, 16);
            std::vector<hbr::Ticket> tickets;
            auto results = std::make_shared<std::vector<double>>(8);
            std::vector<double> expected;
            for (std::size_t job = 0; job < 8; ++job) {
                auto input = std::make_shared<std::vector<double>>(257 * 259);
                auto selected = std::make_shared<std::vector<std::uint8_t>>(input->size());
                for (std::size_t i = 0; i < input->size(); ++i) {
                    (*input)[i] = static_cast<double>(static_cast<long long>((i * (job + 1)) % 997) - 498);
                    (*selected)[i] = (i + job) % 3 ? 1 : 0;
                }
                expected.push_back(reference(*input, *selected));
                tickets.push_back(runtime.submit([input, selected, results, job] {
                    (*results)[job] = hbr::sycl_masked_reduce(*input, *selected);
                }));
            }
            for (std::size_t job = 0; job < tickets.size(); ++job) {
                tickets[job].wait();
                check(tickets[job].state() == hbr::State::completed, "runtime job not completed");
                check((*results)[job] == expected[job], "concurrent SYCL reduction mismatch");
            }
        }

        // A running cancellation is only a request, and a queued one prevents the start.
        {
            hbr::Runtime runtime(1, 4);
            std::promise<void> started, release;
            auto gate = release.get_future().share();
            auto output = std::make_shared<double>(-1.0);
            auto input = std::make_shared<std::vector<double>>(copy);
            auto selected = std::make_shared<std::vector<std::uint8_t>>(copy.size(), 1);
            auto running = runtime.submit([&started, gate, input, selected, output] {
                started.set_value();
                gate.wait();
                *output = hbr::sycl_masked_reduce(*input, *selected);
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
            check(*output == reference(*input, *selected), "work after a cancellation request was wrong");
            check(!ran->load(), "cancelled job ran");
        }
        std::cout << "SYCL reduction: reference agreement, rejections, ownership, runtime concurrency and cancellation passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
