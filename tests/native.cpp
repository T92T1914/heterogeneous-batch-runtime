#include "hbr/kernels.hpp"
#include "hbr/runtime.hpp"
#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

void check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
template<class F> void rejects(F fn) { bool threw = false; try { fn(); } catch (const std::exception&) { threw = true; } check(threw, "expected rejection"); }
int main() {
    try {
        for (const auto n : {std::size_t{0}, std::size_t{1}, std::size_t{3}, std::size_t{7}, std::size_t{32771}}) {
            std::vector<double> x(n); std::vector<std::uint8_t> mask(n);
            double expected = 0;
            for (std::size_t i = 0; i < n; ++i) { x[i] = static_cast<double>(i % 101) - 50; mask[i] = i % 3 ? 255 : 0; if (mask[i]) expected += x[i]; }
            for (const auto backend : {hbr::Backend::scalar, hbr::Backend::optimized})
                for (auto threads : {1u, 2u, 4u})
                    check(hbr::masked_reduce(x, mask, {backend, threads}) == expected, "reduce tail/mask mismatch");
        }
        for (auto rows : {0u, 1u, 2u, 3u, 17u}) for (auto cols : {0u, 1u, 2u, 3u, 19u}) {
            std::vector<double> x(rows * cols); std::vector<std::uint8_t> bytes(rows * cols);
            for (std::size_t i = 0; i < x.size(); ++i) { x[i] = static_cast<double>(i % 113) - 56; bytes[i] = static_cast<std::uint8_t>(i % 256); }
            const auto ref = hbr::stencil3x3(x, rows, cols, {hbr::Backend::scalar, 1});
            const auto fast = hbr::stencil3x3(x, rows, cols, {hbr::Backend::optimized, 2});
            check(ref.size() == x.size(), "stencil shape");
            for (std::size_t i = 0; i < ref.size(); ++i) check(std::abs(ref[i] - fast[i]) <= 1e-12, "stencil differs");
            for (auto tr : {1u, 4u, 128u}) for (auto tc : {1u, 5u, 128u}) {
                const auto a = hbr::tile_histogram(bytes, rows, cols, tr, tc, {hbr::Backend::scalar, 1});
                const auto b = hbr::tile_histogram(bytes, rows, cols, tr, tc, {hbr::Backend::optimized, 2});
                check(a == b, "histogram differs"); std::uint64_t total = 0; for (auto count : a) total += count;
                check(total == bytes.size(), "histogram conservation");
            }
        }
        std::vector<double> bad{std::numeric_limits<double>::quiet_NaN()};
        std::vector<std::uint8_t> mask{0};
        rejects([&] { hbr::masked_reduce(bad, mask); });
        rejects([&] { hbr::stencil3x3(bad, 1, 1); });
        rejects([&] { hbr::masked_reduce({}, mask); });
        rejects([&] { hbr::masked_reduce({}, {}, {hbr::Backend::scalar, 0}); });
        rejects([&] { hbr::tile_histogram({}, 0, 0, 0, 1); });
        rejects([&] { hbr::stencil3x3({}, std::numeric_limits<std::size_t>::max(), 2); });
        const double maximum = std::numeric_limits<double>::max();
        std::vector<double> near_limit(13, std::ldexp(1.0, 969));
        near_limit[0] = std::nextafter(maximum, 0.0);
        std::vector<std::uint8_t> all(near_limit.size(), 1);
        for (auto backend : {hbr::Backend::scalar, hbr::Backend::optimized}) {
            rejects([&] { hbr::masked_reduce(near_limit, all, {backend, 2}); });
            std::vector<double> valid{maximum};
            std::vector<std::uint8_t> one{1};
            check(hbr::masked_reduce(valid, one, {backend, 1}) == maximum, "valid maximal input rejected");
            std::vector<double> cancellation{maximum, -maximum};
            std::vector<std::uint8_t> two{1,1};
            rejects([&] { hbr::masked_reduce(cancellation, two, {backend, 1}); });
            std::vector<double> large_image(25, maximum);
            for (double result : hbr::stencil3x3(large_image, 5, 5, {backend, 2}))
                check(std::isfinite(result), "stencil overflow on finite constant image");
        }
        // Gates establish real state transitions without timing sleeps.
        std::promise<void> started, release;
        auto gate = release.get_future().share();
        hbr::Runtime runtime(1, 1);
        auto running = runtime.submit([&] { started.set_value(); gate.wait(); });
        started.get_future().wait();
        auto queued = runtime.submit([] { throw std::runtime_error("cancelled job executed"); });
        rejects([&] { runtime.submit([] {}); });
        check(queued.cancel(), "queued cancellation not confirmed");
        queued.wait(); check(queued.state() == hbr::State::cancelled, "cancelled state");
        check(!running.cancel(), "running operation cancelled physically");
        check(running.cancellation_requested(), "lost cancellation request");
        check(running.state() == hbr::State::running, "early physical completion");
        release.set_value(); running.wait();
        check(running.state() == hbr::State::completed, "completion after cancellation request");
        hbr::Runtime second(1, 2);
        auto failure = second.submit([] { throw std::runtime_error("expected failure"); });
        rejects([&] { failure.wait(); }); check(failure.state() == hbr::State::failed, "exception state");
        auto owned = std::make_shared<int>(42); std::weak_ptr<int> weak = owned;
        auto owner = second.submit([owned] { check(*owned == 42, "lost owned input"); });
        owned.reset(); owner.wait(); check(weak.expired(), "captures retained after completion");
        std::cout << "Native numerical, validation and lifetime contracts passed. AVX2=" << hbr::avx2_available() << '\n';
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
