#include "hbr/kernels.hpp"
#include "hbr/detail/validation.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#if defined(HBR_HAS_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace hbr {
#ifdef HBR_HAS_AVX2
double reduce_avx2(const double*, const std::uint8_t*, std::size_t);
void stencil_row_avx2(const double*, const double*, const double*, double*, std::size_t);
#endif
bool avx2_available() noexcept {
#if defined(HBR_HAS_AVX2) && defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuidex(info, 1, 0);
    if ((info[2] & (1 << 27)) == 0 || (info[2] & (1 << 28)) == 0) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#elif defined(HBR_HAS_AVX2) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
namespace {
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("shape multiplication overflow");
    return a * b;
}
void validate_options(Options options) {
    if (options.threads < 1 || options.threads > 64)
        throw std::invalid_argument("threads must be in [1, 64]");
    if (options.backend != Backend::scalar && options.backend != Backend::optimized)
        throw std::invalid_argument("unknown backend");
}
void finite(std::span<const double> values) {
    for (double value : values)
        if (!std::isfinite(value)) throw std::invalid_argument("input values must be finite");
}
template<class F> void parallel(std::size_t count, std::size_t requested, F fn) {
    const auto threads = std::min(count, requested);
    if (threads < 2) { if (count) fn(0, count, 0); return; }
    // jthread joins launched workers if a later thread construction throws.
    std::vector<std::jthread> workers;
    workers.reserve(threads - 1);
    const auto part = count / threads;
    const auto remainder = count % threads;
    const auto begin = [&](std::size_t i) { return part * i + std::min(i, remainder); };
    for (std::size_t t = 1; t < threads; ++t) {
        const auto first = begin(t), last = begin(t + 1);
        workers.emplace_back([&fn, first, last, t] { fn(first, last, t); });
    }
    fn(0, begin(1), 0);
}
double reduce_scalar(const double* values, const std::uint8_t* mask, std::size_t n) {
    double result = 0;
    for (std::size_t i = 0; i < n; ++i) if (mask[i]) result += values[i];
    return result;
}
}
double masked_reduce(std::span<const double> values, std::span<const std::uint8_t> mask, Options options) {
    validate_options(options);
    if (values.size() != mask.size()) throw std::invalid_argument("mask and values have different sizes");
    finite(values);
    detail::reduction_range(values, mask);
    if (values.empty()) return 0.0;
    if (options.backend == Backend::scalar) return reduce_scalar(values.data(), mask.data(), values.size());
    const auto threads = values.size() < 32768 ? std::size_t{1} : std::min(options.threads, values.size());
    std::vector<double> partial(threads, 0.0);
    parallel(values.size(), threads, [&](std::size_t first, std::size_t last, std::size_t t) {
#ifdef HBR_HAS_AVX2
        if (avx2_available()) {
            partial[t] = reduce_avx2(values.data() + first, mask.data() + first, last - first);
            return;
        }
#endif
        // Four independent accumulators remove the scalar dependency chain.
        double a = 0, b = 0, c = 0, d = 0;
        auto i = first;
        for (; last - i >= 4; i += 4) {
            if (mask[i]) a += values[i];
            if (mask[i + 1]) b += values[i + 1];
            if (mask[i + 2]) c += values[i + 2];
            if (mask[i + 3]) d += values[i + 3];
        }
        partial[t] = (a + b) + (c + d);
        for (; i < last; ++i) if (mask[i]) partial[t] += values[i];
    });
    double result = 0;
    for (auto value : partial) result += value;
    return result;
}
std::vector<std::uint64_t> tile_histogram(std::span<const std::uint8_t> image,
    std::size_t rows, std::size_t cols, std::size_t tile_rows, std::size_t tile_cols, Options options) {
    validate_options(options);
    if (!tile_rows || !tile_cols) throw std::invalid_argument("tile dimensions must be positive");
    if (product(rows, cols) != image.size()) throw std::invalid_argument("image shape differs from buffer size");
    const auto nr = rows / tile_rows + (rows % tile_rows != 0);
    const auto nc = cols / tile_cols + (cols % tile_cols != 0);
    const auto tiles = product(nr, nc);
    std::vector<std::uint64_t> result(product(tiles, 256), 0);
    parallel(tiles, options.backend == Backend::scalar ? 1 : options.threads,
      [&](std::size_t first, std::size_t last, std::size_t) {
        for (auto tile = first; tile < last; ++tile) {
            const auto r0 = (tile / nc) * tile_rows;
            const auto c0 = (tile % nc) * tile_cols;
            const auto r1 = r0 + std::min(tile_rows, rows - r0);
            const auto c1 = c0 + std::min(tile_cols, cols - c0);
            auto* bins = result.data() + tile * 256;
            if (options.backend == Backend::scalar) {
                for (auto r = r0; r < r1; ++r)
                    for (auto c = c0; c < c1; ++c) ++bins[image[r * cols + c]];
            } else {
                // Private banks break repeated-bin dependency chains. No shared atomics.
                std::uint64_t banks[4][256]{};
                for (auto r = r0; r < r1; ++r) {
                    auto c = c0;
                    for (; c1 - c >= 4; c += 4)
                        for (std::size_t b = 0; b < 4; ++b) ++banks[b][image[r * cols + c + b]];
                    for (; c < c1; ++c) ++banks[0][image[r * cols + c]];
                }
                for (std::size_t b = 0; b < 256; ++b)
                    bins[b] = (banks[0][b] + banks[1][b]) + (banks[2][b] + banks[3][b]);
            }
        }
    });
    return result;
}
std::vector<double> stencil3x3(std::span<const double> image, std::size_t rows, std::size_t cols, Options options) {
    validate_options(options);
    if (product(rows, cols) != image.size()) throw std::invalid_argument("image shape differs from buffer size");
    finite(image);
    std::vector<double> result(image.begin(), image.end());
    if (rows < 3 || cols < 3) return result;
    parallel(rows - 2, options.backend == Backend::scalar ? 1 : options.threads,
      [&](std::size_t first, std::size_t last, std::size_t) {
        for (auto r = first + 1; r < last + 1; ++r) {
            const double* top = image.data() + (r - 1) * cols;
            const double* middle = top + cols;
            const double* bottom = middle + cols;
            double* out = result.data() + r * cols;
#ifdef HBR_HAS_AVX2
            if (options.backend == Backend::optimized && avx2_available()) {
                stencil_row_avx2(top, middle, bottom, out, cols);
                continue;
            }
#endif
            for (std::size_t c = 1; c < cols - 1; ++c) {
                double sum = top[c - 1] / 9.0;
                sum += top[c] / 9.0; sum += top[c + 1] / 9.0;
                sum += middle[c - 1] / 9.0; sum += middle[c] / 9.0; sum += middle[c + 1] / 9.0;
                sum += bottom[c - 1] / 9.0; sum += bottom[c] / 9.0; sum += bottom[c + 1] / 9.0;
                // A finite mean is bounded. The last rounded addition can overflow at DBL_MAX.
                out[c] = std::clamp(sum, -std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
            }
        }
    });
    return result;
}
}
