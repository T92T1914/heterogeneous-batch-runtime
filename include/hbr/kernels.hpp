#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace hbr {
enum class Backend { scalar, optimized };
struct Options { Backend backend = Backend::optimized; std::size_t threads = 2; };
bool avx2_available() noexcept;
double masked_reduce(std::span<const double> values, std::span<const std::uint8_t> mask, Options = {});
std::vector<std::uint64_t> tile_histogram(std::span<const std::uint8_t> image,
    std::size_t rows, std::size_t cols, std::size_t tile_rows, std::size_t tile_cols, Options = {});
std::vector<double> stencil3x3(std::span<const double> image, std::size_t rows, std::size_t cols, Options = {});
}
