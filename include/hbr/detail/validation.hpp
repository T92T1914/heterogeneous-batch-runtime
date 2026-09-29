#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
namespace hbr::detail {
inline void reduction_range(std::span<const double> values, std::span<const std::uint8_t> mask) {
    double upper = 0;
    for (std::size_t i = 0; i < values.size(); ++i) if (mask[i]) {
        const double value = std::abs(values[i]);
        const double sum = upper + value;
        if (!std::isfinite(sum)) throw std::overflow_error("selected absolute sum exceeds binary64 range");
        // TwoSum recovers the rounding residual without relying on extended precision.
        const double recovered = sum - upper;
        const double residual = (upper - (sum - recovered)) + (value - recovered);
        upper = residual > 0 ? std::nextafter(sum, std::numeric_limits<double>::infinity()) : sum;
        if (!std::isfinite(upper)) throw std::overflow_error("selected absolute sum exceeds conservative binary64 bound");
    }
}
}
