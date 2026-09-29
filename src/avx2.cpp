#include <immintrin.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <algorithm>
namespace hbr {
double reduce_avx2(const double* values, const std::uint8_t* mask, std::size_t n) {
    auto sum = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; n - i >= 4; i += 4) {
        auto bits = _mm256_set_epi64x(mask[i+3] ? -1LL : 0LL, mask[i+2] ? -1LL : 0LL,
                                     mask[i+1] ? -1LL : 0LL, mask[i] ? -1LL : 0LL);
        sum = _mm256_add_pd(sum, _mm256_and_pd(_mm256_loadu_pd(values + i), _mm256_castsi256_pd(bits)));
    }
    double lanes[4];
    _mm256_storeu_pd(lanes, sum);
    double result = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
    for (; i < n; ++i) if (mask[i]) result += values[i];
    return result;
}
void stencil_row_avx2(const double* top, const double* middle, const double* bottom,
                     double* out, std::size_t cols) {
    std::size_t c = 1;
    const auto nine = _mm256_set1_pd(9.0);
    for (; cols - 1 - c >= 4; c += 4) {
        auto sum = _mm256_div_pd(_mm256_loadu_pd(top + c - 1), nine);
        sum = _mm256_add_pd(sum, _mm256_div_pd(_mm256_loadu_pd(top + c), nine));
        sum = _mm256_add_pd(sum, _mm256_div_pd(_mm256_loadu_pd(top + c + 1), nine));
        sum = _mm256_add_pd(sum, _mm256_div_pd(_mm256_loadu_pd(middle + c - 1), nine));
        sum = _mm256_add_pd(sum, _mm256_div_pd(_mm256_loadu_pd(middle + c), nine));
        sum = _mm256_add_pd(sum, _mm256_div_pd(_mm256_loadu_pd(middle + c + 1), nine));
        sum = _mm256_add_pd(sum, _mm256_div_pd(_mm256_loadu_pd(bottom + c - 1), nine));
        sum = _mm256_add_pd(sum, _mm256_div_pd(_mm256_loadu_pd(bottom + c), nine));
        sum = _mm256_add_pd(sum, _mm256_div_pd(_mm256_loadu_pd(bottom + c + 1), nine));
        const auto maximum = _mm256_set1_pd(std::numeric_limits<double>::max());
        const auto minimum = _mm256_set1_pd(-std::numeric_limits<double>::max());
        _mm256_storeu_pd(out + c, _mm256_max_pd(minimum, _mm256_min_pd(maximum, sum)));
    }
    for (; c < cols - 1; ++c) {
        double sum = top[c - 1] / 9.0;
        sum += top[c] / 9.0; sum += top[c+1] / 9.0;
        sum += middle[c-1] / 9.0; sum += middle[c] / 9.0; sum += middle[c+1] / 9.0;
        sum += bottom[c-1] / 9.0; sum += bottom[c] / 9.0; sum += bottom[c+1] / 9.0;
        out[c] = std::clamp(sum, -std::numeric_limits<double>::max(), std::numeric_limits<double>::max());
    }
}
}
