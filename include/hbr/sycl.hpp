#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
namespace hbr {
// This header includes no SYCL header, so an application compiles it with any C++20
// compiler and links the shared hbr::hbr_sycl library, which carries its own device code.
struct SyclDevice {
    std::string backend, platform, name, driver_version;
    std::size_t global_memory, local_memory;
    bool fp64, atomic64;
};
// Selects a GPU through the SYCL GPU selector, which honors ONEAPI_DEVICE_SELECTOR, among
// the GPUs that can run every kernel this library carries.
// Throws std::runtime_error when no GPU is available. It never falls back to a CPU device.
SyclDevice sycl_device();
// The counterpart of cuda_tile_histogram, with work group local bins in place of CUDA shared
// bins. A completed synchronous call: the input is borrowed until return, every queued copy
// and kernel has finished before the call returns or throws, and the result owns its storage.
// Validation errors are thrown before any device is selected. A SYCL failure is thrown as
// std::runtime_error.
std::vector<std::uint64_t> sycl_tile_histogram(std::span<const std::uint8_t> image, std::size_t rows,
    std::size_t cols, std::size_t tile_rows, std::size_t tile_cols, bool local_bins = true);
// The counterpart of cuda_masked_reduce: the binary64 sum of the values whose mask byte is
// nonzero, under the same completed call contract. Nonfinite input, a size mismatch and a
// selected absolute sum beyond the binary64 range are rejected before any device is selected,
// as in masked_reduce. The summation order differs from the CPU, so the sum can differ in the
// last bits.
double sycl_masked_reduce(std::span<const double> values, std::span<const std::uint8_t> mask);
// The counterpart of cuda_stencil3x3 and stencil3x3: the mean of each interior 3 by 3
// neighborhood, each input divided by nine and added in the scalar order, bounded to the largest
// finite magnitude, with the border and images under 3 in either dimension copied. Nonfinite
// input and a shape mismatch are rejected before any device is selected. Same completed call
// contract as the histogram.
std::vector<double> sycl_stencil3x3(std::span<const double> image, std::size_t rows, std::size_t cols);
}
