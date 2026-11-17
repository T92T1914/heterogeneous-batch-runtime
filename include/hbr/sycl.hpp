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
}
