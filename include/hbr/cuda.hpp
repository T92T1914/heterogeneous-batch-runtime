#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
namespace hbr {
struct CudaTiming { double upload_ms, kernel_ms, download_ms, host_total_ms; };
template<class T> struct CudaResult { T value; CudaTiming timing; };
struct CudaDevice { std::string name; int major, minor; std::size_t total_memory; };
CudaDevice cuda_device();
CudaResult<double> cuda_masked_reduce(std::span<const double>, std::span<const std::uint8_t>);
CudaResult<std::vector<std::uint64_t>> cuda_tile_histogram(std::span<const std::uint8_t>, std::size_t,
    std::size_t, std::size_t, std::size_t, bool shared_bins = true);
CudaResult<std::vector<double>> cuda_stencil3x3(std::span<const double>, std::size_t, std::size_t);
}
