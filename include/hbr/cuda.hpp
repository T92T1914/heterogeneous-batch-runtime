#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>
#include <memory>
namespace hbr {
struct CudaTiming { double upload_ms, kernel_ms, download_ms, host_total_ms; };
template<class T> struct CudaResult { T value; CudaTiming timing; };
struct CudaDevice { std::string name; int major, minor; std::size_t total_memory; };
CudaDevice cuda_device();
CudaResult<double> cuda_masked_reduce(std::span<const double>, std::span<const std::uint8_t>);
CudaResult<std::vector<std::uint64_t>> cuda_tile_histogram(std::span<const std::uint8_t>, std::size_t,
    std::size_t, std::size_t, std::size_t, bool shared_bins = true);
CudaResult<std::vector<double>> cuda_stencil3x3(std::span<const double>, std::size_t, std::size_t);
struct CudaHistogramResult {
    std::uint64_t request_id;
    std::vector<std::uint64_t> value;
    CudaTiming timing;
};
// A synchronous, fixed-shape context. Use and destroy it on its creating thread
// with the creating CUDA device current. Concurrent access/destruction is unsupported.
// Inputs stay immutable until run returns. Returned vectors never alias the context.
class CudaHistogramContext {
public:
    static constexpr std::size_t maximum_device_bytes = 64 * 1024 * 1024;
    CudaHistogramContext(std::size_t rows, std::size_t cols, std::size_t tile_rows,
                         std::size_t tile_cols, bool shared_bins = true);
    ~CudaHistogramContext();
    CudaHistogramContext(const CudaHistogramContext&) = delete;
    CudaHistogramContext& operator=(const CudaHistogramContext&) = delete;
    CudaHistogramResult run(std::span<const std::uint8_t> image);
    // Idempotent on the owning thread. Throws on a CUDA synchronization error.
    void close();
    std::size_t device_bytes() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
