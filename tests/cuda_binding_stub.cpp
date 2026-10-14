#include "hbr/cuda.hpp"
#include <stdexcept>
#include <thread>

// Link the actual Python binding to a deliberately synthetic host context.
namespace hbr {
struct CudaHistogramContext::Impl {
    std::size_t rows, cols, tr, tc;
    std::uint64_t request_id = 0;
    bool closed = false;
    std::thread::id owner = std::this_thread::get_id();
};
CudaDevice cuda_device() { return {"HOST STAND-IN, NO CUDA EXECUTION", 0, 0, 0}; }
CudaHistogramContext::CudaHistogramContext(std::size_t r, std::size_t c, std::size_t tr, std::size_t tc, bool)
    : impl_(std::make_unique<Impl>()) {
    if (!tr || !tc || r > 1024 || c > 1024) throw std::invalid_argument("fixture shape limit");
    impl_->rows = r; impl_->cols = c; impl_->tr = tr; impl_->tc = tc;
}
CudaHistogramContext::~CudaHistogramContext() = default;
CudaHistogramResult CudaHistogramContext::run(std::span<const std::uint8_t> image) {
    auto& p = *impl_;
    if (p.owner != std::this_thread::get_id() || p.closed) throw std::logic_error("fixture ownership");
    if (image.size() != p.rows*p.cols) throw std::invalid_argument("fixture fixed image size");
    const auto nr = p.rows/p.tr + (p.rows%p.tr != 0), nc = p.cols/p.tc + (p.cols%p.tc != 0);
    std::vector<std::uint64_t> values(nr*nc*256);
    for (std::size_t r = 0; r < p.rows; ++r)
        for (std::size_t c = 0; c < p.cols; ++c)
            ++values[((r/p.tr)*nc + c/p.tc)*256 + image[r*p.cols+c]];
    return {++p.request_id, std::move(values), {0,0,0,0}};
}
void CudaHistogramContext::close() { impl_->closed = true; }
std::size_t CudaHistogramContext::device_bytes() const noexcept { return 0; }
}
