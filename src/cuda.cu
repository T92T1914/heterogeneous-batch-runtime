#include "hbr/cuda.hpp"
#include "hbr/detail/validation.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>
#ifdef HBR_TEST_CUDA_FAULTS
#include "cuda_faults.hpp"
namespace hbr::test {
thread_local int failure = -1, count = 0, live = 0, drain_count = 0, drain_error_count = 0;
void fail_at(int point) { failure = point; count = 0; }
int checkpoints() { return count; }
int live_resources() { return live; }
int drains() { return drain_count; }
int drain_errors() { return drain_error_count; }
}
#endif

namespace hbr {
namespace {
using Clock = std::chrono::steady_clock;
void fault_point() {
#ifdef HBR_TEST_CUDA_FAULTS
    if (test::count++ == test::failure) throw std::runtime_error("injected host exception");
#endif
}
void resource_change(int delta) {
#ifdef HBR_TEST_CUDA_FAULTS
    test::live += delta;
#else
    (void)delta;
#endif
}
void checked(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("shape overflow");
    return a * b;
}
void finite(std::span<const double> data) {
    for (double value : data) if (!std::isfinite(value)) throw std::invalid_argument("input values must be finite");
}
struct Operation {
    cudaStream_t stream{};
    cudaEvent_t event[4]{};
    std::vector<void*> allocations;
    Clock::time_point start;
    explicit Operation(Clock::time_point begin) : start(begin) {
        try {
            fault_point();
            checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            resource_change(1);
            for (auto& e : event) { fault_point(); checked(cudaEventCreate(&e)); resource_change(1); }
        } catch (...) { cleanup(); throw; }
    }
    ~Operation() { cleanup(); }
    Operation(const Operation&) = delete;
    void cleanup() noexcept {
        // Work must finish before its buffers or stream are relinquished, even on failure.
        if (stream) {
            const auto status = cudaStreamSynchronize(stream);
#ifdef HBR_TEST_CUDA_FAULTS
            if (status == cudaSuccess) ++test::drain_count;
            else ++test::drain_error_count;
#else
            (void)status;
#endif
        }
        for (auto p : allocations) { if (cudaFree(p) == cudaSuccess) resource_change(-1); }
        allocations.clear();
        for (auto& e : event) if (e) { if (cudaEventDestroy(e) == cudaSuccess) resource_change(-1); e = nullptr; }
        if (stream) { if (cudaStreamDestroy(stream) == cudaSuccess) resource_change(-1); stream = nullptr; }
    }
    template<class T> T* allocate(std::size_t count) {
        void* data = nullptr;
        fault_point();
        checked(cudaMalloc(&data, product(count, sizeof(T))));
        try { allocations.push_back(data); } catch (...) { cudaFree(data); throw; }
        resource_change(1);
        return static_cast<T*>(data);
    }
    void mark(int index) { fault_point(); checked(cudaEventRecord(event[index], stream)); }
    CudaTiming finish(bool release = true) {
        mark(3);
        fault_point();
        checked(cudaStreamSynchronize(stream));
        float upload, kernel, download;
        checked(cudaEventElapsedTime(&upload, event[0], event[1]));
        checked(cudaEventElapsedTime(&kernel, event[1], event[2]));
        checked(cudaEventElapsedTime(&download, event[2], event[3]));
        if (release) cleanup();
        return {upload, kernel, download, std::chrono::duration<double, std::milli>(Clock::now() - start).count()};
    }
};
__global__ void reduce_kernel(const double* values, const std::uint8_t* mask, std::size_t n, double* partial) {
    __shared__ double lanes[256];
    double value = 0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        if (!mask || mask[i]) value += values[i];
    lanes[threadIdx.x] = value;
    __syncthreads();
    for (unsigned stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) lanes[threadIdx.x] += lanes[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) partial[blockIdx.x] = lanes[0];
}
template<bool Shared> __global__ void histogram_kernel(const std::uint8_t* image, std::size_t rows,
    std::size_t cols, std::size_t tr, std::size_t tc, std::size_t nc, unsigned long long* output) {
    __shared__ unsigned long long bins[256];
    auto* destination = output + static_cast<std::size_t>(blockIdx.x) * 256;
    if constexpr (Shared) bins[threadIdx.x] = 0;
    else destination[threadIdx.x] = 0;
    __syncthreads();
    const auto r0 = (blockIdx.x / nc) * tr, c0 = (blockIdx.x % nc) * tc;
    const auto nr = (tr < rows - r0 ? tr : rows - r0), ncols = (tc < cols - c0 ? tc : cols - c0);
    for (std::size_t i = threadIdx.x; i < nr * ncols; i += blockDim.x) {
        const auto bin = image[(r0 + i / ncols) * cols + c0 + i % ncols];
        if constexpr (Shared) atomicAdd(bins + bin, 1ULL);
        else atomicAdd(destination + bin, 1ULL);
    }
    if constexpr (Shared) { __syncthreads(); destination[threadIdx.x] = bins[threadIdx.x]; }
}
__global__ void stencil_kernel(const double* image, double* output, std::size_t rows, std::size_t cols) {
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < rows * cols; i += gridDim.x * blockDim.x) {
        const auto r = i / cols, c = i % cols;
        if (!r || !c || r + 1 == rows || c + 1 == cols) { output[i] = image[i]; continue; }
        double sum = image[i-cols-1] / 9.0;
        sum += image[i-cols] / 9.0; sum += image[i-cols+1] / 9.0;
        sum += image[i-1] / 9.0; sum += image[i] / 9.0; sum += image[i+1] / 9.0;
        sum += image[i+cols-1] / 9.0; sum += image[i+cols] / 9.0; sum += image[i+cols+1] / 9.0;
        output[i] = fmax(-1.7976931348623157e308, fmin(1.7976931348623157e308, sum));
    }
}
unsigned blocks(std::size_t n) { return static_cast<unsigned>(std::min<std::size_t>(4096, n / 256 + (n % 256 != 0))); }
}
CudaDevice cuda_device() {
    cudaDeviceProp prop{};
    int device;
    checked(cudaGetDevice(&device)); checked(cudaGetDeviceProperties(&prop, device));
    return {prop.name, prop.major, prop.minor, prop.totalGlobalMem};
}
CudaResult<double> cuda_masked_reduce(std::span<const double> values, std::span<const std::uint8_t> mask) {
    const auto start = Clock::now();
    if (values.size() != mask.size()) throw std::invalid_argument("mask shape mismatch");
    finite(values);
    detail::reduction_range(values, mask);
    if (values.empty()) return {0, {0, 0, 0, std::chrono::duration<double, std::milli>(Clock::now()-start).count()}};
    Operation op(start);
    auto* x = op.allocate<double>(values.size()); auto* m = op.allocate<std::uint8_t>(mask.size());
    const auto nb = blocks(values.size());
    auto* partial = op.allocate<double>(nb); auto* result = op.allocate<double>(1);
    op.mark(0);
    checked(cudaMemcpyAsync(x, values.data(), values.size_bytes(), cudaMemcpyHostToDevice, op.stream));
    checked(cudaMemcpyAsync(m, mask.data(), mask.size_bytes(), cudaMemcpyHostToDevice, op.stream));
    op.mark(1);
    reduce_kernel<<<nb, 256, 0, op.stream>>>(x, m, values.size(), partial); checked(cudaGetLastError());
    reduce_kernel<<<1, 256, 0, op.stream>>>(partial, nullptr, nb, result); checked(cudaGetLastError());
    op.mark(2);
    double value;
    checked(cudaMemcpyAsync(&value, result, sizeof(double), cudaMemcpyDeviceToHost, op.stream));
    auto timing = op.finish();
    return {value, timing};
}
CudaResult<std::vector<std::uint64_t>> cuda_tile_histogram(std::span<const std::uint8_t> image,
    std::size_t rows, std::size_t cols, std::size_t tr, std::size_t tc, bool shared_bins) {
    const auto start = Clock::now();
    if (!tr || !tc) throw std::invalid_argument("tile dimensions must be positive");
    if (product(rows, cols) != image.size()) throw std::invalid_argument("image shape mismatch");
    const auto nr = rows / tr + (rows % tr != 0), nc = cols / tc + (cols % tc != 0);
    const auto tiles = product(nr, nc);
    if (tiles > 2147483647ULL) throw std::overflow_error("tile grid too large");
    std::vector<std::uint64_t> value(product(tiles, 256), 0);
    if (image.empty()) return {std::move(value), {0, 0, 0, std::chrono::duration<double, std::milli>(Clock::now()-start).count()}};
    Operation op(start);
    auto* x = op.allocate<std::uint8_t>(image.size());
    auto* bins = op.allocate<unsigned long long>(value.size());
    op.mark(0); checked(cudaMemcpyAsync(x, image.data(), image.size_bytes(), cudaMemcpyHostToDevice, op.stream));
    op.mark(1);
    if (shared_bins) histogram_kernel<true><<<static_cast<unsigned>(tiles),256,0,op.stream>>>(x,rows,cols,tr,tc,nc,bins);
    else histogram_kernel<false><<<static_cast<unsigned>(tiles),256,0,op.stream>>>(x,rows,cols,tr,tc,nc,bins);
    checked(cudaGetLastError()); op.mark(2);
    checked(cudaMemcpyAsync(value.data(), bins, product(value.size(), sizeof(std::uint64_t)), cudaMemcpyDeviceToHost, op.stream));
    auto timing = op.finish(); return {std::move(value), timing};
}
CudaResult<std::vector<double>> cuda_stencil3x3(std::span<const double> image, std::size_t rows, std::size_t cols) {
    const auto start = Clock::now();
    if (product(rows, cols) != image.size()) throw std::invalid_argument("image shape mismatch");
    finite(image); std::vector<double> value(image.size());
    if (image.empty()) return {std::move(value), {0, 0, 0, std::chrono::duration<double, std::milli>(Clock::now()-start).count()}};
    Operation op(start); auto* x = op.allocate<double>(image.size()); auto* output = op.allocate<double>(image.size());
    op.mark(0); checked(cudaMemcpyAsync(x, image.data(), image.size_bytes(), cudaMemcpyHostToDevice, op.stream));
    op.mark(1); stencil_kernel<<<blocks(image.size()),256,0,op.stream>>>(x,output,rows,cols); checked(cudaGetLastError());
    op.mark(2); checked(cudaMemcpyAsync(value.data(), output, image.size_bytes(), cudaMemcpyDeviceToHost, op.stream));
    auto timing = op.finish(); return {std::move(value), timing};
}

struct CudaHistogramContext::Impl {
    const std::size_t rows, cols, tr, tc, pixels, tiles, nc, bytes;
    const bool shared;
    const std::thread::id owner = std::this_thread::get_id();
    int device = 0;
    Operation op{Clock::now()};
    std::uint8_t* image = nullptr;
    unsigned long long* bins = nullptr;
    std::uint64_t sequence = 0;
    bool closed = false;
    bool failed = false;

    Impl(std::size_t r, std::size_t c, std::size_t tile_r, std::size_t tile_c, bool use_shared,
         std::size_t ntiles, std::size_t storage)
        : rows(r), cols(c), tr(tile_r), tc(tile_c), pixels(product(r,c)), tiles(ntiles),
          nc(c/tile_c + (c%tile_c != 0)), bytes(storage), shared(use_shared) {
        checked(cudaGetDevice(&device));
        if (pixels) { image = op.allocate<std::uint8_t>(pixels); bins = op.allocate<unsigned long long>(product(tiles,256)); }
    }
    void check_owner() const {
        if (std::this_thread::get_id() != owner) throw std::logic_error("context belongs to another thread");
        int current;
        checked(cudaGetDevice(&current));
        if (current != device) throw std::logic_error("context belongs to another CUDA device");
    }
};
CudaHistogramContext::CudaHistogramContext(std::size_t rows, std::size_t cols, std::size_t tr,
                                         std::size_t tc, bool shared) {
    if (!tr || !tc) throw std::invalid_argument("tile dimensions must be positive");
    const auto pixels = product(rows,cols);
    const auto tiles = product(rows/tr + (rows%tr != 0), cols/tc + (cols%tc != 0));
    if (tiles > 2147483647ULL) throw std::overflow_error("tile grid too large");
    const auto output_bytes = product(product(tiles,256),sizeof(std::uint64_t));
    if (pixels > maximum_device_bytes || output_bytes > maximum_device_bytes - pixels)
        throw std::length_error("histogram context exceeds 64 MiB device-storage limit");
    impl_ = std::make_unique<Impl>(rows,cols,tr,tc,shared,tiles,pixels+output_bytes);
}
CudaHistogramContext::~CudaHistogramContext() = default;
std::size_t CudaHistogramContext::device_bytes() const noexcept { return impl_->bytes; }
CudaHistogramResult CudaHistogramContext::run(std::span<const std::uint8_t> image) {
    const auto start = Clock::now();
    auto& p = *impl_;
    p.check_owner();
    if (p.closed || p.failed) throw std::logic_error("histogram context is retired");
    // Repeat the simple API's shape and output-size validation on every invocation.
    if (!p.tr || !p.tc || product(p.rows,p.cols) != image.size()) throw std::invalid_argument("image shape mismatch");
    const auto tiles = product(p.rows/p.tr+(p.rows%p.tr != 0), p.cols/p.tc+(p.cols%p.tc != 0));
    if (tiles > 2147483647ULL) throw std::overflow_error("tile grid too large");
    if (p.sequence == std::numeric_limits<std::uint64_t>::max()) throw std::overflow_error("request sequence exhausted");
    std::vector<std::uint64_t> value(product(tiles,256),0);
    const auto id = ++p.sequence;
    if (image.empty()) return {id,std::move(value),{0,0,0,std::chrono::duration<double,std::milli>(Clock::now()-start).count()}};
    p.op.start = start;
    try {
        p.op.mark(0); fault_point();
        checked(cudaMemcpyAsync(p.image,image.data(),image.size_bytes(),cudaMemcpyHostToDevice,p.op.stream));
        p.op.mark(1);
        if (p.shared) histogram_kernel<true><<<static_cast<unsigned>(tiles),256,0,p.op.stream>>>(p.image,p.rows,p.cols,p.tr,p.tc,p.nc,p.bins);
        else histogram_kernel<false><<<static_cast<unsigned>(tiles),256,0,p.op.stream>>>(p.image,p.rows,p.cols,p.tr,p.tc,p.nc,p.bins);
        checked(cudaGetLastError()); fault_point(); p.op.mark(2);
        checked(cudaMemcpyAsync(value.data(),p.bins,product(value.size(),sizeof(std::uint64_t)),cudaMemcpyDeviceToHost,p.op.stream));
        fault_point();
        auto timing = p.op.finish(false);
        return {id,std::move(value),timing};
    } catch (...) {
        // Drain before the host result leaves scope, including a failure after D2H submission.
        // A failed context never reuses storage, even if cleanup itself encounters a device error.
        p.failed = true;
        p.op.cleanup();
        throw;
    }
}
void CudaHistogramContext::close() {
    auto& p = *impl_;
    p.check_owner();
    if (p.closed) return;
    p.closed = true;
    const auto status = p.op.stream ? cudaStreamSynchronize(p.op.stream) : cudaSuccess;
    p.op.cleanup();
    checked(status);
}
}
