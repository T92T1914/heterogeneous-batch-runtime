#include "hbr/cuda.hpp"
#include "hbr/detail/validation.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace hbr {
namespace {
using Clock = std::chrono::steady_clock;
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
            checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            for (auto& e : event) checked(cudaEventCreate(&e));
        } catch (...) { cleanup(); throw; }
    }
    ~Operation() { cleanup(); }
    Operation(const Operation&) = delete;
    void cleanup() noexcept {
        // Work must finish before its buffers or stream are relinquished, even on failure.
        if (stream) cudaStreamSynchronize(stream);
        for (auto p : allocations) cudaFree(p);
        allocations.clear();
        for (auto& e : event) if (e) { cudaEventDestroy(e); e = nullptr; }
        if (stream) { cudaStreamDestroy(stream); stream = nullptr; }
    }
    template<class T> T* allocate(std::size_t count) {
        void* data = nullptr;
        checked(cudaMalloc(&data, product(count, sizeof(T))));
        try { allocations.push_back(data); } catch (...) { cudaFree(data); throw; }
        return static_cast<T*>(data);
    }
    void mark(int index) { checked(cudaEventRecord(event[index], stream)); }
    CudaTiming finish() {
        mark(3);
        checked(cudaStreamSynchronize(stream));
        float upload, kernel, download;
        checked(cudaEventElapsedTime(&upload, event[0], event[1]));
        checked(cudaEventElapsedTime(&kernel, event[1], event[2]));
        checked(cudaEventElapsedTime(&download, event[2], event[3]));
        cleanup();
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
}
