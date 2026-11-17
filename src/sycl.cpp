#include "hbr/sycl.hpp"
#include <sycl/sycl.hpp>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef HBR_TEST_SYCL_FAULTS
#include "sycl_faults.hpp"
namespace hbr::test {
thread_local int failure = -1, count = 0, live = 0, drain_count = 0, drain_error_count = 0;
void fail_at(int point) { failure = point; count = 0; }
int checkpoints() { return count; }
int live_allocations() { return live; }
int drains() { return drain_count; }
int drain_errors() { return drain_error_count; }
}
#endif

namespace hbr {
namespace {
constexpr std::size_t lanes = 256;
void fault_point() {
#ifdef HBR_TEST_SYCL_FAULTS
    if (test::count++ == test::failure) throw std::runtime_error("injected host exception");
#endif
}
void allocation_change(int delta) {
#ifdef HBR_TEST_SYCL_FAULTS
    test::live += delta;
#else
    (void)delta;
#endif
}
std::size_t product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b) throw std::overflow_error("shape overflow");
    return a * b;
}
std::string backend_name(sycl::backend backend) {
    switch (backend) {
    case sycl::backend::ext_oneapi_cuda: return "cuda";
    case sycl::backend::ext_oneapi_hip: return "hip";
    case sycl::backend::ext_oneapi_level_zero: return "level_zero";
    case sycl::backend::opencl: return "opencl";
    default: return "other";
    }
}
template<bool Local> class TileHistogram;
// The GPU selector's score, limited to GPUs that can run every kernel this library carries.
// On a host with GPUs from several backends, the plain GPU selector could prefer one that
// this build has no device image for.
int compatible_gpu(const sycl::device& device) {
    if (!device.is_gpu() || !sycl::is_compatible<TileHistogram<true>>(device)
        || !sycl::is_compatible<TileHistogram<false>>(device)) return -1;
    return sycl::gpu_selector_v(device);
}
sycl::device select_gpu() { return sycl::device([](const sycl::device& device) { return compatible_gpu(device); }); }
// One in order queue for each call. The destructor waits for every command on the queue
// before it frees device memory, so a failure after submission never releases memory or
// borrowed host input that queued work still uses.
class Operation {
public:
    explicit Operation(const sycl::device& device)
        : errors_(std::make_shared<std::exception_ptr>()),
          queue_(device, [errors = errors_](sycl::exception_list list) {
              // Kept, never thrown here, because a handler may run inside a destructor.
              for (const auto& error : list) if (!*errors) *errors = error;
          }, sycl::property::queue::in_order{}) {}
    ~Operation() {
        try {
            queue_.wait();
#ifdef HBR_TEST_SYCL_FAULTS
            ++test::drain_count;
#endif
        } catch (...) {
#ifdef HBR_TEST_SYCL_FAULTS
            ++test::drain_error_count;
#endif
        }
        for (auto* pointer : allocations_) {
            try { sycl::free(pointer, queue_); allocation_change(-1); } catch (...) {}
        }
    }
    Operation(const Operation&) = delete;
    Operation& operator=(const Operation&) = delete;
    template<class T> T* allocate(std::size_t count) {
        fault_point();
        allocations_.reserve(allocations_.size() + 1);
        T* pointer = sycl::malloc_device<T>(count, queue_);
        if (!pointer) throw std::runtime_error("SYCL: device allocation failed");
        allocations_.push_back(pointer);
        allocation_change(1);
        return pointer;
    }
    sycl::queue& queue() { return queue_; }
    void finish() {
        fault_point();
        queue_.wait_and_throw();
        if (*errors_) std::rethrow_exception(*errors_);
    }
private:
    std::shared_ptr<std::exception_ptr> errors_;
    sycl::queue queue_;
    std::vector<void*> allocations_;
};
// Mirrors histogram_kernel in src/cuda.cu: one work group of 256 work items for each tile,
// counting into work group local bins or with atomics straight into that tile's output.
template<bool Local> void launch(sycl::queue& queue, const std::uint8_t* image, std::size_t rows,
    std::size_t cols, std::size_t tr, std::size_t tc, std::size_t nc, std::size_t tiles, std::uint64_t* output) {
    queue.submit([&](sycl::handler& handler) {
        sycl::local_accessor<std::uint64_t, 1> shared(sycl::range<1>(lanes), handler);
        handler.parallel_for<TileHistogram<Local>>(sycl::nd_range<1>(tiles * lanes, lanes),
          [=](sycl::nd_item<1> item) {
            const std::size_t tile = item.get_group(0);
            const std::size_t lane = item.get_local_id(0);
            std::uint64_t* destination = output + tile * lanes;
            if constexpr (Local) shared[lane] = 0;
            else destination[lane] = 0;
            sycl::group_barrier(item.get_group());
            const auto r0 = (tile / nc) * tr, c0 = (tile % nc) * tc;
            const auto tile_rows = tr < rows - r0 ? tr : rows - r0;
            const auto tile_cols = tc < cols - c0 ? tc : cols - c0;
            for (std::size_t i = lane; i < tile_rows * tile_cols; i += lanes) {
                const auto bin = image[(r0 + i / tile_cols) * cols + c0 + i % tile_cols];
                if constexpr (Local) {
                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                        sycl::access::address_space::local_space> count(shared[bin]);
                    count.fetch_add(1);
                } else {
                    // Only this tile's work group touches its output, so work group scope suffices.
                    sycl::atomic_ref<std::uint64_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                        sycl::access::address_space::global_space> count(destination[bin]);
                    count.fetch_add(1);
                }
            }
            if constexpr (Local) {
                sycl::group_barrier(item.get_group());
                destination[lane] = shared[lane];
            }
        });
    });
}
}
SyclDevice sycl_device() {
    // A SYCL error leaves this library only as std::runtime_error, so an application
    // that never includes a SYCL header can still catch it.
    try {
        const sycl::device device = select_gpu();
        return {backend_name(device.get_backend()),
                device.get_platform().get_info<sycl::info::platform::name>(),
                device.get_info<sycl::info::device::name>(),
                device.get_info<sycl::info::device::driver_version>(),
                device.get_info<sycl::info::device::global_mem_size>(),
                device.get_info<sycl::info::device::local_mem_size>(),
                device.has(sycl::aspect::fp64), device.has(sycl::aspect::atomic64)};
    } catch (const sycl::exception& error) {
        throw std::runtime_error(std::string("SYCL: ") + error.what());
    }
}
std::vector<std::uint64_t> sycl_tile_histogram(std::span<const std::uint8_t> image, std::size_t rows,
    std::size_t cols, std::size_t tr, std::size_t tc, bool local_bins) {
    if (!tr || !tc) throw std::invalid_argument("tile dimensions must be positive");
    if (product(rows, cols) != image.size()) throw std::invalid_argument("image shape mismatch");
    const auto nr = rows / tr + (rows % tr != 0), nc = cols / tc + (cols % tc != 0);
    const auto tiles = product(nr, nc);
    if (tiles > 2147483647ULL) throw std::overflow_error("tile grid too large");
    // Declared before the operation, so on failure the queue drains before this storage goes.
    std::vector<std::uint64_t> value(product(tiles, lanes), 0);
    if (image.empty()) return value;
    try {
        const sycl::device device = select_gpu();
        if (!device.has(sycl::aspect::atomic64)) throw std::runtime_error("SYCL: the selected GPU lacks atomic64");
        Operation op(device);
        auto* x = op.allocate<std::uint8_t>(image.size());
        auto* bins = op.allocate<std::uint64_t>(value.size());
        fault_point();
        op.queue().memcpy(x, image.data(), image.size_bytes());
        fault_point();
        if (local_bins) launch<true>(op.queue(), x, rows, cols, tr, tc, nc, tiles, bins);
        else launch<false>(op.queue(), x, rows, cols, tr, tc, nc, tiles, bins);
        fault_point();
        op.queue().memcpy(value.data(), bins, product(value.size(), sizeof(std::uint64_t)));
        op.finish();
        return value;
    } catch (const sycl::exception& error) {
        throw std::runtime_error(std::string("SYCL: ") + error.what());
    }
}
}
