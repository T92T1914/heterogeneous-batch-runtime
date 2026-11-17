#include "hbr/sycl.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <string>

namespace hbr {
namespace {
std::string backend_name(sycl::backend backend) {
    switch (backend) {
    case sycl::backend::ext_oneapi_cuda: return "cuda";
    case sycl::backend::ext_oneapi_hip: return "hip";
    case sycl::backend::ext_oneapi_level_zero: return "level_zero";
    case sycl::backend::opencl: return "opencl";
    default: return "other";
    }
}
}
SyclDevice sycl_device() {
    // A SYCL error leaves this library only as std::runtime_error, so an application
    // that never includes a SYCL header can still catch it.
    try {
        const sycl::device device(sycl::gpu_selector_v);
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
}
