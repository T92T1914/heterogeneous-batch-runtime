#pragma once
#include <cstddef>
#include <string>
namespace hbr {
// This header includes no SYCL header, so an application compiles it with any C++20
// compiler and links the shared hbr::hbr_sycl library, which carries its own device code.
struct SyclDevice {
    std::string backend, platform, name, driver_version;
    std::size_t global_memory, local_memory;
    bool fp64, atomic64;
};
// Selects a GPU through the SYCL GPU selector, which honors ONEAPI_DEVICE_SELECTOR.
// Throws std::runtime_error when no GPU is available. It never falls back to a CPU device.
SyclDevice sycl_device();
}
