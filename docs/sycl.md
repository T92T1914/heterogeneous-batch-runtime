# The SYCL backend

The optional SYCL backend is built with the DPC++ compiler for NVIDIA hardware. In this revision it selects a GPU and reports it, and it runs none of the three workloads yet. The histogram, the masked reduction and the stencil come one at a time, each with its own executed correctness and lifecycle tests. The [backend discovery contract](backend-discovery.md) covers the installed `sycl` component, and the [open gates page](limits.md#a-second-gpu-vendor) explains why this backend does not close the second vendor gate.

## Build

The backend is built and tested only on Linux with DPC++. Configure the whole project with that compiler, because CMake uses one C++ compiler for every target. `-fsycl` is applied to the SYCL library alone, and the other targets compile as ordinary C++. The linker and the tests need the DPC++ runtime libraries, so put them on the loader path first.

```sh
export LD_LIBRARY_PATH=<dpcpp>/lib:$LD_LIBRARY_PATH
cmake -S . -B build/sycl -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=<dpcpp>/bin/clang++ \
  -DHBR_SYCL=ON -DHBR_SYCL_FLAGS=--cuda-path=<cuda-toolkit>
cmake --build build/sycl --parallel 2
ctest --test-dir build/sycl --output-on-failure
```

`HBR_SYCL_TARGETS` sets `-fsycl-targets` and defaults to `nvidia_gpu_sm_89`, which compiles the device code ahead of time for an RTX 4090. `HBR_SYCL_FLAGS` adds options to both compiling and linking, such as the CUDA toolkit path that the NVIDIA target needs for `ptxas` and libdevice. CMake drops a repeated option from such a list, so an option that must repeat, like `-Xsycl-target-backend`, needs the `SHELL:` prefix. Windows is refused at configure time, because the DPC++ Windows release has no CUDA adapter and nothing here was built there.

`hbr::hbr_sycl` is a shared library, so it carries its own device image, and `hbr/sycl.hpp` includes no SYCL header. An application compiled by g++ or any other C++20 compiler links it with no SYCL compiler, as long as the DPC++ runtime libraries are on the loader path.

## Device selection

`hbr::sycl_device()` selects a GPU with the SYCL GPU selector, which honors `ONEAPI_DEVICE_SELECTOR`, and reports its backend, platform, name, driver, memory and its `fp64` and `atomic64` aspects. It never selects a CPU device. A SYCL error leaves the library only as `std::runtime_error`, so an application that includes no SYCL header can still catch it. On a host with GPUs from more than one backend, the selector may prefer a GPU this NVIDIA build cannot run kernels on. Set `ONEAPI_DEVICE_SELECTOR=cuda:gpu` there.

## Tests

Tests whose names start with `sycl_` need a SYCL GPU. `sycl_device` checks the report. `host_sycl_selection` hides every GPU with `ONEAPI_DEVICE_SELECTOR=*:cpu` and expects the `std::runtime_error`, so it runs on any host. On a host with no GPU at all it shows only that the error is converted, not that the selector setting is honored.

The hosted [`SYCL compile`](../.github/workflows/sycl-compile.yml) workflow fetches the DPC++ 7.2.0 release and the CUDA 13.2.2 device tools by SHA-256, builds the backend for `sm_89` and checks that the library carries that device image. It then runs every test whose name does not start with `sycl_`, followed by the installed discovery check with `--sycl-built`. It executes nothing on a device.
