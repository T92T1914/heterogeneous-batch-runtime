# The SYCL backend

The optional SYCL backend is built with the DPC++ compiler for NVIDIA hardware. In this revision it selects a GPU, reports it and runs the tiled histogram. The masked reduction and the stencil come one at a time, each with its own executed correctness and lifecycle tests. The [backend discovery contract](backend-discovery.md) covers the installed `sycl` component, and the [open gates page](limits.md#a-second-gpu-vendor) explains why this backend does not close the second vendor gate.

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

`hbr::sycl_device()` selects a GPU with the SYCL GPU selector, which honors `ONEAPI_DEVICE_SELECTOR`, and reports its backend, platform, name, driver, memory and its `fp64` and `atomic64` aspects. It considers only GPUs that can run every kernel the library carries, so on a host with GPUs from more than one backend it skips one this NVIDIA build has no device image for, and one that lacks an aspect a kernel needs. That skip follows from the code and the SYCL specification. It has not run on a host with GPUs from more than one backend. It never selects a CPU device. A SYCL error leaves the library only as `std::runtime_error`, so an application that includes no SYCL header can still catch it.

## Tiled histogram

`hbr::sycl_tile_histogram(image, rows, cols, tile_rows, tile_cols, local_bins)` is the counterpart of `hbr::cuda_tile_histogram`. It returns 256 exact `uint64` counts for each tile, tiles in row order, with the partial tiles at the right and bottom edges counted like whole ones. One work group of 256 work items counts each tile. With `local_bins`, the default, it counts into work group local bins and copies them out once. Without it, it counts with atomics straight into that tile's output. Both need the `atomic64` aspect, so the selector skips a GPU without it.

Zero tile dimensions and a shape that does not match the image throw `std::invalid_argument`, and a tile grid of more than 8,388,607 tiles throws `std::overflow_error`, all before a device is selected. That limit comes from DPC++, which by default requires a kernel's global range to fit in `int`, and each tile takes 256 work items. `hbr::cuda_tile_histogram` accepts larger grids. An empty image returns an empty result without selecting one. Each call is a completed synchronous call on its own in order queue. The input is borrowed until the call returns. Every queued copy and kernel has finished before the call returns or throws, device memory is freed only after that, and the result owns its storage. An asynchronous SYCL error is kept and thrown after the queue has drained, as `std::runtime_error` like every other SYCL error.

It is a correctness and lifecycle port. It measures no time and makes no performance claim.

## Tests

Tests whose names start with `sycl_` need a SYCL GPU. `sycl_device` checks the report. `sycl_histogram` compares 60 cases with the scalar reference, in both bin modes: the shapes of the CUDA histogram test and 32, 256 and 1024 square images in 64 by 64 tiles with spread and skewed values. It also checks the rejections, that the input is left unchanged and that results own their storage, eight concurrent jobs through a `Runtime` with two workers, and both kinds of cancellation. Cancelling a running job stays a request and the job still completes correctly, while cancelling a queued one is confirmed and it never runs. `sycl_faults` compiles the library again with six test checkpoints, from the two allocations to the final wait, and fails at each one in each bin mode, once with a host exception and once with a `sycl::exception`, which must arrive as `std::runtime_error` with the `SYCL: ` prefix. It then leaves an asynchronous SYCL error for completion to find, as the queue's handler would, and that error must be thrown after the drain. Every time the queue must drain without an error, no device allocation may remain and the next call must be correct. The handler itself never runs in these tests, because no real device error occurs. `host_sycl_selection` hides every GPU with `ONEAPI_DEVICE_SELECTOR=*:cpu`, so it runs on any host. There the histogram must still reject bad shapes with `std::invalid_argument` and an oversized tile grid with `std::overflow_error`, and return an empty result for an empty image, and a real image and the device query must fail with `std::runtime_error`. On a host with no GPU at all it shows only that the error is converted, not that the selector setting is honored.

The hosted [`SYCL compile`](../.github/workflows/sycl-compile.yml) workflow fetches the DPC++ 7.2.0 release and the CUDA 13.2.2 device tools by SHA-256, builds the backend for `sm_89` and checks that the library carries that device image. It then runs every test whose name does not start with `sycl_`, followed by the installed discovery check with `--sycl-built`. It executes nothing on a device.
