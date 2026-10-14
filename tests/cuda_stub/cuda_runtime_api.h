#pragma once
// Host-only fixture. It neither links CUDA nor creates a device context.
using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "host stand-in"; }
