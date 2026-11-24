#include "hbr/cuda.hpp"
#include "hbr/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define HBR_NVML_WINDOWS 1
#elif __has_include(<dlfcn.h>)
#include <dlfcn.h>
#define HBR_NVML_DLOPEN 1
#endif

// Pinned versus pageable host input for the three existing CUDA calls.
// The runtime is unchanged: each call receives a span, and only the memory
// behind the input span differs. Outputs stay in the runtime's own vectors.
using Clock=std::chrono::steady_clock;
constexpr std::array<unsigned,3> sides{32,256,1024};
constexpr std::array<const char*,3> workloads{"masked_reduce","tile_histogram","stencil3x3"};
constexpr std::array<const char*,2> memories{"pageable","pinned"};
constexpr int samples=121, warmups=2;
constexpr std::size_t tile=64;
constexpr int combinations=static_cast<int>(workloads.size()*memories.size());
constexpr auto idle_pause=std::chrono::seconds(2);

void checked(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(status));
}
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }

// The workload and host memory at run position `order` of iteration `iteration` for the side
// at index `condition`. Each iteration takes the three workloads in turn, rotated by
// iteration+condition, and runs each one with pageable and pinned input back to back.
// Pageable goes first in even iterations and pinned in odd ones, so over any six iterations
// each workload holds each place in the round with each memory kind first exactly once.
std::pair<int,int> slot(int condition,int iteration,int order) {
    return {(iteration+condition+order/2)%3, (order%2+iteration%2)%2};
}

// The NVIDIA management library, opened at run time so that the build needs no NVML header
// or stub library. The declarations follow NVML's public C API: nvmlUtilization_t, the
// version 1 nvmlMemory_t, and nvmlReturn_t and nvmlDevice_t as an int and a pointer. On
// Windows the driver installs nvml.dll in System32, and only that folder is searched.
class GpuMonitor {
public:
    struct Reading { unsigned util_percent; unsigned long long memory_mib; };
    explicit GpuMonitor(const std::string& uuid) {
#if defined(HBR_NVML_WINDOWS)
        library_=LoadLibraryExW(L"nvml.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!library_) throw std::runtime_error("NVML: nvml.dll could not be opened from System32");
#elif defined(HBR_NVML_DLOPEN)
        library_=dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL);
        if (!library_) throw std::runtime_error("NVML: libnvidia-ml.so.1 could not be opened");
#else
        (void)uuid;
        throw std::runtime_error("NVML: this platform cannot open a library at run time, so the GPU cannot be read");
#endif
        try {
            const auto init=symbol<int(*)()>("nvmlInit_v2");
            const auto shutdown=symbol<int(*)()>("nvmlShutdown");
            const auto by_uuid=symbol<int(*)(const char*,void**)>("nvmlDeviceGetHandleByUUID");
            error_=symbol<const char*(*)(int)>("nvmlErrorString");
            utilization_=symbol<int(*)(void*,Utilization*)>("nvmlDeviceGetUtilizationRates");
            memory_=symbol<int(*)(void*,Memory*)>("nvmlDeviceGetMemoryInfo");
            check(init(),"nvmlInit_v2");
            shutdown_=shutdown;
            check(by_uuid(uuid.c_str(),&device_),"nvmlDeviceGetHandleByUUID");
        } catch (...) { close(); throw; }
    }
    GpuMonitor(const GpuMonitor&)=delete;
    GpuMonitor& operator=(const GpuMonitor&)=delete;
    ~GpuMonitor() { close(); }
    // Utilization is the percent of the last NVML sample period, between 1/6 and 1 second,
    // in which any kernel of any process ran. Memory is what all processes hold on the card.
    Reading read() const {
        Utilization u{}; Memory m{};
        check(utilization_(device_,&u),"nvmlDeviceGetUtilizationRates");
        check(memory_(device_,&m),"nvmlDeviceGetMemoryInfo");
        return {u.gpu, m.used/(1024ull*1024ull)};
    }
private:
    struct Utilization { unsigned int gpu, memory; };
    struct Memory { unsigned long long total, free, used; };
    template<class F> F symbol(const char* name) const {
#if defined(HBR_NVML_WINDOWS)
        if (auto found=GetProcAddress(static_cast<HMODULE>(library_),name)) return reinterpret_cast<F>(found);
#elif defined(HBR_NVML_DLOPEN)
        if (auto* found=dlsym(library_,name)) return reinterpret_cast<F>(found);
#endif
        throw std::runtime_error(std::string("NVML: no symbol ")+name);
    }
    void check(int status,const char* call) const {
        if (status) throw std::runtime_error(std::string("NVML: ")+call+": "+(error_ ? error_(status) : std::to_string(status)));
    }
    void close() {
        if (shutdown_) shutdown_();
#if defined(HBR_NVML_WINDOWS)
        if (library_) FreeLibrary(static_cast<HMODULE>(library_));
#elif defined(HBR_NVML_DLOPEN)
        if (library_) dlclose(library_);
#endif
        shutdown_=nullptr; library_=nullptr;
    }
    void* library_=nullptr; void* device_=nullptr;
    int (*shutdown_)()=nullptr;
    int (*utilization_)(void*,Utilization*)=nullptr;
    int (*memory_)(void*,Memory*)=nullptr;
    const char* (*error_)(int)=nullptr;
};

// A reading after a pause long enough to cover a whole NVML sample period, so this process
// adds no kernel time to it. It goes to stderr, which the runner keeps in the receipt.
void idle_reading(const GpuMonitor& monitor,const std::string& point) {
    std::this_thread::sleep_for(idle_pause);
    const auto r=monitor.read();
    std::cerr<<"GPU idle "<<point<<": utilization "<<r.util_percent<<" percent, memory "<<r.memory_mib<<" MiB\n";
}

// The UUID of the current CUDA device in NVML's form, so that NVML reads the card the
// CUDA calls use. It is never printed, since the evidence is public.
std::string cuda_uuid() {
    int device=0;
    checked(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    checked(cudaGetDeviceProperties(&prop,device));
    std::ostringstream uuid;
    uuid<<"GPU-"<<std::hex<<std::setfill('0');
    for (int i=0;i<16;++i) {
        if (i==4 || i==6 || i==8 || i==10) uuid<<'-';
        uuid<<std::setw(2)<<static_cast<unsigned>(static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    return uuid.str();
}

template<class T> class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        if (count) checked(cudaMallocHost(reinterpret_cast<void**>(&data_),count*sizeof(T)));
    }
    ~PinnedBuffer() { if (data_) cudaFreeHost(data_); }
    PinnedBuffer(const PinnedBuffer&)=delete;
    PinnedBuffer& operator=(const PinnedBuffer&)=delete;
    void release() {
        if (!data_) return;
        const auto status=cudaFreeHost(data_);
        data_=nullptr;
        checked(status);
    }
    std::span<T> span() const { return {data_,count_}; }
private:
    T* data_=nullptr;
    std::size_t count_;
};

struct Inputs {
    std::span<const double> values;
    std::span<const std::uint8_t> mask, image;
};
struct Sample {
    hbr::CudaTiming timing{}; double reduced=0; std::vector<double> real; std::vector<std::uint64_t> counts;
    std::span<const double> values() const { return real.empty() ? std::span<const double>(&reduced,1) : std::span<const double>(real); }
};

Sample execute(int workload,const Inputs& in,unsigned side) {
    Sample sample;
    const auto start=Clock::now();
    if (workload==0) {
        auto r=hbr::cuda_masked_reduce(in.values,in.mask);
        sample.timing=r.timing; sample.reduced=r.value;
    } else if (workload==1) {
        auto r=hbr::cuda_tile_histogram(in.image,side,side,tile,tile);
        sample.timing=r.timing; sample.counts=std::move(r.value);
    } else {
        auto r=hbr::cuda_stencil3x3(in.values,side,side);
        sample.timing=r.timing; sample.real=std::move(r.value);
    }
    // The clock stops with an owned completed result, before comparison and destruction.
    sample.timing.host_total_ms=elapsed(start);
    return sample;
}
std::size_t upload_bytes(int workload,std::size_t pixels) {
    return workload==0 ? pixels*(sizeof(double)+1) : workload==1 ? pixels : pixels*sizeof(double);
}
std::size_t download_bytes(int workload,unsigned side) {
    const std::size_t tiles=(side/tile+(side%tile!=0))*(side/tile+(side%tile!=0));
    return workload==0 ? sizeof(double) : workload==1 ? tiles*256*sizeof(std::uint64_t)
        : static_cast<std::size_t>(side)*side*sizeof(double);
}
double checksum(const Sample& s) {
    double sum=0;
    if (!s.counts.empty()) { for (std::size_t i=0;i<s.counts.size();++i) sum+=static_cast<double>(s.counts[i]*(i+1)); }
    else { const auto v=s.values(); for (std::size_t i=0;i<v.size();++i) sum+=v[i]*static_cast<double>(i%7+1); }
    return sum;
}
void row(const char* phase,unsigned side,int workload,int memory,int iteration,int order,
         const hbr::CudaTiming& t,double sum,std::size_t up,std::size_t down,const GpuMonitor::Reading& gpu) {
    std::cout<<phase<<','<<side<<','<<(workload<0?"all":workloads[workload])<<','<<memories[memory]<<','
        <<iteration<<','<<order<<','<<t.host_total_ms<<','<<t.upload_ms<<','<<t.kernel_ms<<','
        <<t.download_ms<<','<<sum<<','<<up<<','<<down<<','<<gpu.util_percent<<','<<gpu.memory_mib<<'\n';
}
// Every run row's side, workload, host memory, iteration and order, in the order --run prints
// them, without touching a device, so the hosted job can compare it with the report's.
void sequence() {
    std::cout<<"side,workload,host_memory,iteration,order\n";
    for (int condition=0;condition<static_cast<int>(sides.size());++condition)
        for (int iteration=0;iteration<samples;++iteration)
            for (int order=0;order<combinations;++order) {
                const auto [workload,memory]=slot(condition,iteration,order);
                std::cout<<sides[condition]<<','<<workloads[workload]<<','<<memories[memory]<<','<<iteration<<','<<order<<'\n';
            }
}
void plan() {
    std::cout<<"{\"square_sides\": [";
    for (std::size_t i=0;i<sides.size();++i) std::cout<<(i?", ":"")<<sides[i];
    std::cout<<"], \"workloads\": [";
    for (std::size_t i=0;i<workloads.size();++i) std::cout<<(i?", ":"")<<'"'<<workloads[i]<<'"';
    std::cout<<"], \"host_memory\": [";
    for (std::size_t i=0;i<memories.size();++i) std::cout<<(i?", ":"")<<'"'<<memories[i]<<'"';
    std::cout<<"], \"tile_shape\": ["<<tile<<", "<<tile<<"], \"samples_per_condition\": "<<samples
        <<", \"additional_unrecorded_warmups\": "<<warmups
        <<", \"run_rows\": "<<sides.size()*combinations*samples<<"}\n";
}

int main(int argc,char** argv) {
    try {
        if (argc!=2) return 2;
        const std::string mode=argv[1];
        if (mode=="--plan") { plan(); return 0; }
        if (mode=="--sequence") { sequence(); return 0; }
        if (mode!="--run") return 2;
        const auto device=hbr::cuda_device();
        std::cerr<<device.name<<" SM "<<device.major<<'.'<<device.minor<<'\n';
        // Create the primary context before any recorded row, so no pinned allocation pays for it.
        checked(cudaFree(nullptr));
        const GpuMonitor monitor(cuda_uuid());
        std::cerr<<"NVML: the CUDA device found by its UUID\n";
        std::cout<<"phase,side,workload,host_memory,iteration,order,host_ms,upload_ms,kernel_ms,download_ms,checksum,upload_bytes,download_bytes,gpu_util_percent,gpu_memory_mib\n"
            <<std::setprecision(17);
        for (int condition=0;condition<static_cast<int>(sides.size());++condition) {
            const unsigned side=sides[condition];
            idle_reading(monitor,"before "+std::to_string(side));
            const std::size_t pixels=static_cast<std::size_t>(side)*side;
            std::vector<double> values(pixels);
            std::vector<std::uint8_t> mask(pixels), image(pixels);
            for (std::size_t i=0;i<pixels;++i) {
                values[i]=static_cast<double>(i%1009)/1009-0.5; mask[i]=i%3!=0; image[i]=static_cast<std::uint8_t>(i*13);
            }
            const std::size_t pinned_bytes=pixels*(sizeof(double)+2);
            // Every reading is taken before its clock starts, so none is inside a timing.
            auto gpu=monitor.read();
            auto start=Clock::now();
            PinnedBuffer<double> pinned_values(pixels);
            PinnedBuffer<std::uint8_t> pinned_mask(pixels), pinned_image(pixels);
            row("setup",side,-1,1,-1,-1,{0,0,0,elapsed(start)},0,pinned_bytes,0,gpu);
            std::copy(values.begin(),values.end(),pinned_values.span().begin());
            std::copy(mask.begin(),mask.end(),pinned_mask.span().begin());
            std::copy(image.begin(),image.end(),pinned_image.span().begin());
            if (std::memcmp(values.data(),pinned_values.span().data(),pixels*sizeof(double))
                || std::memcmp(mask.data(),pinned_mask.span().data(),pixels)
                || std::memcmp(image.data(),pinned_image.span().data(),pixels))
                throw std::runtime_error("pinned input differs from pageable input");
            const std::array<Inputs,2> inputs{
                Inputs{values,mask,image},
                Inputs{pinned_values.span(),pinned_mask.span(),pinned_image.span()}};
            // Independent references: the scalar CPU path for every workload.
            const double reduce_reference=hbr::masked_reduce(values,mask,{hbr::Backend::scalar,1});
            const auto histogram_reference=hbr::tile_histogram(image,side,side,tile,tile,{hbr::Backend::scalar,1});
            const auto stencil_reference=hbr::stencil3x3(values,side,side,{hbr::Backend::scalar,1});
            std::array<std::vector<double>,3> first_real;
            auto verify=[&](int workload,const Sample& s) {
                if (workload==1) {
                    if (s.counts!=histogram_reference) throw std::runtime_error("histogram mismatch");
                    return;
                }
                const auto v=s.values();
                // Written as !(difference <= tolerance) so that a NaN output fails.
                if (workload==0 ? !(std::abs(v[0]-reduce_reference)<=1e-9) : v.size()!=stencil_reference.size())
                    throw std::runtime_error("reference mismatch");
                if (workload==2) for (std::size_t i=0;i<v.size();++i)
                    if (!(std::abs(v[i]-stencil_reference[i])<=1e-12)) throw std::runtime_error("stencil mismatch");
                // Same kernel and same bytes must give the same bits from either host memory.
                if (first_real[workload].empty()) first_real[workload].assign(v.begin(),v.end());
                else if (std::memcmp(first_real[workload].data(),v.data(),v.size()*sizeof(double)))
                    throw std::runtime_error("pinned and pageable results differ");
            };
            for (int iteration=0;iteration<samples;++iteration) {
                for (int order=0;order<combinations;++order) {
                    const auto [workload,memory]=slot(condition,iteration,order);
                    gpu=monitor.read();
                    const auto sample=execute(workload,inputs[memory],side);
                    verify(workload,sample);
                    row("run",side,workload,memory,iteration,order,sample.timing,checksum(sample),
                        upload_bytes(workload,pixels),download_bytes(workload,side),gpu);
                }
                if (!iteration) for (int warm=0;warm<warmups;++warm) for (int index=0;index<combinations;++index)
                    verify(index%3,execute(index%3,inputs[index/3],side));
            }
            gpu=monitor.read();
            start=Clock::now();
            pinned_values.release(); pinned_mask.release(); pinned_image.release();
            row("retire",side,-1,1,-1,-1,{0,0,0,elapsed(start)},0,pinned_bytes,0,gpu);
        }
        idle_reading(monitor,"after "+std::to_string(sides.back()));
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
