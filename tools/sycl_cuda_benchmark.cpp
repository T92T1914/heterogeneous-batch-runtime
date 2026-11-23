#include "hbr/cuda.hpp"
#include "hbr/kernels.hpp"
#include "hbr/sycl.hpp"
#include <cuda_runtime.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#if __has_include(<dlfcn.h>)
#include <dlfcn.h>
#define HBR_HAS_DLOPEN 1
#endif

// The complete CUDA and SYCL calls of the three workloads, timed at one host boundary and
// interleaved in one process. Both libraries are used unchanged through their public calls,
// with pageable host input and results that each library allocates and returns.
using Clock=std::chrono::steady_clock;
constexpr std::array<unsigned,3> sides{32,256,1024};
constexpr std::array<const char*,3> workloads{"masked_reduce","tile_histogram","stencil3x3"};
constexpr std::array<const char*,2> backends{"cuda","sycl"};
constexpr int samples=121, warmups=2;
constexpr std::size_t tile=64;
constexpr int combinations=static_cast<int>(workloads.size()*backends.size());
constexpr auto idle_pause=std::chrono::seconds(2);

void checked(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(status));
}
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }

// The workload and backend at run position `order` of iteration `iteration` for the side at
// index `condition`. Each iteration takes the three workloads in turn, rotated by
// iteration+condition, and runs each one as a CUDA call and a SYCL call back to back. CUDA
// goes first in even iterations and SYCL in odd ones, so over any six iterations each
// workload holds each place in the round with each backend first exactly once.
std::pair<int,int> slot(int condition,int iteration,int order) {
    return {(iteration+condition+order/2)%3, (order%2+iteration%2)%2};
}

// The NVIDIA management library, opened at run time so that the build needs no NVML header
// or stub library. The declarations follow NVML's public C API: nvmlUtilization_t, the
// version 1 nvmlMemory_t, and nvmlReturn_t and nvmlDevice_t as an int and a pointer.
class GpuMonitor {
public:
    struct Reading { unsigned util_percent; unsigned long long memory_mib; };
    explicit GpuMonitor(const std::string& uuid) {
#ifdef HBR_HAS_DLOPEN
        library_=dlopen("libnvidia-ml.so.1",RTLD_NOW|RTLD_LOCAL);
        if (!library_) throw std::runtime_error("NVML: libnvidia-ml.so.1 could not be opened");
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
#else
        (void)uuid;
        throw std::runtime_error("NVML: this platform has no dlopen, so the GPU cannot be read");
#endif
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
#ifdef HBR_HAS_DLOPEN
        if (auto* found=dlsym(library_,name)) return reinterpret_cast<F>(found);
#endif
        throw std::runtime_error(std::string("NVML: no symbol ")+name);
    }
    void check(int status,const char* call) const {
        if (status) throw std::runtime_error(std::string("NVML: ")+call+": "+(error_ ? error_(status) : std::to_string(status)));
    }
    void close() {
        if (shutdown_) shutdown_();
#ifdef HBR_HAS_DLOPEN
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

struct Inputs {
    std::span<const double> values;
    std::span<const std::uint8_t> mask, image;
};
struct Sample {
    // Only the CUDA library records device events. The SYCL library records none.
    bool device_timing=false; hbr::CudaTiming timing{}; double host_ms=0;
    double reduced=0; std::vector<double> real; std::vector<std::uint64_t> counts;
    std::span<const double> values() const { return real.empty() ? std::span<const double>(&reduced,1) : std::span<const double>(real); }
};

Sample execute(int workload,int backend,const Inputs& in,unsigned side) {
    Sample sample;
    const auto start=Clock::now();
    if (backend==0) {
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
        sample.device_timing=true;
    } else {
        if (workload==0) sample.reduced=hbr::sycl_masked_reduce(in.values,in.mask);
        else if (workload==1) sample.counts=hbr::sycl_tile_histogram(in.image,side,side,tile,tile);
        else sample.real=hbr::sycl_stencil3x3(in.values,side,side);
    }
    // The clock stops with an owned completed result, before comparison and destruction,
    // at the same point for both backends.
    sample.host_ms=elapsed(start);
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
bool same_bits(const Sample& a,const Sample& b) {
    if (!a.counts.empty() || !b.counts.empty()) return a.counts==b.counts;
    const auto x=a.values(), y=b.values();
    return x.size()==y.size() && !std::memcmp(x.data(),y.data(),x.size_bytes());
}
void row(unsigned side,int workload,int backend,int iteration,int order,const Sample& s,std::size_t pixels,
         const GpuMonitor::Reading& gpu) {
    std::cout<<side<<','<<workloads[workload]<<','<<backends[backend]<<','<<iteration<<','<<order<<','<<s.host_ms<<',';
    // A SYCL row leaves the device fields empty, so no zero reads as a measurement.
    if (s.device_timing) std::cout<<s.timing.upload_ms<<','<<s.timing.kernel_ms<<','<<s.timing.download_ms<<',';
    else std::cout<<",,,";
    std::cout<<checksum(s)<<','<<upload_bytes(workload,pixels)<<','<<download_bytes(workload,side)<<','
        <<gpu.util_percent<<','<<gpu.memory_mib<<'\n';
}
// Every run row's side, workload, backend, iteration and order, in the order --run prints
// them, without touching a device, so the hosted job can compare it with the report's.
void sequence() {
    std::cout<<"side,workload,backend,iteration,order\n";
    for (int condition=0;condition<static_cast<int>(sides.size());++condition)
        for (int iteration=0;iteration<samples;++iteration)
            for (int order=0;order<combinations;++order) {
                const auto [workload,backend]=slot(condition,iteration,order);
                std::cout<<sides[condition]<<','<<workloads[workload]<<','<<backends[backend]<<','<<iteration<<','<<order<<'\n';
            }
}
void plan() {
    std::cout<<"{\"square_sides\": [";
    for (std::size_t i=0;i<sides.size();++i) std::cout<<(i?", ":"")<<sides[i];
    std::cout<<"], \"workloads\": [";
    for (std::size_t i=0;i<workloads.size();++i) std::cout<<(i?", ":"")<<'"'<<workloads[i]<<'"';
    std::cout<<"], \"backends\": [";
    for (std::size_t i=0;i<backends.size();++i) std::cout<<(i?", ":"")<<'"'<<backends[i]<<'"';
    std::cout<<"], \"tile_shape\": ["<<tile<<", "<<tile<<"], \"samples_per_condition\": "<<samples
        <<", \"additional_unrecorded_warmups\": "<<warmups
        <<", \"run_rows\": "<<sides.size()*combinations*samples<<"}\n";
}

// The inputs, the scalar CPU references and the checks for one image side.
class Side {
public:
    explicit Side(unsigned side) : pixels_(static_cast<std::size_t>(side)*side),
        values_(pixels_), mask_(pixels_), image_(pixels_) {
        for (std::size_t i=0;i<pixels_;++i) {
            values_[i]=static_cast<double>(i%1009)/1009-0.5; mask_[i]=i%3!=0; image_[i]=static_cast<std::uint8_t>(i*13);
        }
        reduce_reference_=hbr::masked_reduce(values_,mask_,{hbr::Backend::scalar,1});
        histogram_reference_=hbr::tile_histogram(image_,side,side,tile,tile,{hbr::Backend::scalar,1});
        stencil_reference_=hbr::stencil3x3(values_,side,side,{hbr::Backend::scalar,1});
    }
    Inputs inputs() const { return {values_,mask_,image_}; }
    std::size_t pixels() const { return pixels_; }
    // Throws unless the output matches the scalar reference, and unless every output of one
    // backend, workload and side is equal bit for bit to that backend's first one.
    void verify(int workload,int backend,const Sample& s) {
        if (workload==1) {
            if (s.counts!=histogram_reference_) throw std::runtime_error(std::string(backends[backend])+" histogram mismatch");
        } else {
            const auto v=s.values();
            // Written as !(difference <= tolerance) so that a NaN output fails.
            if (workload==0 ? !(std::abs(v[0]-reduce_reference_)<=1e-9) : v.size()!=stencil_reference_.size())
                throw std::runtime_error(std::string(backends[backend])+" reference mismatch");
            if (workload==2) for (std::size_t i=0;i<v.size();++i)
                if (!(std::abs(v[i]-stencil_reference_[i])<=1e-12))
                    throw std::runtime_error(std::string(backends[backend])+" stencil mismatch");
        }
        auto& first=first_[backend][workload];
        if (first.empty()) first.push_back(s);
        else if (!same_bits(first.front(),s)) throw std::runtime_error(std::string(backends[backend])+" output changed between samples");
    }
private:
    std::size_t pixels_;
    std::vector<double> values_;
    std::vector<std::uint8_t> mask_, image_;
    double reduce_reference_=0;
    std::vector<std::uint64_t> histogram_reference_;
    std::vector<double> stencil_reference_;
    std::array<std::array<std::vector<Sample>,3>,2> first_;
};

// Device discovery for both backends before any recorded call. cudaFree(nullptr) creates
// the CUDA primary context. The SYCL query selects the same GPU the calls select. On a host
// with two different cards the backends could pick different ones, so differing names stop
// the benchmark. Two cards of the same model would still pass this check. Returns the UUID
// of the CUDA device in NVML's form, so that NVML reads the card the CUDA calls use.
std::string discover() {
    const auto cuda=hbr::cuda_device();
    checked(cudaFree(nullptr));
    const auto sycl=hbr::sycl_device();
    std::cerr<<"CUDA: "<<cuda.name<<" SM "<<cuda.major<<'.'<<cuda.minor<<'\n'
        <<"SYCL: "<<sycl.backend<<' '<<sycl.name<<" driver "<<sycl.driver_version<<'\n';
    if (cuda.name!=sycl.name) throw std::runtime_error("CUDA and SYCL report different GPUs");
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

int main(int argc,char** argv) {
    try {
        if (argc!=2) return 2;
        const std::string mode=argv[1];
        if (mode=="--plan") { plan(); return 0; }
        if (mode=="--sequence") { sequence(); return 0; }
        if (mode!="--run" && mode!="--verify") return 2;
        // The UUID stays out of the output, since the evidence is public. A mismatch stops here.
        const GpuMonitor monitor(discover());
        std::cerr<<"NVML: the CUDA device found by its UUID\n";
        if (mode=="--verify") {
            const auto r=monitor.read();
            std::cerr<<"GPU reading: utilization "<<r.util_percent<<" percent, memory "<<r.memory_mib<<" MiB\n";
            // One call of each condition, checked as in a run, with no time printed.
            std::cout<<"side,workload,cuda_checksum,sycl_checksum,outputs_equal_bit_for_bit\n"<<std::setprecision(17);
            for (auto side : sides) {
                Side data(side);
                for (int workload=0;workload<static_cast<int>(workloads.size());++workload) {
                    const auto cuda=execute(workload,0,data.inputs(),side);
                    data.verify(workload,0,cuda);
                    const auto sycl=execute(workload,1,data.inputs(),side);
                    data.verify(workload,1,sycl);
                    std::cout<<side<<','<<workloads[workload]<<','<<checksum(cuda)<<','<<checksum(sycl)<<','
                        <<(same_bits(cuda,sycl)?"yes":"no")<<'\n';
                }
            }
            return 0;
        }
        std::cout<<"side,workload,backend,iteration,order,host_ms,upload_ms,kernel_ms,download_ms,checksum,upload_bytes,download_bytes,gpu_util_percent,gpu_memory_mib\n"
            <<std::setprecision(17);
        for (int condition=0;condition<static_cast<int>(sides.size());++condition) {
            const unsigned side=sides[condition];
            Side data(side);
            idle_reading(monitor,"before "+std::to_string(side));
            for (int iteration=0;iteration<samples;++iteration) {
                for (int order=0;order<combinations;++order) {
                    const auto [workload,backend]=slot(condition,iteration,order);
                    // Read before the clock starts, so the reading is never inside a timing.
                    const auto gpu=monitor.read();
                    const auto sample=execute(workload,backend,data.inputs(),side);
                    data.verify(workload,backend,sample);
                    row(side,workload,backend,iteration,order,sample,data.pixels(),gpu);
                }
                if (!iteration) for (int warm=0;warm<warmups;++warm) for (int index=0;index<combinations;++index)
                    data.verify(index%3,index/3,execute(index%3,index/3,data.inputs(),side));
            }
        }
        idle_reading(monitor,"after "+std::to_string(sides.back()));
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
