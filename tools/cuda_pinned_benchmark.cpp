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
#include <stdexcept>
#include <string>
#include <vector>

// Pinned versus pageable host input for the three existing CUDA calls.
// The runtime is unchanged: each call receives a span, and only the memory
// behind the input span differs. Outputs stay in the runtime's own vectors.
using Clock=std::chrono::steady_clock;
constexpr std::array<unsigned,3> sides{32,256,1024};
constexpr std::array<const char*,3> workloads{"masked_reduce","tile_histogram","stencil3x3"};
constexpr std::array<const char*,2> memories{"pageable","pinned"};
constexpr int samples=25, warmups=2;
constexpr std::size_t tile=64;
constexpr int combinations=static_cast<int>(workloads.size()*memories.size());

void checked(cudaError_t status) {
    if (status!=cudaSuccess) throw std::runtime_error(std::string("CUDA: ")+cudaGetErrorString(status));
}
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }

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
         const hbr::CudaTiming& t,double sum,std::size_t up,std::size_t down) {
    std::cout<<phase<<','<<side<<','<<(workload<0?"all":workloads[workload])<<','<<memories[memory]<<','
        <<iteration<<','<<order<<','<<t.host_total_ms<<','<<t.upload_ms<<','<<t.kernel_ms<<','
        <<t.download_ms<<','<<sum<<','<<up<<','<<down<<'\n';
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
        if (mode!="--run") return 2;
        const auto device=hbr::cuda_device();
        std::cerr<<device.name<<" SM "<<device.major<<'.'<<device.minor<<'\n';
        // Create the primary context before any recorded row, so no pinned allocation pays for it.
        checked(cudaFree(nullptr));
        std::cout<<"phase,side,workload,host_memory,iteration,order,host_ms,upload_ms,kernel_ms,download_ms,checksum,upload_bytes,download_bytes\n"
            <<std::setprecision(17);
        int condition=0;
        for (auto side : sides) {
            const std::size_t pixels=static_cast<std::size_t>(side)*side;
            std::vector<double> values(pixels);
            std::vector<std::uint8_t> mask(pixels), image(pixels);
            for (std::size_t i=0;i<pixels;++i) {
                values[i]=static_cast<double>(i%1009)/1009-0.5; mask[i]=i%3!=0; image[i]=static_cast<std::uint8_t>(i*13);
            }
            const std::size_t pinned_bytes=pixels*(sizeof(double)+2);
            auto start=Clock::now();
            PinnedBuffer<double> pinned_values(pixels);
            PinnedBuffer<std::uint8_t> pinned_mask(pixels), pinned_image(pixels);
            row("setup",side,-1,1,-1,-1,{0,0,0,elapsed(start)},0,pinned_bytes,0);
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
                    const int index=(order+iteration+condition)%combinations;
                    const int workload=index%3, memory=index/3;
                    const auto sample=execute(workload,inputs[memory],side);
                    verify(workload,sample);
                    row("run",side,workload,memory,iteration,order,sample.timing,checksum(sample),
                        upload_bytes(workload,pixels),download_bytes(workload,side));
                }
                if (!iteration) for (int warm=0;warm<warmups;++warm) for (int index=0;index<combinations;++index)
                    verify(index%3,execute(index%3,inputs[index/3],side));
            }
            start=Clock::now();
            pinned_values.release(); pinned_mask.release(); pinned_image.release();
            row("retire",side,-1,1,-1,-1,{0,0,0,elapsed(start)},0,pinned_bytes,0);
            ++condition;
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
