#include "hbr/cuda.hpp"
#include "hbr/kernels.hpp"
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>

using Clock=std::chrono::steady_clock;
constexpr std::array<const char*,6> names{"cpu_optimized_1","cpu_optimized_2","cuda_simple_global",
    "cuda_simple_shared","cuda_reuse_global","cuda_reuse_shared"};
double elapsed(Clock::time_point start) { return std::chrono::duration<double,std::milli>(Clock::now()-start).count(); }
std::vector<std::uint64_t> oracle(const std::vector<std::uint8_t>& image, std::size_t side) {
    const auto nt=side/64+(side%64!=0);
    std::vector<std::uint64_t> bins(nt*nt*256);
    for (std::size_t r=0;r<side;++r) for (std::size_t c=0;c<side;++c)
        ++bins[((r/64)*nt+c/64)*256+image[r*side+c]];
    return bins;
}
struct Sample { std::vector<std::uint64_t> value; hbr::CudaTiming timing{}; std::uint64_t request_id=0; };
Sample execute(int backend,const std::vector<std::uint8_t>& image,std::size_t side,
               std::array<std::unique_ptr<hbr::CudaHistogramContext>,2>& contexts) {
    Sample sample;
    const auto start=Clock::now();
    if (backend<2) sample.value=hbr::tile_histogram(image,side,side,64,64,{hbr::Backend::optimized,static_cast<std::size_t>(backend+1)});
    else if (backend<4) {
        auto result=hbr::cuda_tile_histogram(image,side,side,64,64,backend==3);
        sample.value=std::move(result.value); sample.timing=result.timing;
    } else {
        auto result=contexts[backend-4]->run(image);
        sample.value=std::move(result.value); sample.timing=result.timing; sample.request_id=result.request_id;
    }
    sample.timing.host_total_ms=elapsed(start);
    return sample;
}
void row(const char* phase,unsigned side,const char* distribution,int backend,int iteration,int order,
         const hbr::CudaTiming& timing,std::uint64_t checksum,std::uint64_t request,std::size_t bytes) {
    std::cout<<phase<<','<<side<<','<<distribution<<','<<names[backend]<<','<<iteration<<','<<order<<','
        <<timing.host_total_ms<<','<<timing.upload_ms<<','<<timing.kernel_ms<<','<<timing.download_ms<<','
        <<checksum<<','<<request<<','<<bytes<<'\n';
}
int main(int argc,char** argv) {
    try {
        if (argc!=2 || std::string(argv[1])!="--run") return 2;
        const auto device=hbr::cuda_device();
        std::cerr<<device.name<<" SM "<<device.major<<'.'<<device.minor<<'\n';
        std::cout<<"phase,side,distribution,backend,iteration,order,host_ms,upload_ms,kernel_ms,download_ms,checksum,request_id,device_bytes\n"<<std::setprecision(17);
        int condition=0;
        for (auto side : {32u,256u,1024u}) for (auto distribution : {"uniform","single_bin"}) {
            std::vector<std::uint8_t> image(static_cast<std::size_t>(side)*side);
            for (std::size_t i=0;i<image.size();++i) image[i]=std::string(distribution)=="uniform" ? static_cast<std::uint8_t>(i*13) : 7;
            const auto expected=oracle(image,side);
            std::array<std::unique_ptr<hbr::CudaHistogramContext>,2> contexts;
            for (int method=0;method<2;++method) {
                const auto start=Clock::now();
                contexts[method]=std::make_unique<hbr::CudaHistogramContext>(side,side,64,64,method==1);
                const auto duration=elapsed(start);
                row("setup",side,distribution,method+4,-1,method,{0,0,0,duration},0,0,contexts[method]->device_bytes());
            }
            for (int iteration=0;iteration<25;++iteration) {
                for (int order=0;order<6;++order) {
                    const int backend=(order+iteration+condition)%6;
                    const auto sample=execute(backend,image,side,contexts);
                    if (sample.value!=expected) throw std::runtime_error("full histogram mismatch");
                    std::uint64_t checksum=0;
                    for (std::size_t i=0;i<sample.value.size();++i) checksum+=sample.value[i]*(i+1);
                    row("run",side,distribution,backend,iteration,order,sample.timing,checksum,sample.request_id,
                        backend>=4 ? contexts[backend-4]->device_bytes() : 0);
                }
                if (!iteration) for (int warm=0;warm<2;++warm) for (int backend=0;backend<6;++backend)
                    if (execute(backend,image,side,contexts).value!=expected) throw std::runtime_error("warmup mismatch");
            }
            for (int method=0;method<2;++method) {
                const auto bytes=contexts[method]->device_bytes();
                const auto start=Clock::now();
                contexts[method]->close(); contexts[method].reset();
                row("retire",side,distribution,method+4,-1,method,{0,0,0,elapsed(start)},0,0,bytes);
            }
            ++condition;
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
