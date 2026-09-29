#include "hbr/cuda.hpp"
#include "hbr/kernels.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    try {
        if (argc != 2 || (std::string(argv[1]) != "--run" && std::string(argv[1]) != "--profile-case")) return 2;
        const bool profile = std::string(argv[1]) == "--profile-case";
        const auto device = hbr::cuda_device();
        std::cerr << device.name << " SM " << device.major << '.' << device.minor << '\n';
        std::cout << "workload,side,distribution,backend,iteration,upload_ms,kernel_ms,download_ms,host_total_ms,checksum\n" << std::setprecision(17);
        for (auto side : (profile ? std::vector<unsigned>{256} : std::vector<unsigned>{32,256,1024})) {
            std::vector<double> values(static_cast<std::size_t>(side)*side);
            std::vector<std::uint8_t> image(values.size()), mask(values.size());
            for (std::size_t i = 0; i < values.size(); ++i) { values[i] = static_cast<double>(i%1009)/1009 - 0.5; mask[i] = i%3 != 0; }
            for (auto distribution : {"uniform", "single_bin"}) {
                for (std::size_t i = 0; i < image.size(); ++i) image[i] = std::string(distribution)=="uniform" ? static_cast<std::uint8_t>(i*13) : 7;
                for (auto workload : {"masked_reduce", "tile_histogram", "stencil3x3"}) {
                    if (std::string(distribution)=="single_bin" && std::string(workload)!="tile_histogram") continue;
                    for (auto backend : {"cpu_optimized_2", "cuda_baseline", "cuda_shared"}) {
                        if (std::string(backend)=="cuda_shared" && std::string(workload)!="tile_histogram") continue;
                        for (int iteration=0; iteration<(profile?1:6); ++iteration) {
                            hbr::CudaTiming timing{}; double checksum=0;
                            const auto begin=std::chrono::steady_clock::now();
                            if (std::string(backend)=="cpu_optimized_2") {
                                if (std::string(workload)=="masked_reduce") checksum=hbr::masked_reduce(values,mask);
                                else if(std::string(workload)=="tile_histogram") { auto r=hbr::tile_histogram(image,side,side,64,64); checksum=static_cast<double>(r[7]); }
                                else { auto r=hbr::stencil3x3(values,side,side); checksum=r[r.size()/2]; }
                            } else if (std::string(workload)=="masked_reduce") {
                                auto r=hbr::cuda_masked_reduce(values,mask); checksum=r.value; timing=r.timing;
                            } else if (std::string(workload)=="tile_histogram") {
                                auto r=hbr::cuda_tile_histogram(image,side,side,64,64,std::string(backend)=="cuda_shared"); checksum=static_cast<double>(r.value[7]); timing=r.timing;
                            } else { auto r=hbr::cuda_stencil3x3(values,side,side); checksum=r.value[r.value.size()/2]; timing=r.timing; }
                            // Same completed-call boundary for every backend, including host result destruction.
                            timing.host_total_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                            std::cout<<workload<<','<<side<<','<<distribution<<','<<backend<<','<<iteration<<','<<timing.upload_ms<<','<<timing.kernel_ms<<','<<timing.download_ms<<','<<timing.host_total_ms<<','<<checksum<<'\n';
                        }
                    }
                }
            }
        }
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
