#include <hbr/kernels.hpp>
#ifdef CHECK_CUDA
#include <hbr/cuda.hpp>
#endif
#include <vector>
int main() {
    const std::vector<double> values{2,-3,5};
    const std::vector<std::uint8_t> mask{1,0,1};
    if (hbr::masked_reduce(values,mask)!=7) return 1;
#ifdef CHECK_CUDA
    if (hbr::cuda_masked_reduce(values,mask).value!=7) return 2;
    hbr::CudaHistogramContext histogram(1,3,1,3);
    const auto result=histogram.run(mask);
    if (result.request_id!=1 || result.value[0]!=1 || result.value[1]!=2) return 3;
    histogram.close();
#endif
}
