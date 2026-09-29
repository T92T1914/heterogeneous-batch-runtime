#include "hbr/cuda.hpp"
#include <algorithm>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

void check(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
template<class F> void rejects(F f) { bool caught=false; try { f(); } catch (const std::exception&) { caught=true; } check(caught,"expected rejection"); }
std::vector<std::uint64_t> oracle(const std::vector<std::uint8_t>& data, std::size_t rows,
                                 std::size_t cols, std::size_t tr, std::size_t tc) {
    const auto nr=rows/tr+(rows%tr!=0), nc=cols/tc+(cols%tc!=0);
    std::vector<std::uint64_t> out(nr*nc*256);
    for (std::size_t r=0; r<rows; ++r) for (std::size_t c=0; c<cols; ++c)
        ++out[((r/tr)*nc+c/tc)*256+data[r*cols+c]];
    return out;
}
int main() {
    try {
        for (auto rows : {0u,1u,2u,17u,257u}) for (auto cols : {0u,1u,7u,259u})
            for (auto shared : {false,true}) {
                hbr::CudaHistogramContext context(rows,cols,31,33,shared);
                std::vector<std::uint8_t> data(static_cast<std::size_t>(rows)*cols,7);
                auto first=context.run(data);
                const auto original=oracle(data,rows,cols,31,33);
                check(first.value==original && first.request_id==1,"first histogram");
                for (std::size_t i=0;i<data.size();++i) data[i]=static_cast<std::uint8_t>(i*13);
                auto second=context.run(data);
                check(second.value==oracle(data,rows,cols,31,33) && second.request_id==2,"new input not uploaded");
                check(first.value==original,"previous output aliased");
                std::fill(first.value.begin(),first.value.end(),999);
                rejects([&]{ (void)context.run(std::vector<std::uint8_t>(data.size()+1)); });
                auto foreign=std::async(std::launch::async,[&]{ rejects([&]{ (void)context.run(data); }); rejects([&]{ context.close(); }); });
                foreign.get();
                auto third=context.run(data);
                check(third.request_id==3 && third.value==second.value,"rejection consumed ID or corrupted output");
                check(std::accumulate(third.value.begin(),third.value.end(),std::uint64_t{})==data.size(),"conservation");
                check(context.device_bytes()==data.size()+third.value.size()*sizeof(std::uint64_t),"storage accounting");
                context.close(); context.close();
                rejects([&]{ (void)context.run(data); });
                check(second.value==third.value,"close invalidated result");
            }
        rejects([]{ hbr::CudaHistogramContext c(1,1,0,1); });
        rejects([]{ hbr::CudaHistogramContext c(std::numeric_limits<std::size_t>::max(),2,1,1); });
        rejects([]{ hbr::CudaHistogramContext c(1024,1024,1,1); });
        // A returned host result outlives every context resource.
        auto retained=[] { hbr::CudaHistogramContext c(1,1,1,1); return c.run(std::vector<std::uint8_t>{19}); }();
        check(retained.value[19]==1,"destruction invalidated result");
        std::cout<<"CUDA reuse: changed inputs, exact bins, ownership, IDs, rejection and retirement passed\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
