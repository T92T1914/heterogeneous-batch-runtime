#include "hbr/cuda.hpp"
#include "hbr/kernels.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <limits>
int main() {
    try {
        const auto device = hbr::cuda_device();
        std::cout << device.name << " capability " << device.major << '.' << device.minor << '\n';
        for (auto rows : {0u, 1u, 2u, 3u, 17u, 257u}) for (auto cols : {0u, 1u, 7u, 259u}) {
            std::vector<double> x(rows * cols); std::vector<std::uint8_t> bytes(rows * cols), mask(rows * cols);
            for (std::size_t i = 0; i < x.size(); ++i) { x[i] = (static_cast<double>(i % 103) - 51.5) / 103; bytes[i] = static_cast<std::uint8_t>(i); mask[i] = i % 3 ? 255 : 0; }
            auto reduced = hbr::cuda_masked_reduce(x,mask);
            if (!std::isfinite(reduced.value) || std::abs(reduced.value - hbr::masked_reduce(x,mask,{hbr::Backend::scalar,1})) > 1e-9) throw std::runtime_error("CUDA reduction mismatch");
            auto filtered = hbr::cuda_stencil3x3(x,rows,cols).value;
            auto reference = hbr::stencil3x3(x,rows,cols,{hbr::Backend::scalar,1});
            for (std::size_t i = 0; i < x.size(); ++i) if (!std::isfinite(filtered[i]) || std::abs(filtered[i] - reference[i]) > 1e-12) throw std::runtime_error("CUDA stencil mismatch");
            for (auto shared : {false,true}) {
                const auto actual = hbr::cuda_tile_histogram(bytes,rows,cols,31,33,shared).value;
                if (actual != hbr::tile_histogram(bytes,rows,cols,31,33,{hbr::Backend::scalar,1})) throw std::runtime_error("CUDA histogram mismatch");
            }
        }
        const auto largest = std::numeric_limits<double>::max();
        for (const auto value : {largest, -largest}) {
            const std::vector<double> input(25, value);
            const auto output = hbr::cuda_stencil3x3(input,5,5).value;
            const auto reference = hbr::stencil3x3(input,5,5,{hbr::Backend::scalar,1});
            for (std::size_t i=0; i<output.size(); ++i)
                if (!std::isfinite(output[i]) || output[i]!=reference[i]) throw std::runtime_error("CUDA extreme stencil mismatch");
        }
        bool overflow_rejected=false;
        try { (void)hbr::cuda_masked_reduce(std::vector<double>{largest,1.0},std::vector<std::uint8_t>{1,1}); }
        catch(const std::overflow_error&) { overflow_rejected=true; }
        if (!overflow_rejected) throw std::runtime_error("CUDA reduction overflow guard missing");
        for (auto value : {std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
            bool rejected=false;
            try { (void)hbr::cuda_masked_reduce(std::vector<double>{value},std::vector<std::uint8_t>{0}); }
            catch(const std::invalid_argument&) { rejected=true; }
            if (!rejected) throw std::runtime_error("CUDA nonfinite input accepted");
        }
        for (auto side : {32u,256u,1024u}) for (auto skewed : {false,true}) {
            std::vector<std::uint8_t> input(static_cast<std::size_t>(side)*side);
            for (std::size_t i=0; i<input.size(); ++i) input[i]=skewed ? 7 : static_cast<std::uint8_t>(i*13);
            const auto expected=hbr::tile_histogram(input,side,side,64,64,{hbr::Backend::scalar,1});
            for (auto shared : {false,true})
                if (hbr::cuda_tile_histogram(input,side,side,64,64,shared).value!=expected) throw std::runtime_error("CUDA benchmark histogram mismatch");
        }
        std::cout << "CUDA completed-operation numerical checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
