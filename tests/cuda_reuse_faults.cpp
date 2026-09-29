#include "hbr/cuda.hpp"
#include "../src/cuda_faults.hpp"
#include <iostream>
#include <stdexcept>

void check(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
int main() {
    try {
        hbr::test::fail_at(-1);
        int init_points;
        { hbr::CudaHistogramContext c(17,19,8,7); init_points=hbr::test::checkpoints(); }
        check(hbr::test::live_resources()==0,"initial resource accounting");
        for (int point=0;point<init_points;++point) {
            hbr::test::fail_at(point);
            bool caught=false;
            try { hbr::CudaHistogramContext c(17,19,8,7); }
            catch (const std::runtime_error&) { caught=true; }
            check(caught && hbr::test::live_resources()==0,"partial initialization leaked resources");
        }
        const std::vector<std::uint8_t> data(17*19,7);
        int run_points;
        hbr::test::fail_at(-1);
        { hbr::CudaHistogramContext c(17,19,8,7); hbr::test::fail_at(-1); (void)c.run(data); run_points=hbr::test::checkpoints(); }
        for (int point=0;point<run_points;++point) {
            hbr::test::fail_at(-1);
            hbr::CudaHistogramContext c(17,19,8,7);
            const auto before=hbr::test::drains();
            hbr::test::fail_at(point);
            bool caught=false;
            try { (void)c.run(data); } catch (const std::runtime_error&) { caught=true; }
            check(caught,"execution injection missing");
            check(hbr::test::drains()>before && hbr::test::live_resources()==0,"failed execution did not drain and release");
            hbr::test::fail_at(-1);
            bool retired=false;
            try { (void)c.run(data); } catch (const std::logic_error&) { retired=true; }
            check(retired,"failed context reused");
            c.close(); c.close();
        }
        check(hbr::test::live_resources()==0,"resources remain");
        check(hbr::test::drain_errors()==0,"cleanup synchronization failed");
        std::cout<<init_points<<" acquisition and "<<run_points<<" execution host-exception checkpoints passed\n";
        std::cout<<"Actual device loss, CUDA OOM and GPU sanitizer coverage are not established\n";
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
