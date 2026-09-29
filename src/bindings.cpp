#include "hbr/kernels.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace {
hbr::Options options(const std::string& backend, std::size_t threads) {
    if (backend != "scalar" && backend != "optimized") throw std::invalid_argument("backend must be scalar or optimized");
    return {backend == "scalar" ? hbr::Backend::scalar : hbr::Backend::optimized, threads};
}
template<class T> std::vector<T> snapshot(const py::array& input, int ndim) {
    if (!input.dtype().is(py::dtype::of<T>())) throw py::type_error("exact native dtype required, no implicit conversion");
    if (input.ndim() != ndim) throw py::value_error("incorrect input rank");
    if (!(input.flags() & py::array::c_style)) throw py::value_error("C contiguous input required");
    std::vector<T> data(static_cast<std::size_t>(input.size()));
    if (!data.empty()) std::memcpy(data.data(), input.data(), data.size() * sizeof(T));
    return data;
}
template<class T> py::array_t<T> array(const std::vector<T>& data, const std::vector<py::ssize_t>& shape) {
    py::array_t<T> result(shape);
    if (!data.empty()) std::memcpy(result.mutable_data(), data.data(), data.size() * sizeof(T));
    return result;
}
}
PYBIND11_MODULE(_native, m) {
    m.doc() = "Synchronous operations. Inputs are copied while holding the GIL.";
    m.def("avx2_available", &hbr::avx2_available);
    m.def("masked_reduce", [](const py::array& x, const py::array& mask, const std::string& backend, std::size_t threads) {
        auto values = snapshot<double>(x, 1);
        auto selected = snapshot<std::uint8_t>(mask, 1);
        const auto config = options(backend, threads);
        double result;
        { py::gil_scoped_release release; result = hbr::masked_reduce(values, selected, config); }
        return result;
    }, py::arg("values").noconvert(), py::arg("mask").noconvert(), py::arg("backend") = "optimized", py::arg("threads") = 2);
    m.def("tile_histogram", [](const py::array& x, std::size_t tr, std::size_t tc, const std::string& backend, std::size_t threads) {
        auto values = snapshot<std::uint8_t>(x, 2);
        const auto rows = static_cast<std::size_t>(x.shape(0)), cols = static_cast<std::size_t>(x.shape(1));
        const auto config = options(backend, threads);
        std::vector<std::uint64_t> result;
        { py::gil_scoped_release release; result = hbr::tile_histogram(values, rows, cols, tr, tc, config); }
        return array(result, {static_cast<py::ssize_t>(rows / tr + (rows % tr != 0)),
                              static_cast<py::ssize_t>(cols / tc + (cols % tc != 0)), 256});
    }, py::arg("image").noconvert(), py::arg("tile_rows"), py::arg("tile_cols"), py::arg("backend") = "optimized", py::arg("threads") = 2);
    m.def("stencil3x3", [](const py::array& x, const std::string& backend, std::size_t threads) {
        auto values = snapshot<double>(x, 2);
        const auto rows = static_cast<std::size_t>(x.shape(0)), cols = static_cast<std::size_t>(x.shape(1));
        const auto config = options(backend, threads);
        std::vector<double> result;
        { py::gil_scoped_release release; result = hbr::stencil3x3(values, rows, cols, config); }
        return array(result, {static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(cols)});
    }, py::arg("image").noconvert(), py::arg("backend") = "optimized", py::arg("threads") = 2);
}
