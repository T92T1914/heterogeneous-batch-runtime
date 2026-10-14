#include "../src/python_arrays.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdint>
PYBIND11_MODULE(_array_boundary_test, m) {
    m.def("copy", [](const std::vector<std::uint64_t>& data, const std::vector<pybind11::ssize_t>& shape) {
        return hbr::python::array(data, shape);
    });
}
