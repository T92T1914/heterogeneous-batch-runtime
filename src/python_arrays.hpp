#pragma once
#include <pybind11/numpy.h>
#include <cstring>
#include <limits>
#include <vector>

namespace hbr::python {
template<class T>
pybind11::array_t<T> array(const std::vector<T>& data,
                          const std::vector<pybind11::ssize_t>& shape) {
    std::size_t count = 1;
    const auto limit = static_cast<std::size_t>((std::numeric_limits<pybind11::ssize_t>::max)());
    for (const auto dimension : shape) {
        if (dimension < 0) throw pybind11::value_error("negative result dimension");
        const auto size = static_cast<std::size_t>(dimension);
        if (size && count > limit / size) throw pybind11::value_error("result shape product overflow");
        count *= size;
    }
    if (count > limit / sizeof(T))
        throw pybind11::value_error("result byte count overflow");
    // NumPy strides are signed byte counts. Empty arrays still require every
    // intermediate C stride to be representable before pybind11 computes it.
    std::size_t stride = sizeof(T);
    for (std::size_t i = shape.size(); i > 1; --i) {
        const auto size = static_cast<std::size_t>(shape[i - 1]);
        if (size && stride > limit / size) throw pybind11::value_error("result stride overflow");
        stride *= size;
    }
    if (count != data.size()) throw pybind11::value_error("result shape does not match native vector length");
    pybind11::array_t<T> result(shape);
    if (count) std::memcpy(result.mutable_data(), data.data(), count * sizeof(T));
    return result;
}
}
