#pragma once

// Include from your existing sc_bindings.cpp, and call bind_dscf(m)
// INSIDE the existing PYBIND11_MODULE(puf_aff3ct, m) block.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "polar_dscf.hpp"

namespace py = pybind11;

inline void bind_dscf(py::module_& m) {
    py::class_<puf::PolarDSCF>(m, "PolarDSCF")
        .def(py::init<int, const std::vector<int>&, float, int>(),
             py::arg("n"), py::arg("frozen_positions"),
             py::arg("alpha") = 0.3f, py::arg("num_trials") = 10)
        .def_property_readonly("n", &puf::PolarDSCF::n)
        .def_property_readonly("k", &puf::PolarDSCF::k)
        .def_property_readonly("info_positions", &puf::PolarDSCF::info_positions)
        .def("decode", [](puf::PolarDSCF& self,
                          py::array_t<float, py::array::c_style | py::array::forcecast> llrs,
                          py::function verifier,
                          const std::vector<uint8_t>& xor_mask) {
            const auto b = llrs.request();
            if (b.ndim != 1 || b.shape[0] != self.n())
                throw py::value_error("llrs must be a 1-D array of shape (N,)");
            // Python verifier is deliberately for validation, not threaded throughput.
            // Pass the same KeyChecker.verify() predicate used by DSCFlip.py.
            auto result = self.decode(static_cast<const float*>(b.ptr),
                [&verifier](const std::vector<uint8_t>& key) {
                    py::array_t<uint8_t> candidate(key.size());
                    auto* dst = static_cast<uint8_t*>(candidate.mutable_data());
                    std::copy(key.begin(), key.end(), dst);
                    return verifier(candidate).cast<bool>();
                }, xor_mask);

            py::dict out;
            out["key"] = py::cast(result.key);
            out["leaf_llrs"] = py::cast(result.leaf_llrs);
            out["verified"] = result.verified;
            out["attempts"] = result.attempts;
            out["flip_position"] = result.flip_position;
            out["ranked_positions"] = py::cast(result.ranked_positions);
            out["log_metrics"] = py::cast(result.log_metrics);
            return out;
        }, py::arg("llrs"), py::arg("verifier"),
           py::arg("xor_mask") = std::vector<uint8_t>{});
}
