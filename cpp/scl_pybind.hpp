
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "polar_scl_puf.hpp"

namespace py = pybind11;

inline void bind_scl_puf(py::module_& m)
{
    py::class_<puf::PolarSCLPUF>(
        m, "PolarSCLPUF"
    )
    .def(
        py::init<
            int,
            const std::vector<int>&,
            int,
            const std::vector<uint8_t>&,
            const std::string&,
            int
        >(),
        py::arg("n"),
        py::arg("frozen_positions"),
        py::arg("list_size"),
        py::arg("expected_tag"),
        py::arg("checker") = "sha256",
        py::arg("hash_bits") = 128
    )

    .def_property_readonly(
        "n", &puf::PolarSCLPUF::n
    )

    .def_property_readonly(
        "k", &puf::PolarSCLPUF::k
    )

    .def_property_readonly(
        "list_size", &puf::PolarSCLPUF::list_size
    )

    .def_property_readonly(
        "info_positions",
        &puf::PolarSCLPUF::info_positions
    )

    .def(
        "decode",
        [](
            puf::PolarSCLPUF& self,
            py::array_t<
                float,
                py::array::c_style |
                py::array::forcecast
            > llrs,
            const std::vector<uint8_t>& xor_mask
        )
        {
            const auto b = llrs.request();

            if (b.ndim != 1 ||
                b.shape[0] != self.n())
                throw py::value_error(
                    "llrs must have shape (N,)."
                );

            puf::SCLPUFResult result;

            {
                py::gil_scoped_release release;

                result = self.decode(
                    static_cast<const float*>(b.ptr),
                    xor_mask
                );
            }

            py::dict out;

            out["key"] = py::cast(result.key);
            out["verified"] = result.verified;

            out["list_size"] = result.list_size;

            out["surviving_paths"] =
                result.surviving_paths;

            out["candidates_checked"] =
                result.candidates_checked;

            out["verified_candidates"] =
                result.verified_candidates;

            out["selected_path"] =
                result.selected_path;

            out["path_metric"] =
                result.path_metric;

            return out;
        },
        py::arg("llrs"),
        py::arg("xor_mask") =
            std::vector<uint8_t>{}
    );
}
