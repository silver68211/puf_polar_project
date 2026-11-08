
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bsc_monte_carlo_polar.hpp"

namespace py = pybind11;

inline void bind_bsc_monte_carlo_polar(py::module_& m)
{
    py::class_<puf::BSCMonteCarloPolar>(
        m, "BSCMonteCarloPolar"
    )
    .def(
        py::init<
            int,
            const std::vector<int>&,
            const std::vector<uint8_t>&,
            const std::vector<uint8_t>&,
            const std::vector<uint8_t>&,
            const std::vector<uint8_t>&,
            const std::string&,
            const std::string&,
            int,
            float,
            int,
            float,
            int,
            bool,
            uint64_t,
            int
        >(),
        py::arg("n"),
        py::arg("frozen_positions"),
        py::arg("x"),
        py::arg("z"),
        py::arg("secret"),
        py::arg("expected_tag"),
        py::arg("decoder") = "SC",
        py::arg("checker") = "sha256",
        py::arg("hash_bits") = 128,
        py::arg("prob_one") = 0.5f,
        py::arg("num_threads") = 1,
        py::arg("alpha") = 0.3367f,
        py::arg("num_trials") = 10,
        py::arg("scramble_codeword") = false,
        py::arg("seed") = 2026,
        py::arg("list_size") = 1
    )

    .def(
        "run",
        [](
            const puf::BSCMonteCarloPolar& self,
            double p,
            uint64_t min_errors,
            uint64_t max_blocks
        )
        {
            puf::PolarMCResult r;

            {
                py::gil_scoped_release release;

                r = self.run(
                    p,
                    min_errors,
                    max_blocks
                );
            }

            py::dict out;

            out["blocks"] = r.blocks;
            out["failures"] = r.failures;
            out["BLER"] = r.bler;

            out["seconds"] = r.seconds;
            out["frames_per_second"] =
                r.frames_per_second;

            out["list_size"] = r.list_size;
            out["complexity_model"] =
                r.complexity_model;

            out["nlogn_per_pass"] =
                r.nlogn_per_pass;

            out["total_sc_equivalent_units"] =
                r.total_sc_equivalent_units;

            out["total_sc_equivalent_ops"] =
                r.total_sc_equivalent_ops;

            out["normalized_complexity"] =
                r.normalized_complexity;

            if (r.complexity_model == "nominal_L")
            {
                // No actual SC pass count is measured
                // for SCL.
                out["total_sc_passes"] = py::none();
                out["mean_extra_flips"] = py::none();
                out["max_observed_passes"] = py::none();
            }
            else
            {
                out["total_sc_passes"] =
                    r.total_sc_passes;

                out["mean_extra_flips"] =
                    r.mean_extra_flips;

                out["max_observed_passes"] =
                    r.max_observed_passes;
            }

            return out;
        },
        py::arg("p"),
        py::arg("min_errors") = 100,
        py::arg("max_blocks") = 1000000000
    );
}
