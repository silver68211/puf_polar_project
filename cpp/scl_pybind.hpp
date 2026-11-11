
#pragma once

// ============================================================
// Python bindings for PolarSCLPUF
// ============================================================
//
// Exposes the native AFF3CT-based SCL decoder to Python.
//
// Supported enrollment schemes:
//
//   1. Syndrome / transform enrollment
//   2. Conventional code-offset enrollment
//
// Both enrollment schemes provide channel LLRs
// corresponding to a zero-frozen polar codeword.
//
// The decoder is independent of the enrollment scheme.
//
// Features:
//
//   - Successive Cancellation List (SCL) decoding
//   - Configurable list size L
//   - SHA-256 / CRC16-aided candidate selection
//   - Optional information-bit scrambling removal
//   - Path-selection and verification statistics
//
// Candidate verification is performed in native C++
// using the public verification tag.
//
// The true enrolled secret is never used for
// selecting SCL decoding candidates.
//
// Include this header in sc_bindings.cpp and call:
//
//   bind_scl_puf(m);
//
// inside PYBIND11_MODULE(puf_aff3ct, m).
//
// ============================================================

#include <cstdint>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "polar_scl_puf.hpp"


namespace py = pybind11;


// ============================================================
// 1. Register PolarSCLPUF with Python
// ============================================================

inline void bind_scl_puf(py::module_& m)
{
    py::class_<puf::PolarSCLPUF>(
        m,
        "PolarSCLPUF"
    )

    // ========================================================
    // 2. Constructor
    // ========================================================
    //
    // Parameters:
    //
    //   n:
    //       Polar-code length N.
    //
    //   frozen_positions:
    //       Frozen-bit indices.
    //
    //   list_size:
    //       Maximum number of surviving SCL paths.
    //
    //   expected_tag:
    //       Public SHA-256 digest or CRC16 parity bits.
    //
    //   checker:
    //       "sha256" or "crc16".
    //
    //   hash_bits:
    //       Number of retained SHA-256 digest bits.
    //
    // ========================================================

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

    // ========================================================
    // 3. Read-only decoder properties
    // ========================================================

    .def_property_readonly(
        "n",
        &puf::PolarSCLPUF::n
    )

    .def_property_readonly(
        "k",
        &puf::PolarSCLPUF::k
    )

    .def_property_readonly(
        "list_size",
        &puf::PolarSCLPUF::list_size
    )

    .def_property_readonly(
        "info_positions",
        &puf::PolarSCLPUF::info_positions
    )

    // ========================================================
    // 4. SCL decoding
    // ========================================================
    //
    // Python interface:
    //
    //   decoder.decode(
    //       llrs,
    //       xor_mask=[]
    //   )
    //
    // Parameters:
    //
    //   llrs:
    //       One-dimensional array of N channel LLRs.
    //       Positive values favor bit 0.
    //
    //   xor_mask:
    //       Optional K-bit scrambling mask.
    //
    //       PolarSCLPUF removes this mask before
    //       verifying each candidate key.
    //
    // ========================================================

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
            // ------------------------------------------------
            // 5. Validate input LLR array
            // ------------------------------------------------

            const auto b = llrs.request();

            if (
                b.ndim != 1 ||
                b.shape[0] != self.n()
            )
            {
                throw py::value_error(
                    "llrs must have shape (N,)."
                );
            }

            // =================================================
            // 6. Native SCL decoding
            // =================================================
            //
            // The Python GIL can be released because
            // candidate verification is performed
            // entirely in native C++.
            //
            // The Python LLR array remains alive
            // throughout the decoding call.
            //
            // =================================================

            puf::SCLPUFResult result;

            {
                py::gil_scoped_release release;

                result = self.decode(
                    static_cast<const float*>(b.ptr),
                    xor_mask
                );
            }

            // =================================================
            // 7. Prepare Python result
            // =================================================
            //
            // The GIL has been reacquired automatically.
            //
            // =================================================

            py::dict out;

            // ------------------------------------------------
            // Reconstructed information bits
            // ------------------------------------------------

            out["key"] = py::cast(
                result.key
            );

            // ------------------------------------------------
            // Public-tag verification result
            // ------------------------------------------------

            out["verified"] =
                result.verified;

            // ------------------------------------------------
            // Configured SCL list size
            // ------------------------------------------------

            out["list_size"] =
                result.list_size;

            // ------------------------------------------------
            // Number of surviving decoding paths
            // ------------------------------------------------

            out["surviving_paths"] =
                result.surviving_paths;

            // ------------------------------------------------
            // Number of verified/tested candidates
            // ------------------------------------------------

            out["candidates_checked"] =
                result.candidates_checked;

            out["verified_candidates"] =
                result.verified_candidates;

            // ------------------------------------------------
            // Selected AFF3CT path identifier
            // ------------------------------------------------

            out["selected_path"] =
                result.selected_path;

            // ------------------------------------------------
            // Selected path metric
            // ------------------------------------------------

            out["path_metric"] =
                result.path_metric;

            return out;
        },

        py::arg("llrs"),

        py::arg("xor_mask") =
            std::vector<uint8_t>{}
    );
}
