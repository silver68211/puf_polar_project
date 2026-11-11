
#pragma once

// ============================================================
// Python bindings for the PolarDSCF decoder
// ============================================================
//
// Exposes the native AFF3CT-based SC/DSCF decoder to Python.
//
// Supported enrollment schemes:
//   1. Syndrome / transform construction
//   2. Conventional code-offset construction
//
// Both enrollment schemes supply LLRs corresponding to
// a zero-frozen polar codeword.
//
// The decoder does not need to know which enrollment
// scheme generated the observations.
//
// Features:
//   - Standard SC decoding (num_trials = 0)
//   - Order-one DSCF decoding
//   - Python-based candidate verification
//   - Optional information-bit scrambling removal
//   - Decoding statistics and flip-position information
//
// IMPORTANT:
//
// The Python verifier is intended for correctness testing
// and validation, not high-throughput multithreaded Monte Carlo.
//
// For large simulations, BSCMonteCarloPolar uses the native
// C++ verification implementation instead.
//
// Include this header in sc_bindings.cpp and call:
//
//   bind_dscf(m);
//
// inside the existing PYBIND11_MODULE(puf_aff3ct, m) block.
//
// ============================================================

#include <algorithm>
#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "polar_dscf.hpp"


namespace py = pybind11;


// ============================================================
// 1. Register PolarDSCF with Python
// ============================================================

inline void bind_dscf(py::module_& m)
{
    py::class_<puf::PolarDSCF>(
        m,
        "PolarDSCF"
    )

    // ========================================================
    // 2. Constructor
    // ========================================================
    //
    // Parameters:
    //
    //   n:
    //       Polar-code block length N.
    //
    //   frozen_positions:
    //       Frozen-bit indices.
    //
    //   alpha:
    //       DSCF candidate-ranking parameter.
    //
    //   num_trials:
    //       Maximum number of additional flip attempts.
    //
    //       num_trials = 0 gives standard SC decoding.
    //
    // ========================================================

    .def(
        py::init<
            int,
            const std::vector<int>&,
            float,
            int
        >(),

        py::arg("n"),

        py::arg("frozen_positions"),

        py::arg("alpha") = 0.3f,

        py::arg("num_trials") = 10
    )

    // ========================================================
    // 3. Read-only decoder properties
    // ========================================================

    .def_property_readonly(
        "n",
        &puf::PolarDSCF::n
    )

    .def_property_readonly(
        "k",
        &puf::PolarDSCF::k
    )

    .def_property_readonly(
        "info_positions",
        &puf::PolarDSCF::info_positions
    )

    // ========================================================
    // 4. SC / DSCF decoding
    // ========================================================
    //
    // Python interface:
    //
    //   decoder.decode(
    //       llrs,
    //       verifier,
    //       xor_mask=[]
    //   )
    //
    // Parameters:
    //
    //   llrs:
    //       One-dimensional array of N channel LLRs.
    //       Positive LLR favors bit 0.
    //
    //   verifier:
    //       Python callable that accepts a candidate key
    //       and returns True or False.
    //
    //   xor_mask:
    //       Optional K-bit scrambling mask.
    //       Removed before candidate verification.
    //
    // The callback must verify a PUBLIC tag, not compare
    // the candidate with the true enrolled secret.
    //
    // ========================================================

    .def(
        "decode",

        [](
            puf::PolarDSCF& self,

            py::array_t<
                float,
                py::array::c_style |
                py::array::forcecast
            > llrs,

            py::function verifier,

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
                    "llrs must be a 1-D array "
                    "of shape (N,)."
                );
            }

            // ------------------------------------------------
            // 6. Native decoding with Python verification
            // ------------------------------------------------
            //
            // The Python GIL remains held during decoding
            // because the native decoder can invoke the
            // Python verifier multiple times.
            //
            // PolarDSCF removes xor_mask before passing
            // candidate keys to this callback.
            //
            // ------------------------------------------------

            auto result = self.decode(
                static_cast<const float*>(b.ptr),

                [&verifier](
                    const std::vector<uint8_t>& key
                ) -> bool
                {
                    // ----------------------------------------
                    // Convert candidate key to NumPy uint8
                    // ----------------------------------------

                    py::array_t<uint8_t> candidate(
                        key.size()
                    );

                    auto* dst =
                        static_cast<uint8_t*>(
                            candidate.mutable_data()
                        );

                    std::copy(
                        key.begin(),
                        key.end(),
                        dst
                    );

                    // ----------------------------------------
                    // Verify using the supplied Python
                    // SHA-256 or CRC16 checker.
                    // ----------------------------------------

                    return verifier(
                        candidate
                    ).cast<bool>();
                },

                xor_mask
            );

            // =================================================
            // 7. Prepare Python result
            // =================================================

            py::dict out;

            // -----------------------------------------------
            // Decoded key
            // -----------------------------------------------

            out["key"] =
                py::cast(result.key);

            // -----------------------------------------------
            // Leaf LLR values from retained path
            // -----------------------------------------------

            out["leaf_llrs"] =
                py::cast(result.leaf_llrs);

            // -----------------------------------------------
            // Whether the verifier accepted the result
            // -----------------------------------------------

            out["verified"] =
                result.verified;

            // -----------------------------------------------
            // Total SC decoding passes
            // -----------------------------------------------

            out["attempts"] =
                result.attempts;

            // -----------------------------------------------
            // Accepted flip position
            //
            // -1 means initial SC estimate retained
            // or no flip candidate passed verification.
            // -----------------------------------------------

            out["flip_position"] =
                result.flip_position;

            // -----------------------------------------------
            // Information-bit positions ranked by
            // decreasing DSCF metric
            // -----------------------------------------------

            out["ranked_positions"] =
                py::cast(
                    result.ranked_positions
                );

            // -----------------------------------------------
            // Log-domain DSCF ranking metrics
            // -----------------------------------------------

            out["log_metrics"] =
                py::cast(
                    result.log_metrics
                );

            return out;
        },

        py::arg("llrs"),

        py::arg("verifier"),

        py::arg("xor_mask") =
            std::vector<uint8_t>{}
    );
}
