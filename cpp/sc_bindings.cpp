
#include <aff3ct.hpp>

// ============================================================
// Pybind11
// ============================================================

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// ============================================================
// Standard library
// ============================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// ============================================================
// Native decoder and Monte Carlo bindings
// ============================================================

#include "bsc_monte_carlo_polar_pybind.hpp"
#include "dscf_pybind.hpp"
#include "scl_pybind.hpp"

namespace py = pybind11;


// ============================================================
// Legacy SC Monte Carlo binding
// ============================================================
//
// Implemented in bsc_monte_carlo.cpp.
//
// The main research simulations use BSCMonteCarloPolar,
// which supports SC, DSCF, and SCL.
//
// ============================================================

void bind_bsc_monte_carlo(py::module_& m);


// ============================================================
// 1. AFF3CT Polar SC Decoder
// ============================================================
//
// Python class:
//     PolarSC
//
// Decoder:
//     AFF3CT Decoder_polar_SC_naive<int, float>
//
// Input:
//     Channel LLRs with shape [B, N].
//
// Output:
//     Decoded information bits with shape [B, K].
//
// LLR convention:
//     Positive -> bit 0
//     Negative -> bit 1
//
// Supports both SRAM-PUF enrollment schemes:
//
//     1. Syndrome / transform
//     2. Conventional code offset
//
// The enrollment offset must be applied to the
// observations before calling decode().
//
// This class does not perform enrollment, hashing,
// CRC verification, or Monte Carlo simulation.
//
// ============================================================

class PolarSC
{
private:

    // Polar code length.
    int N;

    // Number of information bits.
    int K;

    // AFF3CT frozen-bit convention:
    //
    // true  = frozen position
    // false = information position
    std::vector<bool> frozen_bits;

    // Native AFF3CT SC decoder.
    std::unique_ptr<
        aff3ct::module::Decoder_polar_SC_naive<int, float>
    > decoder;


public:

    // ========================================================
    // 2. Constructor
    // ========================================================

    PolarSC(
        int N_,
        const std::vector<int>& frozen_positions
    )
        : N(N_),
          K(0)
    {
        // ----------------------------------------------------
        // Validate polar-code length before allocation
        // ----------------------------------------------------

        if (
            N < 2 ||
            (N & (N - 1)) != 0
        )
        {
            throw std::invalid_argument(
                "N must be a power of two >= 2."
            );
        }

        // ----------------------------------------------------
        // Validate number of frozen positions
        // ----------------------------------------------------

        if (
            frozen_positions.empty() ||
            frozen_positions.size() >=
                static_cast<size_t>(N)
        )
        {
            throw std::invalid_argument(
                "Require 0 < number of frozen bits < N."
            );
        }

        // ----------------------------------------------------
        // Initialize frozen-bit mask
        // ----------------------------------------------------

        frozen_bits.assign(
            static_cast<size_t>(N),
            false
        );

        // ----------------------------------------------------
        // Validate and register frozen positions
        // ----------------------------------------------------

        for (int pos : frozen_positions)
        {
            if (
                pos < 0 ||
                pos >= N
            )
            {
                throw std::invalid_argument(
                    "Frozen position outside [0, N)."
                );
            }

            if (
                frozen_bits[
                    static_cast<size_t>(pos)
                ]
            )
            {
                throw std::invalid_argument(
                    "Duplicate frozen position."
                );
            }

            frozen_bits[
                static_cast<size_t>(pos)
            ] = true;
        }

        // ----------------------------------------------------
        // Determine number of information bits
        // ----------------------------------------------------

        K = N - static_cast<int>(
            frozen_positions.size()
        );

        // ----------------------------------------------------
        // Initialize AFF3CT non-systematic SC decoder
        // ----------------------------------------------------

        decoder = std::make_unique<
            aff3ct::module::Decoder_polar_SC_naive<
                int,
                float
            >
        >(
            K,
            N,
            frozen_bits
        );
    }


    // ========================================================
    // 3. Decoder properties
    // ========================================================

    int get_N() const
    {
        return N;
    }

    int get_K() const
    {
        return K;
    }


    // ========================================================
    // 4. Batch SC decoding
    // ========================================================
    //
    // Parameters:
    //
    //   llr:
    //       Float array of shape [batch_size, N].
    //
    // Returns:
    //
    //   NumPy uint8 array of shape [batch_size, K].
    //
    // The input must contain channel LLRs after
    // applying the appropriate enrollment offset.
    //
    // The decoder itself is independent of whether
    // syndrome or code-offset enrollment was used.
    //
    // ========================================================

    py::array_t<std::uint8_t> decode(
        py::array_t<
            float,
            py::array::c_style |
            py::array::forcecast
        > llr
    )
    {
        // ----------------------------------------------------
        // Validate input dimensions
        // ----------------------------------------------------

        const auto input = llr.request();

        if (input.ndim != 2)
        {
            throw std::invalid_argument(
                "Expected a 2D array [batch_size, N]."
            );
        }

        if (input.shape[1] != N)
        {
            throw std::invalid_argument(
                "Last dimension must equal N."
            );
        }

        const auto batch_size =
            static_cast<py::ssize_t>(
                input.shape[0]
            );

        // ----------------------------------------------------
        // Allocate output
        // ----------------------------------------------------

        py::array_t<std::uint8_t> result(
            {
                batch_size,
                static_cast<py::ssize_t>(K)
            }
        );

        // ----------------------------------------------------
        // Create NumPy accessors
        // ----------------------------------------------------

        auto in = llr.unchecked<2>();
        auto out = result.mutable_unchecked<2>();

        // ----------------------------------------------------
        // Reusable decoding buffers
        // ----------------------------------------------------

        std::vector<float> y(
            static_cast<size_t>(N)
        );

        std::vector<int> decoded(
            static_cast<size_t>(K)
        );

        // ====================================================
        // 5. Execute AFF3CT SC decoding
        // ====================================================
        //
        // Release the Python GIL during the decoding loop.
        //
        // This does not make a single PolarSC instance
        // safe for simultaneous calls from multiple threads.
        //
        // Use one decoder instance per worker when
        // performing parallel decoding.
        //
        // ====================================================

        {
            py::gil_scoped_release release;

            for (
                py::ssize_t b = 0;
                b < batch_size;
                ++b
            )
            {
                // --------------------------------------------
                // Copy channel LLRs into the native buffer
                // --------------------------------------------

                for (int j = 0; j < N; ++j)
                {
                    y[static_cast<size_t>(j)] =
                        in(b, j);
                }

                // --------------------------------------------
                // Perform AFF3CT SC decoding
                // --------------------------------------------

                decoder->decode_siho(
                    y,
                    decoded
                );

                // --------------------------------------------
                // Store decoded information bits
                // --------------------------------------------

                for (int j = 0; j < K; ++j)
                {
                    out(b, j) =
                        static_cast<std::uint8_t>(
                            decoded[
                                static_cast<size_t>(j)
                            ]
                        );
                }
            }
        }

        return result;
    }
};


// ============================================================
// 6. Python module registration
// ============================================================
//
// Python module:
//     puf_aff3ct
//
// Registered classes:
//
//     PolarSC
//         Batch SC decoder.
//
//     BSCMonteCarlo
//         Legacy native SC Monte Carlo engine.
//
//     PolarDSCF
//         SC / order-one DSCF decoder.
//
//     BSCMonteCarloPolar
//         Native SC / DSCF / SCL Monte Carlo engine.
//
//     PolarSCLPUF
//         SHA-256 / CRC16-aided SCL decoder.
//
// ============================================================

PYBIND11_MODULE(puf_aff3ct, m)
{
    m.doc() =
        "AFF3CT-based polar decoding and SRAM-PUF "
        "Monte Carlo simulation";

    // ========================================================
    // 7. Register batch SC decoder
    // ========================================================

    py::class_<PolarSC>(
        m,
        "PolarSC"
    )

    .def(
        py::init<
            int,
            const std::vector<int>&
        >(),

        py::arg("N"),
        py::arg("frozen_positions")
    )

    .def(
        "decode",
        &PolarSC::decode,
        py::arg("llr")
    )

    .def_property_readonly(
        "N",
        &PolarSC::get_N
    )

    .def_property_readonly(
        "K",
        &PolarSC::get_K
    );

    // ========================================================
    // 8. Register native decoders and simulation engines
    // ========================================================

    // Legacy SC-only Monte Carlo.
    bind_bsc_monte_carlo(m);

    // Native SC / DSCF decoder.
    bind_dscf(m);

    // Native multithreaded SC / DSCF / SCL Monte Carlo.
    bind_bsc_monte_carlo_polar(m);

    // Native SHA-256 / CRC16-aided SCL decoder.
    bind_scl_puf(m);
}
