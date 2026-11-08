
#include <aff3ct.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>     // Python list <-> std::vector conversion

#include "dscf_pybind.hpp"
#include "bsc_monte_carlo_polar_pybind.hpp"

#include "scl_pybind.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
void bind_bsc_monte_carlo(py::module_& m);

class PolarSC
{
private:
    int N;
    int K;

    // AFF3CT convention:
    // true  = frozen bit
    // false = information bit
    std::vector<bool> frozen_bits;

    std::unique_ptr<
        aff3ct::module::Decoder_polar_SC_naive<int, float>
    > decoder;

public:
    PolarSC(int N_, std::vector<int> frozen_positions)
        : N(N_), K(0), frozen_bits(N_, false)
    {
        if (N < 2 || (N & (N - 1)) != 0)
            throw std::invalid_argument(
                "N must be a power of two >= 2."
            );

        if (frozen_positions.size() >=
            static_cast<size_t>(N))
            throw std::invalid_argument(
                "Invalid number of frozen positions."
            );

        // Check frozen positions.
        for (int pos : frozen_positions)
        {
            if (pos < 0 || pos >= N)
                throw std::invalid_argument(
                    "Frozen position outside [0, N)."
                );

            if (frozen_bits[pos])
                throw std::invalid_argument(
                    "Duplicate frozen position."
                );

            frozen_bits[pos] = true;
        }

        K = N - static_cast<int>(frozen_positions.size());

        // Create the AFF3CT non-systematic SC decoder.
        decoder = std::make_unique<
            aff3ct::module::Decoder_polar_SC_naive<int, float>
        >(K, N, frozen_bits);
    }

    int get_N() const { return N; }
    int get_K() const { return K; }

    py::array_t<std::uint8_t> decode(
        py::array_t<float,
            py::array::c_style | py::array::forcecast> llr)
    {
        auto input = llr.request();

        if (input.ndim != 2)
            throw std::invalid_argument(
                "Expected a 2D array [batch_size, N]."
            );

        if (input.shape[1] != N)
            throw std::invalid_argument(
                "Last dimension must equal N."
            );

        const auto batch_size =
            static_cast<py::ssize_t>(input.shape[0]);

        py::array_t<std::uint8_t> result(
            {batch_size, static_cast<py::ssize_t>(K)}
        );

        auto in = llr.unchecked<2>();
        auto out = result.mutable_unchecked<2>();

        // Reuse buffers instead of allocating each frame.
        std::vector<float> y(N);
        std::vector<int> decoded(K);

        // Release the Python GIL while executing C++.
        // This does not make this decoder instance thread-safe.
        py::gil_scoped_release release;

        for (py::ssize_t b = 0; b < batch_size; ++b)
        {
            for (int j = 0; j < N; ++j)
                y[j] = in(b, j);

            // AFF3CT SC decoding.
            // y contains LLRs, not hard bits.
            decoder->decode_siho(y, decoded);

            for (int j = 0; j < K; ++j)
                out(b, j) =
                    static_cast<std::uint8_t>(decoded[j]);
        }

        return result;
    }
};


PYBIND11_MODULE(puf_aff3ct, m)
{
    m.doc() =
        "AFF3CT-based SC polar decoder for SRAM-PUFs";

    py::class_<PolarSC>(m, "PolarSC")
        .def(
            py::init<int, std::vector<int>>(),
            py::arg("N"),
            py::arg("frozen_positions")
        )
        .def("decode", &PolarSC::decode)
        .def_property_readonly("N", &PolarSC::get_N)
        .def_property_readonly("K", &PolarSC::get_K);
    bind_bsc_monte_carlo(m);
    bind_dscf(m);
    bind_bsc_monte_carlo_polar(m);
    bind_scl_puf(m);
}

