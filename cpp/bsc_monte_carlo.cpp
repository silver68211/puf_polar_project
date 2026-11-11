
#include <aff3ct.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;


// ============================================================
// Native BSC Monte Carlo simulator for polar SRAM-PUFs
// ============================================================
//
// Decoder:
//   AFF3CT Successive Cancellation (SC)
//
// Supported enrollment schemes:
//
//   1. Syndrome / transform construction
//      S = (X G_N)_A
//      Z = embed((X G_N)_F) G_N
//
//   2. Conventional code-offset construction
//      S = independently generated secret
//      C = encode(S), with zero frozen bits
//      Z = X XOR C
//
// Both schemes satisfy:
//
//   (X XOR Z) G_N = U
//
//   U_F = 0
//   U_A = S
//
// Channel:
//   Binary Symmetric Channel (BSC)
//
// Optional features:
//   - Biased SRAM-PUF source prior
//   - Random valid-codeword scrambling
//   - Multithreaded Monte Carlo
//   - Live progress reporting
//
// Output:
//   blocks
//   failures
//   BLER
//   seconds
//   frames_per_second
//
// IMPORTANT:
//
// This is the legacy SC-only simulator.
//
// The main SC/DSCF/SCL simulator uses
// BSCMonteCarloPolar, defined in
// bsc_monte_carlo_polar.hpp.
//
// ============================================================


// ============================================================
// 1. Monte Carlo progress reporting
// ============================================================

static void print_progress(
    double p,
    const std::atomic<uint64_t>& blocks,
    const std::atomic<uint64_t>& failures
)
{
    const uint64_t n_blocks =
        blocks.load(std::memory_order_relaxed);

    const uint64_t n_failures =
        failures.load(std::memory_order_relaxed);

    const double bler = n_blocks > 0
        ? static_cast<double>(n_failures) / n_blocks
        : 0.0;

    std::cout
        << "\r\033[2K"
        << std::fixed
        << std::setprecision(3)
        << "p: " << std::setw(6) << p
        << " | Blocks: " << std::setw(12) << n_blocks
        << " | Failures: " << std::setw(6) << n_failures
        << " | BLER: "
        << std::scientific
        << std::setprecision(5)
        << bler
        << std::flush;
}


// ============================================================
// 2. Convert Python arrays to C++ binary vectors
// ============================================================
//
// Input:
//   One-dimensional NumPy array of length expected_length.
//
// Output:
//   std::vector<uint8_t> containing binary values.
//
// This function expects the existing NumPy uint8 input
// convention used by the Python binding.
//
// ============================================================

static std::vector<uint8_t> read_bits(
    py::array_t<
        uint8_t,
        py::array::c_style | py::array::forcecast
    > array,
    int expected_length
)
{
    const auto a = array.request();

    // --------------------------------------------------------
    // Validate array shape
    // --------------------------------------------------------

    if (
        a.ndim != 1 ||
        a.shape[0] != expected_length
    )
    {
        throw std::invalid_argument(
            "Invalid binary array length."
        );
    }

    const auto* ptr =
        static_cast<const uint8_t*>(a.ptr);

    std::vector<uint8_t> bits(
        ptr,
        ptr + expected_length
    );

    // --------------------------------------------------------
    // Validate binary values
    // --------------------------------------------------------

    for (uint8_t bit : bits)
    {
        if (bit > 1)
        {
            throw std::invalid_argument(
                "Bits must be 0 or 1."
            );
        }
    }

    return bits;
}


// ============================================================
// 3. Non-systematic polar transform
// ============================================================
//
// Computes:
//
//   x = u G_N
//
// where:
//
//   G_N = F^(tensor log2(N))
//
//   F = [[1,0],
//        [1,1]]
//
// All operations are performed over GF(2).
//
// The same transform is used for polar encoding
// and for validating the enrolled coset.
//
// ============================================================

static void polar_transform(
    std::vector<uint8_t>& x
)
{
    const int n = static_cast<int>(x.size());

    for (int step = 1; step < n; step *= 2)
    {
        for (
            int start = 0;
            start < n;
            start += 2 * step
        )
        {
            for (int j = 0; j < step; ++j)
            {
                x[start + j] ^=
                    x[start + j + step];
            }
        }
    }
}


// ============================================================
// 4. SC Monte Carlo simulator
// ============================================================

class BSCMonteCarlo
{
private:

    // --------------------------------------------------------
    // Polar-code parameters
    // --------------------------------------------------------

    int N;
    int K;

    // Number of parallel C++ workers.
    int num_threads;

    // Probability that a physical SRAM bit equals 1.
    double q;

    // Enable random valid-codeword scrambling.
    bool scramble_codeword;

    // Frozen-bit mask:
    //   true  = frozen position
    //   false = information position
    std::vector<bool> frozen_mask;

    // Information-bit positions in ascending order.
    std::vector<int> info_positions;

    // --------------------------------------------------------
    // Fixed SRAM-PUF enrollment
    // --------------------------------------------------------

    // Physical SRAM-PUF response.
    std::vector<uint8_t> X;

    // Full N-bit reconstruction offset.
    //
    // Syndrome:
    //   Z = embed((X G_N)_F) G_N
    //
    // Code offset:
    //   Z = X XOR encode(S)
    std::vector<uint8_t> Z;

    // Enrolled secret key.
    std::vector<uint8_t> S;


public:

    // ========================================================
    // 5. Constructor
    // ========================================================

    BSCMonteCarlo(
        int n,
        std::vector<int> frozen_positions,
        py::array_t<uint8_t> x,
        py::array_t<uint8_t> z,
        py::array_t<uint8_t> key,
        double prob_one,
        int threads,
        bool use_scrambling = false
    )
        : N(n),
          K(0),
          num_threads(threads),
          q(prob_one),
          scramble_codeword(use_scrambling)
    {
        // ----------------------------------------------------
        // Validate polar-code length
        // ----------------------------------------------------

        if (
            N < 2 ||
            (N & (N - 1)) != 0
        )
        {
            throw std::invalid_argument(
                "N must be a power of two."
            );
        }

        // ----------------------------------------------------
        // Validate simulation parameters
        // ----------------------------------------------------

        if (num_threads < 1)
        {
            throw std::invalid_argument(
                "num_threads must be >= 1."
            );
        }

        if (
            !std::isfinite(q) ||
            !(q > 0.0 && q < 1.0)
        )
        {
            throw std::invalid_argument(
                "q must be in (0,1)."
            );
        }

        // ----------------------------------------------------
        // Construct frozen-bit mask
        // ----------------------------------------------------

        frozen_mask.assign(N, false);

        for (int pos : frozen_positions)
        {
            if (
                pos < 0 ||
                pos >= N ||
                frozen_mask[pos]
            )
            {
                throw std::invalid_argument(
                    "Invalid or repeated frozen position."
                );
            }

            frozen_mask[pos] = true;
        }

        // ----------------------------------------------------
        // Identify information-bit positions
        // ----------------------------------------------------

        for (int i = 0; i < N; ++i)
        {
            if (!frozen_mask[i])
            {
                info_positions.push_back(i);
            }
        }

        K = static_cast<int>(
            info_positions.size()
        );

        if (K <= 0 || K >= N)
        {
            throw std::invalid_argument(
                "Require 0 < K < N."
            );
        }

        // ----------------------------------------------------
        // Copy enrollment data from Python
        // ----------------------------------------------------

        X = read_bits(x, N);
        Z = read_bits(z, N);
        S = read_bits(key, K);

        // ====================================================
        // 6. Validate enrollment
        // ====================================================
        //
        // Both schemes must satisfy:
        //
        //   C = X XOR Z
        //   U = C G_N
        //
        //   U_F = 0
        //   U_A = S
        //
        // This condition is independent of how S and Z
        // were constructed during enrollment.
        //
        // ====================================================

        std::vector<uint8_t> u_check(N);

        // Apply the full reconstruction offset.

        for (int i = 0; i < N; ++i)
        {
            u_check[i] = X[i] ^ Z[i];
        }

        // Recover polar-domain bits.

        polar_transform(u_check);

        // ----------------------------------------------------
        // Verify zero frozen bits
        // ----------------------------------------------------

        for (int i = 0; i < N; ++i)
        {
            if (
                frozen_mask[i] &&
                u_check[i] != 0
            )
            {
                throw std::invalid_argument(
                    "Invalid coset shift: "
                    "frozen bit is nonzero."
                );
            }
        }

        // ----------------------------------------------------
        // Verify enrolled information bits
        // ----------------------------------------------------

        for (int j = 0; j < K; ++j)
        {
            if (
                u_check[info_positions[j]] != S[j]
            )
            {
                throw std::invalid_argument(
                    "Enrollment key does not match X and Z."
                );
            }
        }
    }


    // ========================================================
    // 7. Run Monte Carlo simulation
    // ========================================================

    py::dict run(
        double p,
        uint64_t min_errors,
        uint64_t max_blocks
    )
    {
        // ----------------------------------------------------
        // Validate BSC and stopping parameters
        // ----------------------------------------------------

        if (
            !std::isfinite(p) ||
            p < 0.0 ||
            p >= 0.5
        )
        {
            throw std::invalid_argument(
                "Require finite 0 <= p < 0.5."
            );
        }

        if (
            min_errors == 0 ||
            max_blocks == 0
        )
        {
            throw std::invalid_argument(
                "min_errors and max_blocks must be positive."
            );
        }

        // ====================================================
        // 8. BSC channel parameters
        // ====================================================

        // Channel LLR magnitude:
        //
        //   lambda = log((1-p)/p)
        //
        // p=0 is represented by a large finite magnitude.

        const float lambda = (p == 0.0)
            ? 50.0f
            : static_cast<float>(
                std::log((1.0 - p) / p)
            );

        // SRAM-PUF source prior:
        //
        //   prior = log((1-q)/q)
        //
        // For q=0.5, prior=0.

        const float prior =
            static_cast<float>(
                std::log((1.0 - q) / q)
            );

        // ====================================================
        // 9. Monte Carlo counters
        // ====================================================

        // Number of allocated frame IDs.
        std::atomic<uint64_t> next_frame{0};

        // Number of completed frames.
        std::atomic<uint64_t> blocks{0};

        // Number of failed key reconstructions.
        std::atomic<uint64_t> failures{0};

        // Stop all workers if an exception occurs.
        std::atomic<bool> abort{false};

        // Protect progress printing.
        std::mutex progress_mutex;

        // Capture worker exceptions safely.
        std::mutex exception_mutex;
        std::exception_ptr first_exception;

        // Number of frames between progress updates.
        constexpr uint64_t PRINT_INTERVAL = 100000;

        // ====================================================
        // 10. Initialize AFF3CT decoders
        // ====================================================
        //
        // One independent SC decoder is created per worker.
        //
        // AFF3CT decoder state is mutable and must not
        // be shared between worker threads.
        //
        // ====================================================

        using SCDecoder =
            aff3ct::module::Decoder_polar_SC_naive<
                int,
                float
            >;

        std::vector<std::unique_ptr<SCDecoder>> decoders;

        decoders.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t)
        {
            decoders.emplace_back(
                std::make_unique<SCDecoder>(
                    K,
                    N,
                    frozen_mask
                )
            );
        }

        const auto start =
            std::chrono::steady_clock::now();

        // ====================================================
        // 11. Execute multithreaded Monte Carlo
        // ====================================================

        {
            // Release Python GIL while C++ workers execute.

            py::gil_scoped_release release;

            std::vector<std::thread> workers;

            workers.reserve(num_threads);

            // ------------------------------------------------
            // Start worker threads
            // ------------------------------------------------

            for (int t = 0; t < num_threads; ++t)
            {
                workers.emplace_back([&, t]()
                {
                    try
                    {
                        // ====================================
                        // 12. Worker random number generator
                        // ====================================

                        // Independent pseudorandom stream.
                        //
                        // This legacy implementation uses
                        // std::random_device for seeding.
                        //
                        // Unlike BSCMonteCarloPolar, it does
                        // not expose a fixed simulation seed.

                        std::random_device rd;

                        std::seed_seq seed{
                            rd(),
                            rd(),
                            rd(),
                            rd(),
                            static_cast<unsigned int>(t)
                        };

                        std::mt19937_64 rng(seed);

                        // ------------------------------------
                        // BSC noise generation
                        // ------------------------------------
                        //
                        // Compare a uniformly distributed
                        // 64-bit integer to a threshold.
                        //
                        // P(error) is approximately p, with
                        // resolution 2^-64.
                        //
                        // The threshold is strictly below
                        // 2^63 since p < 0.5.

                        const uint64_t noise_threshold =
                            static_cast<uint64_t>(
                                std::ldexp(p, 64)
                            );

                        // Uniform scrambling information bits.
                        std::bernoulli_distribution mask_bit(
                            0.5
                        );

                        // ====================================
                        // 13. Worker-local buffers
                        // ====================================

                        // Channel LLRs.
                        std::vector<float> llr(N);

                        // Decoded information bits.
                        std::vector<int> decoded(K);

                        // Random polar information vector.
                        std::vector<uint8_t> random_u(N, 0);

                        // Encoded scrambling codeword.
                        std::vector<uint8_t> random_x(N, 0);

                        auto& sc = *decoders[t];

                        // ====================================
                        // 14. Main frame-processing loop
                        // ====================================

                        while (
                            !abort.load(
                                std::memory_order_relaxed
                            ) &&
                            failures.load(
                                std::memory_order_relaxed
                            ) < min_errors
                        )
                        {
                            // --------------------------------
                            // Reserve one unique frame
                            // --------------------------------

                            const uint64_t id =
                                next_frame.fetch_add(
                                    1,
                                    std::memory_order_relaxed
                                );

                            if (id >= max_blocks)
                            {
                                break;
                            }

                            // ================================
                            // 15. Optional scrambling
                            // ================================

                            if (scramble_codeword)
                            {
                                // Frozen positions are zero.

                                std::fill(
                                    random_u.begin(),
                                    random_u.end(),
                                    0
                                );

                                // Generate random information bits.

                                for (int j = 0; j < K; ++j)
                                {
                                    random_u[info_positions[j]] =
                                        mask_bit(rng) ? 1 : 0;
                                }

                                // Encode the random mask.
                                //
                                // The resulting vector is a
                                // valid zero-frozen codeword.

                                random_x = random_u;

                                polar_transform(random_x);
                            }

                            // ================================
                            // 16. Generate BSC observations
                            // ================================

                            for (int i = 0; i < N; ++i)
                            {
                                // ----------------------------
                                // Generate BSC noise bit
                                // ----------------------------

                                const uint8_t e =
                                    static_cast<uint8_t>(
                                        rng() < noise_threshold
                                    );

                                // ----------------------------
                                // Noisy physical SRAM response
                                //
                                // Y = X XOR E
                                // ----------------------------

                                const uint8_t y =
                                    X[i] ^ e;

                                // ----------------------------
                                // Channel observation LLR
                                //
                                // Positive LLR favors bit 0.
                                // Negative LLR favors bit 1.
                                //
                                // q=0.5: prior=0.
                                //
                                // At p=0, the observation is
                                // taken as certain.
                                // ----------------------------

                                const float L =
                                    (p == 0.0)
                                    ? (
                                        y == 0
                                        ? lambda
                                        : -lambda
                                    )
                                    : prior + (
                                        y == 0
                                        ? lambda
                                        : -lambda
                                    );

                                // ============================
                                // 17. Apply reconstruction
                                //     offset and scrambling
                                // ============================

                                // Syndrome:
                                //   Z = frozen-helper offset
                                //
                                // Code offset:
                                //   Z = X XOR encode(S)
                                //
                                // Optional scrambling adds
                                // another valid polar codeword.

                                const uint8_t shift =
                                    Z[i] ^
                                    (
                                        scramble_codeword
                                        ? random_x[i]
                                        : 0
                                    );

                                // Change LLR sign wherever
                                // the combined offset is 1.

                                llr[i] =
                                    shift ? -L : L;
                            }

                            // ================================
                            // 18. AFF3CT SC decoding
                            // ================================

                            sc.decode_siho(
                                llr,
                                decoded
                            );

                            // ================================
                            // 19. Reconstruct enrolled secret
                            // ================================

                            bool error = false;

                            for (int j = 0; j < K; ++j)
                            {
                                uint8_t estimated =
                                    static_cast<uint8_t>(
                                        decoded[j]
                                    );

                                // Remove information-bit
                                // scrambling before comparison.

                                if (scramble_codeword)
                                {
                                    estimated ^=
                                        random_u[
                                            info_positions[j]
                                        ];
                                }

                                // The true enrolled key is
                                // used only for BLER evaluation.

                                if (estimated != S[j])
                                {
                                    error = true;
                                    break;
                                }
                            }

                            // ================================
                            // 20. Update Monte Carlo counters
                            // ================================

                            if (error)
                            {
                                failures.fetch_add(
                                    1,
                                    std::memory_order_relaxed
                                );
                            }

                            const uint64_t completed =
                                blocks.fetch_add(
                                    1,
                                    std::memory_order_relaxed
                                ) + 1;

                            // ================================
                            // 21. Periodic progress output
                            // ================================

                            if (
                                completed % PRINT_INTERVAL == 0
                            )
                            {
                                std::lock_guard<std::mutex> lock(
                                    progress_mutex
                                );

                                print_progress(
                                    p,
                                    blocks,
                                    failures
                                );
                            }
                        }
                    }

                    catch (...)
                    {
                        // Capture the first worker exception
                        // instead of terminating the process.

                        {
                            std::lock_guard<std::mutex> lock(
                                exception_mutex
                            );

                            if (!first_exception)
                            {
                                first_exception =
                                    std::current_exception();
                            }
                        }

                        abort.store(
                            true,
                            std::memory_order_relaxed
                        );
                    }
                });
            }

            // ------------------------------------------------
            // Wait for all workers
            // ------------------------------------------------

            for (auto& worker : workers)
            {
                worker.join();
            }
        }

        // ====================================================
        // 22. Propagate worker exceptions
        // ====================================================

        if (first_exception)
        {
            std::rethrow_exception(
                first_exception
            );
        }

        // ====================================================
        // 23. Final simulation statistics
        // ====================================================

        const auto end =
            std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(
                end - start
            ).count();

        const uint64_t total_blocks =
            blocks.load();

        const uint64_t total_errors =
            failures.load();

        if (total_blocks == 0)
        {
            throw std::runtime_error(
                "Monte Carlo completed zero frames."
            );
        }

        const double bler =
            static_cast<double>(
                total_errors
            ) / total_blocks;

        const double frames_per_second =
            seconds > 0.0
            ? static_cast<double>(
                total_blocks
            ) / seconds
            : 0.0;

        // ----------------------------------------------------
        // Final progress output
        // ----------------------------------------------------

        print_progress(
            p,
            blocks,
            failures
        );

        std::cout << '\n';

        // ====================================================
        // 24. Return results to Python
        // ====================================================

        py::dict result;

        result["blocks"] = total_blocks;

        result["failures"] = total_errors;

        result["BLER"] = bler;

        result["seconds"] = seconds;

        result["frames_per_second"] =
            frames_per_second;

        return result;
    }
};


// ============================================================
// 25. Python binding
// ============================================================
//
// Registers the legacy BSCMonteCarlo class.
//
// This function is called from sc_bindings.cpp.
//
// The API is retained for compatibility with existing
// Python scripts.
//
// ============================================================

void bind_bsc_monte_carlo(
    py::module_& m
)
{
    py::class_<BSCMonteCarlo>(
        m,
        "BSCMonteCarlo"
    )

    // --------------------------------------------------------
    // Constructor
    // --------------------------------------------------------

    .def(
        py::init<
            int,
            std::vector<int>,
            py::array_t<uint8_t>,
            py::array_t<uint8_t>,
            py::array_t<uint8_t>,
            double,
            int,
            bool
        >(),

        py::arg("N"),

        py::arg("frozen_positions"),

        py::arg("X"),

        py::arg("Z"),

        py::arg("S"),

        py::arg("prob_one") = 0.5,

        py::arg("num_threads") = 1,

        py::arg("scramble_codeword") = false
    )

    // --------------------------------------------------------
    // Run Monte Carlo
    // --------------------------------------------------------

    .def(
        "run",
        &BSCMonteCarlo::run,

        py::arg("p"),

        py::arg("min_errors"),

        py::arg("max_blocks")
    );
}
