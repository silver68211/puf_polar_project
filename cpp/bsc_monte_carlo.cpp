
// #include <aff3ct.hpp>

// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
// #include <pybind11/stl.h>

// #include <algorithm>
// #include <atomic>
// #include <chrono>
// #include <cmath>
// #include <cstdint>
// #include <memory>
// #include <random>
// #include <stdexcept>
// #include <thread>
// #include <vector>
// #include <iomanip>
// #include <iostream>

// namespace py = pybind11;


// // a function to format output


// #include <iostream>
// #include <iomanip>
// #include <atomic>
// #include <cstdint>

// void print_progress(
//     double p,
//     const std::atomic<uint64_t>& blocks,
//     const std::atomic<uint64_t>& failures)
// {
//     const uint64_t n_blocks = blocks.load();
//     const uint64_t n_failures = failures.load();

//     const double bler = n_blocks > 0
//         ? static_cast<double>(n_failures) / n_blocks
//         : 0.0;

//     std::cout << "\r\033[2K"
//               << std::fixed << std::setprecision(3)
//               << "p: " << std::setw(6) << p
//               << " | Blocks: " << std::setw(12) << n_blocks
//               << " | Failures: " << std::setw(6) << n_failures
//               << " | BLER: " << std::scientific << std::setprecision(5)
//               << bler
//               << std::flush;
// }


// // Convert a one-dimensional NumPy binary array to C++ vector.
// static std::vector<uint8_t> read_bits(
//     py::array_t<uint8_t, py::array::c_style |
//                          py::array::forcecast> array,
//     int expected_length)
// {
//     auto a = array.request();

//     if (a.ndim != 1 || a.shape[0] != expected_length)
//         throw std::invalid_argument("Invalid binary array length.");

//     const auto* ptr = static_cast<const uint8_t*>(a.ptr);
//     std::vector<uint8_t> bits(ptr, ptr + expected_length);

//     for (auto bit : bits)
//         if (bit > 1)
//             throw std::invalid_argument("Bits must be 0 or 1.");

//     return bits;
// }



// static void polar_transform(std::vector<uint8_t>& x)
// {
//     const int N = static_cast<int>(x.size());

//     for (int step = 1; step < N; step *= 2)
//     {
//         for (int start = 0; start < N; start += 2 * step)
//         {
//             for (int j = 0; j < step; ++j)
//             {
//                 x[start + j] ^= x[start + j + step];
//             }
//         }
//     }
// }


// class BSCMonteCarlo
// {
// private:
//     int N;
//     int K;
//     int num_threads;
//     double q;
//     bool scramble_codeword;

//     std::vector<bool> frozen_mask;
//     std::vector<int> info_positions;

//     std::vector<uint8_t> X;       // Fixed SRAM pattern
//     std::vector<uint8_t> Z;       // Fixed coset shift
//     std::vector<uint8_t> S;       // Fixed secret key

// public:

//     BSCMonteCarlo(
//         int n,
//         std::vector<int> frozen_positions,
//         py::array_t<uint8_t> x,
//         py::array_t<uint8_t> z,
//         py::array_t<uint8_t> key,
//         double prob_one,
//         int threads, 
//         bool use_scrambling = false)
//         : N(n), K(0), num_threads(threads), q(prob_one), scramble_codeword(use_scrambling)
//     {
//         if (N < 2 || (N & (N - 1)) != 0)
//             throw std::invalid_argument("N must be a power of two.");

//         if (num_threads < 1)
//             throw std::invalid_argument("num_threads must be >= 1.");

//         if (!(q > 0.0 && q < 1.0))
//             throw std::invalid_argument("q must be in (0,1).");

//         // AFF3CT: true = frozen, false = information.
//         frozen_mask.assign(N, false);

//         for (int pos : frozen_positions)
//         {
//             if (pos < 0 || pos >= N || frozen_mask[pos])
//                 throw std::invalid_argument(
//                     "Invalid or repeated frozen position.");

//             frozen_mask[pos] = true;
//         }

//         // Information positions in ascending index order.
//         for (int i = 0; i < N; ++i)
//             if (!frozen_mask[i])
//                 info_positions.push_back(i);

//         K = static_cast<int>(info_positions.size());

//         if (K <= 0 || K >= N)
//             throw std::invalid_argument("Require 0 < K < N.");

//         // Copy device information from Python.
//         X = read_bits(x, N);
//         Z = read_bits(z, N);
//         S = read_bits(key, K);
//     }


//     py::dict run(
//         double p,
//         uint64_t min_errors,
//         uint64_t max_blocks)
//     {
//         if (p < 0.0 || p >= 0.5)
//             throw std::invalid_argument("Require 0 <= p < 0.5.");

//         if (min_errors == 0 || max_blocks == 0)
//             throw std::invalid_argument(
//                 "min_errors and max_blocks must be positive.");

//         // Channel LLR magnitude.
//         // p=0 is supported for noiseless correctness testing.
//         const float lambda = (p == 0.0)
//             ? 50.0f
//             : static_cast<float>(std::log((1.0 - p) / p));

//         // Prior LLR for biased SRAM.
//         const float prior = static_cast<float>(
//             std::log((1.0 - q) / q));

//         // Shared atomic counters.
//         std::atomic<uint64_t> next_frame{0};
//         std::atomic<uint64_t> blocks{0};
//         std::atomic<uint64_t> failures{0};

//         // One independent AFF3CT decoder per worker.
//         // Do not share one mutable decoder across threads.
//         using SCDecoder =
//             aff3ct::module::Decoder_polar_SC_naive<int, float>;

//         std::vector<std::unique_ptr<SCDecoder>> decoders;

//         for (int t = 0; t < num_threads; ++t)
//         {
//             decoders.emplace_back(
//                 std::make_unique<SCDecoder>(K, N, frozen_mask));
//         }

//         const auto start = std::chrono::steady_clock::now();

//         {
//             // Python is free while the C++ workers run.
//             py::gil_scoped_release release;

//             std::vector<std::thread> workers;
//             workers.reserve(num_threads);

//             for (int t = 0; t < num_threads; ++t)
//             {
//                 workers.emplace_back([&, t]()
//                 {
//                     // Fresh independent pseudorandom stream
//                     // for each worker, without a fixed seed.
//                     std::random_device rd;

//                     std::seed_seq seed{
//                         rd(), rd(), rd(), rd(),
//                         static_cast<unsigned int>(t)
//                     };

//                     std::mt19937_64 rng(seed);
//                     std::bernoulli_distribution noise(p);

//                     // Buffers are private to this worker.
//                     std::vector<float> llr(N);
//                     std::vector<int> decoded(K);

//                     auto& sc = *decoders[t];

//                     while (failures.load(
//                                std::memory_order_relaxed) < min_errors)
//                     {
//                         // Allocate one independent noisy read.
//                         uint64_t id = next_frame.fetch_add(
//                             1, std::memory_order_relaxed);

//                         if (id >= max_blocks)
//                             break;

//                         // --------------------------------------
//                         // 1. Generate BSC noise and channel LLRs
//                         // --------------------------------------
//                         for (int i = 0; i < N; ++i)
//                         {
//                             // uint8_t e = noise(rng) ? 1 : 0;

//                             // // Noisy SRAM observation:
//                             // // Y = X XOR E
//                             // uint8_t y = X[i] ^ e;

//                             // // Biased/unbiased posterior LLR.
//                             // float L = prior +
//                             //     (y == 0 ? lambda : -lambda);

//                             // // Apply coset shift Z:
//                             // // L' = (-1)^Z * L
//                             // llr[i] = Z[i] ? -L : L;

//                             // 1. BPSK modulation: 0 -> +1, 1 -> -1
//                             // BSC crossover probability
//                             const float lambda = std::log((1.0f - p) / p);

//                             // Received binary bit
//                             const uint8_t y = X[i] ^ (noise(rng) ? 1 : 0);

//                             // AFF3CT LLR: positive favors bit 0
//                             const float L = (y == 0) ? lambda : -lambda;

//                             // Apply SRAM-PUF coset shift
//                             llr[i] = Z[i] ? -L : L;
//                         }

//                         // --------------------------------------
//                         // 2. AFF3CT SC decoding
//                         // --------------------------------------
//                         sc.decode_siho(llr, decoded);

//                         // --------------------------------------
//                         // 3. Compare reconstructed key with S
//                         // --------------------------------------
//                         bool error = false;

//                         for (int j = 0; j < K; ++j)
//                         {
//                             if (decoded[j] != S[j])
//                             {
//                                 error = true;
//                                 break;
//                             }
//                         }

//                         blocks.fetch_add(
//                             1, std::memory_order_relaxed);

//                         if (error)
//                             failures.fetch_add(
//                                 1, std::memory_order_relaxed);

//                         // Print progress every 1000 blocks.
//                         if (id % 1000 == 0){
//                             print_progress(p, blocks, failures);
//                         }
//                     }
//                 });
//             }

//             // Wait until every C++ worker finishes.
//             for (auto& worker : workers)
//                 worker.join();
//         }

//         const auto end = std::chrono::steady_clock::now();

//         double seconds =
//             std::chrono::duration<double>(end - start).count();

//         const uint64_t total_blocks = blocks.load();
//         const uint64_t total_errors = failures.load();

//         py::dict result;

//         result["blocks"] = total_blocks;
//         result["failures"] = total_errors;
//         result["BLER"] =
//             static_cast<double>(total_errors) / total_blocks;
//         result["seconds"] = seconds;
//         result["frames_per_second"] =
//             static_cast<double>(total_blocks) / seconds;
//         std::cout << std::endl;
//         return result;
//     }
// };


// // Called from sc_bindings.cpp.
// void bind_bsc_monte_carlo(py::module_& m)
// {
//     py::class_<BSCMonteCarlo>(m, "BSCMonteCarlo")

//         .def(py::init<
//             int,
//             std::vector<int>,
//             py::array_t<uint8_t>,
//             py::array_t<uint8_t>,
//             py::array_t<uint8_t>,
//             double,
//             int
//         >(),
//             py::arg("N"),
//             py::arg("frozen_positions"),
//             py::arg("X"),
//             py::arg("Z"),
//             py::arg("S"),
//             py::arg("prob_one") = 0.5,
//             py::arg("num_threads") = 1)

//         .def(
//             "run",
//             &BSCMonteCarlo::run,
//             py::arg("p"),
//             py::arg("min_errors"),
//             py::arg("max_blocks")
//         );
// }



#include <aff3ct.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;


// ==========================================================
// 1. Progress printing
// ==========================================================

static void print_progress(
    double p,
    const std::atomic<uint64_t>& blocks,
    const std::atomic<uint64_t>& failures)
{
    const uint64_t n_blocks = blocks.load();
    const uint64_t n_failures = failures.load();

    const double bler = n_blocks > 0
        ? static_cast<double>(n_failures) / n_blocks
        : 0.0;

    std::cout << "\r\033[2K"
              << std::fixed << std::setprecision(3)
              << "p: " << std::setw(6) << p
              << " | Blocks: " << std::setw(12) << n_blocks
              << " | Failures: " << std::setw(6) << n_failures
              << " | BLER: "
              << std::scientific << std::setprecision(5)
              << bler
              << std::flush;
}


// ==========================================================
// 2. Convert and validate Python binary arrays
// ==========================================================

static std::vector<uint8_t> read_bits(
    py::array_t<uint8_t,
        py::array::c_style | py::array::forcecast> array,
    int expected_length)
{
    auto a = array.request();

    if (a.ndim != 1 || a.shape[0] != expected_length)
        throw std::invalid_argument(
            "Invalid binary array length.");

    const auto* ptr =
        static_cast<const uint8_t*>(a.ptr);

    std::vector<uint8_t> bits(
        ptr, ptr + expected_length);

    for (auto bit : bits)
    {
        if (bit > 1)
            throw std::invalid_argument(
                "Bits must be 0 or 1.");
    }

    return bits;
}


// ==========================================================
// 3. Non-systematic polar transform
//
// x = u * F^(tensor log2(N)) over GF(2)
//
// Used for generating random scrambling codewords.
// ==========================================================

static void polar_transform(std::vector<uint8_t>& x)
{
    const int N = static_cast<int>(x.size());

    for (int step = 1; step < N; step *= 2)
    {
        for (int start = 0;
             start < N;
             start += 2 * step)
        {
            for (int j = 0; j < step; ++j)
            {
                x[start + j] ^=
                    x[start + j + step];
            }
        }
    }
}


// ==========================================================
// 4. BSC Monte Carlo simulator
// ==========================================================

class BSCMonteCarlo
{
private:

    int N;
    int K;
    int num_threads;

    double q;

    // Enable random valid-codeword scrambling.
    bool scramble_codeword;

    // Polar code construction.
    std::vector<bool> frozen_mask;
    std::vector<int> info_positions;

    // Fixed SRAM-PUF enrollment.
    std::vector<uint8_t> X;
    std::vector<uint8_t> Z;
    std::vector<uint8_t> S;


public:

    // ------------------------------------------------------
    // Constructor
    // ------------------------------------------------------

    BSCMonteCarlo(
        int n,
        std::vector<int> frozen_positions,
        py::array_t<uint8_t> x,
        py::array_t<uint8_t> z,
        py::array_t<uint8_t> key,
        double prob_one,
        int threads,
        bool use_scrambling = false)

        : N(n),
          K(0),
          num_threads(threads),
          q(prob_one),
          scramble_codeword(use_scrambling)
    {
        if (N < 2 || (N & (N - 1)) != 0)
            throw std::invalid_argument(
                "N must be a power of two.");

        if (num_threads < 1)
            throw std::invalid_argument(
                "num_threads must be >= 1.");

        if (!(q > 0.0 && q < 1.0))
            throw std::invalid_argument(
                "q must be in (0,1).");

        // AFF3CT:
        // true  = frozen bit
        // false = information bit
        frozen_mask.assign(N, false);

        for (int pos : frozen_positions)
        {
            if (pos < 0 ||
                pos >= N ||
                frozen_mask[pos])
            {
                throw std::invalid_argument(
                    "Invalid or repeated frozen position.");
            }

            frozen_mask[pos] = true;
        }

        // Store information positions in ascending order.
        for (int i = 0; i < N; ++i)
        {
            if (!frozen_mask[i])
                info_positions.push_back(i);
        }

        K = static_cast<int>(info_positions.size());

        if (K <= 0 || K >= N)
            throw std::invalid_argument(
                "Require 0 < K < N.");

        // Copy enrollment data from Python.
        X = read_bits(x, N);
        Z = read_bits(z, N);
        S = read_bits(key, K);

        // --------------------------------------------------
        // Validate the enrollment / coset transformation.
        //
        // U = (X XOR Z) * G_N
        //
        // Frozen bits of U must be zero.
        // Information bits of U must equal S.
        // --------------------------------------------------

        std::vector<uint8_t> u_check(N);

        for (int i = 0; i < N; ++i)
            u_check[i] = X[i] ^ Z[i];

        polar_transform(u_check);

        for (int i = 0; i < N; ++i)
        {
            if (frozen_mask[i] && u_check[i] != 0)
                throw std::invalid_argument(
                    "Invalid coset shift: frozen bit is nonzero.");
        }

        for (int j = 0; j < K; ++j)
        {
            if (u_check[info_positions[j]] != S[j])
                throw std::invalid_argument(
                    "Enrollment key does not match X and Z.");
        }
    }


    // ------------------------------------------------------
    // Run Monte Carlo simulation for one crossover p.
    // ------------------------------------------------------

    py::dict run(
        double p,
        uint64_t min_errors,
        uint64_t max_blocks)
    {
        if (p < 0.0 || p >= 0.5)
            throw std::invalid_argument(
                "Require 0 <= p < 0.5.");

        if (min_errors == 0 || max_blocks == 0)
            throw std::invalid_argument(
                "min_errors and max_blocks must be positive.");

        // --------------------------------------------------
        // BSC LLR magnitude
        //
        // lambda = log((1-p)/p)
        //
        // For p=0, use a finite large magnitude.
        // --------------------------------------------------

        const float lambda = (p == 0.0)
            ? 50.0f
            : static_cast<float>(
                std::log((1.0 - p) / p));

        // SRAM source prior.
        // Equals zero for q=0.5.
        const float prior = static_cast<float>(
            std::log((1.0 - q) / q));

        // --------------------------------------------------
        // Shared Monte Carlo counters
        // --------------------------------------------------

        std::atomic<uint64_t> next_frame{0};
        std::atomic<uint64_t> blocks{0};
        std::atomic<uint64_t> failures{0};

        // Protect terminal output.
        std::mutex progress_mutex;

        constexpr uint64_t PRINT_INTERVAL = 100000;

        // --------------------------------------------------
        // Create one independent AFF3CT decoder per worker
        // --------------------------------------------------

        using SCDecoder =
            aff3ct::module::Decoder_polar_SC_naive<
                int, float>;

        std::vector<std::unique_ptr<SCDecoder>> decoders;

        decoders.reserve(num_threads);

        for (int t = 0; t < num_threads; ++t)
        {
            decoders.emplace_back(
                std::make_unique<SCDecoder>(
                    K, N, frozen_mask));
        }

        const auto start =
            std::chrono::steady_clock::now();

        {
            // Release Python GIL during C++ simulation.
            py::gil_scoped_release release;

            std::vector<std::thread> workers;
            workers.reserve(num_threads);

            // ==============================================
            // Parallel Monte Carlo workers
            // ==============================================

            for (int t = 0; t < num_threads; ++t)
            {
                workers.emplace_back([&, t]()
                {
                    // --------------------------------------
                    // Independent random generator
                    // --------------------------------------

                    std::random_device rd;

                    std::seed_seq seed{
                        rd(), rd(), rd(), rd(),
                        static_cast<unsigned int>(t)
                    };

                    std::mt19937_64 rng(seed);

                    // BSC noise: flip with probability p.
                    // std::bernoulli_distribution noise(p);
                    const uint64_t noise_threshold = static_cast<uint64_t>(std::ldexp(p, 64));

                    // Uniform random scrambling bits.
                    std::bernoulli_distribution mask_bit(0.5);

                    // --------------------------------------
                    // Worker-local buffers
                    // --------------------------------------

                    std::vector<float> llr(N);
                    std::vector<int> decoded(K);

                    // Random polar information vector.
                    std::vector<uint8_t> random_u(N, 0);

                    // Encoded random scrambling codeword.
                    std::vector<uint8_t> random_x(N, 0);

                    auto& sc = *decoders[t];

                    // ======================================
                    // Main simulation loop
                    // ======================================

                    while (failures.load(
                               std::memory_order_relaxed)
                           < min_errors)
                    {
                        // Allocate one unique frame.
                        const uint64_t id =
                            next_frame.fetch_add(
                                1,
                                std::memory_order_relaxed);

                        if (id >= max_blocks)
                            break;

                        // ----------------------------------
                        // 1. Generate random codeword mask
                        // ----------------------------------

                        if (scramble_codeword)
                        {
                            // Frozen bits must be zero.
                            std::fill(
                                random_u.begin(),
                                random_u.end(),
                                0);

                            // Randomize information bits.
                            for (int j = 0; j < K; ++j)
                            {
                                random_u[info_positions[j]] =
                                    mask_bit(rng) ? 1 : 0;
                            }

                            // Encode random information bits.
                            random_x = random_u;

                            polar_transform(random_x);
                        }

                        // ----------------------------------
                        // 2. BSC channel and LLR generation
                        // ----------------------------------

                        for (int i = 0; i < N; ++i)
                        {
                            // BSC noise bit.
                            const uint8_t e =
                                static_cast<uint8_t>(rng() < noise_threshold);
                                // noise(rng) ? 1 : 0;

                            // Received physical SRAM bit.
                            // Y = X XOR E
                            const uint8_t y = X[i] ^ e;

                            // AFF3CT LLR convention:
                            // positive -> bit 0
                            // negative -> bit 1

                            // For p=0, the channel
                            // observation is certain.
                            const float L = (p == 0.0)
                                ? (y == 0 ? lambda : -lambda)
                                : prior +
                                    (y == 0 ? lambda : -lambda);

                            // --------------------------------
                            // 3. Apply coset and scrambling
                            // --------------------------------

                            const uint8_t shift =
                                Z[i] ^
                                (scramble_codeword
                                    ? random_x[i]
                                    : 0);

                            llr[i] = shift ? -L : L;
                        }

                        // ----------------------------------
                        // 4. AFF3CT SC decoding
                        // ----------------------------------

                        sc.decode_siho(llr, decoded);

                        // ----------------------------------
                        // 5. Recover original secret key
                        // ----------------------------------

                        bool error = false;

                        for (int j = 0; j < K; ++j)
                        {
                            uint8_t estimated =
                                static_cast<uint8_t>(
                                    decoded[j]);

                            // Remove the random mask.
                            if (scramble_codeword)
                            {
                                estimated ^=
                                    random_u[info_positions[j]];
                            }

                            // Compare with enrolled secret.
                            if (estimated != S[j])
                            {
                                error = true;
                                break;
                            }
                        }

                        // ----------------------------------
                        // 6. Update simulation statistics
                        // ----------------------------------

                        if (error)
                        {
                            failures.fetch_add(
                                1,
                                std::memory_order_relaxed);
                        }

                        const uint64_t completed =
                            blocks.fetch_add(
                                1,
                                std::memory_order_relaxed) + 1;

                        // ----------------------------------
                        // 7. Periodic progress output
                        // ----------------------------------

                        if (completed % PRINT_INTERVAL == 0)
                        {
                            std::lock_guard<std::mutex> lock(
                                progress_mutex);

                            print_progress(
                                p, blocks, failures);
                        }
                    }
                });
            }

            // Wait for all workers.
            for (auto& worker : workers)
            {
                worker.join();
            }
        }

        // ==============================================
        // Final results
        // ==============================================

        const auto end =
            std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(
                end - start).count();

        const uint64_t total_blocks = blocks.load();
        const uint64_t total_errors = failures.load();

        const double bler =
            static_cast<double>(total_errors) /
            total_blocks;

        // Final accurate progress line.
        print_progress(p, blocks, failures);

        std::cout << '\n';

        py::dict result;

        result["blocks"] = total_blocks;
        result["failures"] = total_errors;
        result["BLER"] = bler;
        result["seconds"] = seconds;

        result["frames_per_second"] =
            static_cast<double>(total_blocks) /
            seconds;

        return result;
    }
};


// ==========================================================
// 5. Python binding
// ==========================================================

// This function must be called by PYBIND11_MODULE
// in sc_bindings.cpp.

void bind_bsc_monte_carlo(py::module_& m)
{
    py::class_<BSCMonteCarlo>(
        m, "BSCMonteCarlo")

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

        .def(
            "run",
            &BSCMonteCarlo::run,

            py::arg("p"),
            py::arg("min_errors"),
            py::arg("max_blocks")
        );
}
