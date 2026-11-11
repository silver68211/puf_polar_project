
#pragma once

// Native multithreaded BSC Monte Carlo for SRAM-PUF.
//
// Decoders:
//   SC   : one SC pass per frame
//   DSCF : 1 + actual additional SC passes
//   SCL  : fixed list size L
//
// Normalized complexity:
//   SC   = 1
//   DSCF = average number of actual SC passes
//   SCL  = L (nominal approximation)
//
// All verification is performed in C++.

#include "native_key_checker.hpp"
#include "polar_dscf.hpp"
#include "polar_scl_puf.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <unistd.h>
#include <sys/ioctl.h>


namespace puf {

// ============================================================
// 1. Monte Carlo results
// ============================================================

struct PolarMCResult
{
    uint64_t blocks = 0;
    uint64_t failures = 0;

    // Actual SC passes, meaningful for SC and DSCF only.
    uint64_t total_sc_passes = 0;

    // SC/DSCF: actual number of SC passes
    // SCL: nominal L units per decoded frame
    uint64_t total_sc_equivalent_units = 0;

    uint64_t nlogn_per_pass = 0;
    uint64_t total_sc_equivalent_ops = 0;

    double seconds = 0.0;
    double bler = 0.0;
    double frames_per_second = 0.0;

    double normalized_complexity = 0.0;
    double mean_extra_flips = 0.0;

    int max_observed_passes = 0;
    int list_size = 1;

    std::string complexity_model;
};


// ============================================================
// 2. Live progress display
// ============================================================

inline void print_progress(
    const std::string& decoder,
    double p,
    const std::atomic<uint64_t>& blocks,
    const std::atomic<uint64_t>& failures,
    const std::atomic<uint64_t>& equivalent_units,
    const std::chrono::steady_clock::time_point& start,
    bool finished = false)
{
    const uint64_t n_blocks =
        blocks.load(std::memory_order_relaxed);

    const uint64_t n_failures =
        failures.load(std::memory_order_relaxed);

    const uint64_t n_units =
        equivalent_units.load(std::memory_order_relaxed);

    const double bler = n_blocks > 0
        ? static_cast<double>(n_failures) / n_blocks
        : 0.0;

    // Normalized SC-equivalent complexity.
    const double complexity = n_blocks > 0
        ? static_cast<double>(n_units) / n_blocks
        : 0.0;

    const double seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start
        ).count();

    const double fps = seconds > 0.0
        ? static_cast<double>(n_blocks) / seconds
        : 0.0;

    std::ostringstream line;

    line << "DEC: "
         << std::left << std::setw(10) << decoder

         << " | P: "
         << std::fixed << std::setprecision(3)
         << std::setw(5) << p

         << " | BLER: "
         << std::scientific << std::setprecision(5)
         << bler

         << " | FRMs: "
         << std::scientific << std::setprecision(5)
         << std::right << std::setw(10) << n_blocks

         << " | ERRs: "
         << std::setw(8) << n_failures

        //  << " | SC-equiv: "
        //  << std::setw(12) << n_units

         << " | COM: "
         << std::scientific << std::setprecision(5)
         << std::setw(10) << complexity

         << " | FRMs/s: "
         << std::scientific << std::setprecision(5)
         << std::setw(10) << fps;

    // Rewrite the same terminal line.
    std::cout << "\r\033[2K" << line.str();

    if (finished)
        std::cout << '\n';

    std::cout << std::flush;
}


// ============================================================
// 3. Multithreaded worker management
// ============================================================

template <typename Worker>
inline void run_workers_with_progress(
    int num_threads,
    Worker&& worker,
    const std::string& decoder,
    double p,
    const std::atomic<uint64_t>& completed,
    const std::atomic<uint64_t>& errors,
    const std::atomic<uint64_t>& equivalent_units,
    const std::chrono::steady_clock::time_point& started)
{
    if (num_threads < 1)
        throw std::invalid_argument(
            "Number of threads must be positive."
        );

    std::mutex progress_mutex;
    std::condition_variable progress_cv;

    bool progress_done = false;

    // Dedicated progress monitor.
    std::thread progress_thread([&]()
    {
        std::unique_lock<std::mutex> lock(
            progress_mutex
        );

        while (!progress_done)
        {
            lock.unlock();

            print_progress(
                decoder,
                p,
                completed,
                errors,
                equivalent_units,
                started
            );

            lock.lock();

            progress_cv.wait_for(
                lock,
                std::chrono::milliseconds(500),
                [&]() { return progress_done; }
            );
        }

        lock.unlock();

        // Preserve the final result.
        print_progress(
            decoder,
            p,
            completed,
            errors,
            equivalent_units,
            started,
            true
        );
    });

    // Start decoding workers.
    std::vector<std::thread> workers;

    try
    {
        workers.reserve(
            static_cast<size_t>(num_threads)
        );

        for (int t = 0; t < num_threads; ++t)
        {
            workers.emplace_back(
                [&worker, t]()
                {
                    worker(t);
                }
            );
        }
    }
    catch (...)
    {
        for (auto& w : workers)
            if (w.joinable())
                w.join();

        {
            std::lock_guard<std::mutex> lock(
                progress_mutex
            );

            progress_done = true;
        }

        progress_cv.notify_one();
        progress_thread.join();

        throw;
    }

    // Wait for all workers.
    for (auto& w : workers)
        w.join();

    // Stop the progress monitor.
    {
        std::lock_guard<std::mutex> lock(
            progress_mutex
        );

        progress_done = true;
    }

    progress_cv.notify_one();
    progress_thread.join();
}


// ============================================================
// 4. Non-systematic polar transform
// ============================================================

inline void polar_transform(std::vector<uint8_t>& u)
{
    const size_t n = u.size();

    for (size_t half = 1; half < n; half <<= 1)
    {
        for (size_t start = 0;
             start < n;
             start += 2 * half)
        {
            for (size_t j = 0; j < half; ++j)
            {
                u[start + j] ^=
                    u[start + half + j];
            }
        }
    }
}


// ============================================================
// 5. Monte Carlo engine
// ============================================================

class BSCMonteCarloPolar
{
private:

    int n_;
    int k_;

    int threads_;
    int trials_;
    int list_size_;
    int hash_bits_;

    float alpha_;
    float prob_one_;

    uint64_t seed_;

    std::string decoder_;
    std::string checker_id_;

    bool scramble_;

    std::vector<int> frozen_;
    std::vector<int> info_;

    std::vector<uint8_t> x_;
    std::vector<uint8_t> z_;
    std::vector<uint8_t> secret_;
    std::vector<uint8_t> public_tag_;

    NativeKeyChecker checker_;

    uint64_t nlogn_ = 0;


public:

    // ========================================================
    // 6. Constructor
    // ========================================================

    BSCMonteCarloPolar(
        int n,
        const std::vector<int>& frozen,
        const std::vector<uint8_t>& x,
        const std::vector<uint8_t>& z,
        const std::vector<uint8_t>& secret,
        const std::vector<uint8_t>& expected_tag,
        const std::string& decoder = "SC",
        const std::string& checker = "sha256",
        int hash_bits = 128,
        float prob_one = 0.5f,
        int num_threads = 1,
        float alpha = 0.3367f,
        int num_trials = 10,
        bool scramble_codeword = false,
        uint64_t seed = 2026,
        int list_size = 1
    )
        : n_(n),
          k_(
              n - static_cast<int>(frozen.size())
          ),
          threads_(num_threads),
          trials_(num_trials),
          list_size_(list_size),
          hash_bits_(hash_bits),
          alpha_(alpha),
          prob_one_(prob_one),
          seed_(seed),
          decoder_(decoder),
          checker_id_(checker),
          scramble_(scramble_codeword),
          frozen_(frozen),
          x_(x),
          z_(z),
          secret_(secret),
          public_tag_(expected_tag),
          checker_(
              checker,
              secret.size(),
              expected_tag,
              static_cast<size_t>(hash_bits)
          )
    {
        // ----------------------------------------
        // Validate code parameters
        // ----------------------------------------

        if (
            n_ < 2 ||
            (n_ & (n_ - 1)) ||
            k_ <= 0 ||
            k_ >= n_
        )
        {
            throw std::invalid_argument(
                "Require power-of-two N and 0<K<N."
            );
        }

        if (
            x_.size() != static_cast<size_t>(n_) ||
            z_.size() != x_.size() ||
            secret_.size() != static_cast<size_t>(k_)
        )
        {
            throw std::invalid_argument(
                "X, Z, S must have lengths N, N, K."
            );
        }

        // ----------------------------------------
        // Validate decoder selection
        // ----------------------------------------

        if (
            decoder_ != "SC" &&
            decoder_ != "DSCF" &&
            decoder_ != "SCL"
        )
        {
            throw std::invalid_argument(
                "decoder must be SC, DSCF, or SCL."
            );
        }

        if (
            list_size_ < 1 ||
            (list_size_ & (list_size_ - 1))
        )
        {
            throw std::invalid_argument(
                "SCL list size must be a positive "
                "power of two."
            );
        }

        if (
            threads_ < 1 ||
            trials_ < 0 ||
            !(prob_one_ > 0.0f && prob_one_ < 1.0f) ||
            !std::isfinite(alpha_) ||
            alpha_ <= 0.0f
        )
        {
            throw std::invalid_argument(
                "Invalid threads, trials, alpha "
                "or PUF bit prior."
            );
        }

        // ----------------------------------------
        // Validate frozen positions
        // ----------------------------------------

        std::vector<uint8_t> is_frozen(
            static_cast<size_t>(n_), 0
        );

        for (int i : frozen_)
        {
            if (
                i < 0 ||
                i >= n_ ||
                is_frozen[static_cast<size_t>(i)]
            )
            {
                throw std::invalid_argument(
                    "Invalid or duplicate frozen position."
                );
            }

            is_frozen[static_cast<size_t>(i)] = 1;
        }

        for (int i = 0; i < n_; ++i)
        {
            if (!is_frozen[static_cast<size_t>(i)])
                info_.push_back(i);
        }

        // ----------------------------------------
        // Validate PUF enrollment
        // ----------------------------------------

        // Supports both enrollment schemes:
        //
        // 1. Syndrome construction:
        //    Z = embed((X G_N)_F) G_N
        //    S = (X G_N)_A
        //
        // 2. Conventional code-offset construction:
        //    S is independently generated.
        //    C = encode(S), with zero frozen bits.
        //    Z = X XOR C
        //
        // For both constructions:
        //
        //    C = X XOR Z
        //    U = C G_N
        //
        // The following conditions must hold:
        //
        //    U_F = 0
        //    U_A = S
        //
        // Therefore, the decoder can operate on either
        // enrollment scheme without modification.

        std::vector<uint8_t> transformed = x_;

        for (int i = 0; i < n_; ++i)
        {
            if (
                x_[static_cast<size_t>(i)] > 1 ||
                z_[static_cast<size_t>(i)] > 1
            )
            {
                throw std::invalid_argument(
                    "X and Z must be binary."
                );
            }

            transformed[static_cast<size_t>(i)] ^=
                z_[static_cast<size_t>(i)];
        }

        polar_transform(transformed);

        for (int i = 0; i < n_; ++i)
        {
            if (
                is_frozen[static_cast<size_t>(i)] &&
                transformed[static_cast<size_t>(i)] != 0
            )
            {
                throw std::invalid_argument(
                    "X XOR Z is not zero-frozen. "
                    "Check the encoded helper Z."
                );
            }
        }

        for (int j = 0; j < k_; ++j)
        {
            if (
                secret_[static_cast<size_t>(j)] > 1 ||
                transformed[
                    static_cast<size_t>(info_[j])
                ] != secret_[static_cast<size_t>(j)]
            )
            {
                throw std::invalid_argument(
                    "S is inconsistent with X, Z "
                    "and information positions."
                );
            }
        }

        // Verify the public tag against the
        // enrolled secret once at initialization.
        if (!checker_.verify(secret_))
        {
            throw std::invalid_argument(
                "Public tag does not match "
                "the enrolled secret."
            );
        }

        // N * log2(N)
        for (int i = n_; i > 1; i >>= 1)
        {
            nlogn_ += static_cast<uint64_t>(n_);
        }
    }


    // ========================================================
    // 7. Monte Carlo simulation
    // ========================================================

    PolarMCResult run(
        double p,
        uint64_t min_errors,
        uint64_t max_blocks
    ) const
    {
        if (
            !std::isfinite(p) ||
            p < 0.0 ||
            p >= 0.5 ||
            min_errors == 0 ||
            max_blocks == 0
        )
        {
            throw std::invalid_argument(
                "Require 0<=p<0.5 and positive "
                "stopping limits."
            );
        }

        const auto started =
            std::chrono::steady_clock::now();

        // ----------------------------------------
        // Channel parameters
        // ----------------------------------------

        const float magnitude = p == 0.0
            ? 80.0f
            : static_cast<float>(
                std::min(
                    80.0,
                    std::log((1.0 - p) / p)
                )
            );

        const float prior =
            std::log(
                (1.0f - prob_one_) / prob_one_
            );

        // ----------------------------------------
        // Monte Carlo counters
        // ----------------------------------------

        std::atomic<uint64_t> claimed(0);
        std::atomic<uint64_t> completed(0);
        std::atomic<uint64_t> errors(0);

        // Actual SC passes: SC and DSCF only.
        std::atomic<uint64_t> sc_passes(0);

        // Shared normalized-complexity units.
        std::atomic<uint64_t> equivalent_units(0);

        std::atomic<int> maximum_passes(0);

        // Exception propagation from workers.
        std::atomic<bool> abort(false);

        std::mutex error_mutex;
        std::exception_ptr first_exception;


        // ====================================================
        // 8. Worker function
        // ====================================================

        auto worker = [&](int tid)
        {
            try
            {
                // ------------------------------------
                // Create one decoder per thread
                // ------------------------------------

                std::unique_ptr<PolarDSCF> sc_decoder;
                std::unique_ptr<PolarSCLPUF> scl_decoder;

                if (decoder_ == "SCL")
                {
                    scl_decoder =
                        std::make_unique<PolarSCLPUF>(
                            n_,
                            frozen_,
                            list_size_,
                            public_tag_,
                            checker_id_,
                            hash_bits_
                        );
                }
                else
                {
                    sc_decoder =
                        std::make_unique<PolarDSCF>(
                            n_,
                            frozen_,
                            alpha_,
                            decoder_ == "SC"
                                ? 0
                                : trials_
                        );
                }

                // One native verifier per worker.
                NativeKeyChecker worker_checker(
                    checker_id_,
                    secret_.size(),
                    public_tag_,
                    static_cast<size_t>(hash_bits_)
                );

                // ------------------------------------
                // Random number generator
                // ------------------------------------

                std::seed_seq seedseq{
                    static_cast<uint32_t>(seed_),
                    static_cast<uint32_t>(seed_ >> 32),
                    static_cast<uint32_t>(tid),
                    0xD5CFU
                };

                std::mt19937_64 rng(seedseq);

                std::bernoulli_distribution noise(p);

                // ------------------------------------
                // Reusable per-worker buffers
                // ------------------------------------

                std::vector<float> llr(
                    static_cast<size_t>(n_)
                );

                std::vector<uint8_t> u(
                    static_cast<size_t>(n_), 0
                );

                std::vector<uint8_t> mask(
                    static_cast<size_t>(k_), 0
                );

                const std::vector<uint8_t> empty_mask;


                // =================================================
                // 9. Process Monte Carlo frames
                // =================================================

                while (
                    !abort.load(
                        std::memory_order_relaxed
                    )
                )
                {
                    // Stop after enough block errors.
                    if (
                        errors.load(
                            std::memory_order_relaxed
                        ) >= min_errors
                    )
                    {
                        break;
                    }

                    // --------------------------------
                    // Reserve one frame
                    // --------------------------------

                    uint64_t old = claimed.load(
                        std::memory_order_relaxed
                    );

                    do
                    {
                        if (old >= max_blocks)
                            return;
                    }
                    while (
                        !claimed.compare_exchange_weak(
                            old,
                            old + 1,
                            std::memory_order_relaxed
                        )
                    );

                    // --------------------------------
                    // Optional codeword scrambling
                    // --------------------------------

                    if (scramble_)
                    {
                        std::fill(
                            u.begin(),
                            u.end(),
                            0
                        );

                        uint64_t word = 0;
                        int remaining = 0;

                        for (int j = 0; j < k_; ++j)
                        {
                            if (remaining == 0)
                            {
                                word = rng();
                                remaining = 64;
                            }

                            mask[static_cast<size_t>(j)] =
                                static_cast<uint8_t>(
                                    word & 1ULL
                                );

                            word >>= 1;
                            --remaining;

                            u[
                                static_cast<size_t>(
                                    info_[j]
                                )
                            ] = mask[
                                static_cast<size_t>(j)
                            ];
                        }

                        polar_transform(u);
                    }

                    // --------------------------------
                    // BSC channel and coset shift
                    // --------------------------------

                    for (int i = 0; i < n_; ++i)
                    {
                        const size_t idx =
                            static_cast<size_t>(i);

                        // Physical noisy SRAM bit.
                        const uint8_t y =
                            static_cast<uint8_t>(
                                x_[idx] ^
                                (noise(rng) ? 1u : 0u)
                            );

                        // Positive LLR favors bit 0.
                        const float observation =
                            (y ? -magnitude : magnitude)
                            + prior;

                        // Convert to zero-frozen
                        // polar-code observation.
                        const uint8_t shift =
                            static_cast<uint8_t>(
                                z_[idx] ^
                                (
                                    scramble_
                                    ? u[idx]
                                    : 0u
                                )
                            );

                        llr[idx] = shift
                            ? -observation
                            : observation;
                    }

                    const auto& active_mask =
                        scramble_
                        ? mask
                        : empty_mask;

                    // --------------------------------
                    // Decode
                    // --------------------------------

                    std::vector<uint8_t> decoded_key;

                    uint64_t actual_passes = 0;
                    uint64_t work_units = 0;

                    if (scl_decoder)
                    {
                        // ============================
                        // SCL decoding
                        // ============================

                        const auto result =
                            scl_decoder->decode(
                                llr.data(),
                                active_mask
                            );

                        decoded_key = result.key;

                        // Nominal SC-equivalent
                        // complexity: L per frame.
                        //
                        // This is NOT a measured
                        // number of SC traversals.
                        work_units =
                            static_cast<uint64_t>(
                                list_size_
                            );
                    }
                    else
                    {
                        // ============================
                        // SC / DSCF decoding
                        // ============================

                        const auto check =
                            [&](const std::vector<uint8_t>& candidate)
                        {
                            // SC returns its estimate.
                            // DSCF requires public
                            // verification for retries.
                            return decoder_ == "DSCF"
                                ? worker_checker.verify(
                                    candidate
                                  )
                                : true;
                        };

                        const auto result =
                            sc_decoder->decode(
                                llr.data(),
                                check,
                                active_mask
                            );

                        decoded_key = result.key;

                        if (
                            result.attempts < 1 ||
                            result.attempts >
                                1 + (
                                    decoder_ == "SC"
                                    ? 0
                                    : trials_
                                )
                        )
                        {
                            throw std::runtime_error(
                                "Invalid SC/DSCF "
                                "attempt count."
                            );
                        }

                        // Actual number of SC passes.
                        actual_passes =
                            static_cast<uint64_t>(
                                result.attempts
                            );

                        work_units = actual_passes;

                        // Record maximum passes
                        // observed on any frame.
                        int seen = maximum_passes.load(
                            std::memory_order_relaxed
                        );

                        while (
                            seen < result.attempts &&
                            !maximum_passes
                                .compare_exchange_weak(
                                    seen,
                                    result.attempts,
                                    std::memory_order_relaxed
                                )
                        ) {}
                    }

                    // --------------------------------
                    // Validate decoded key length
                    // --------------------------------

                    if (
                        decoded_key.size() !=
                        static_cast<size_t>(k_)
                    )
                    {
                        throw std::runtime_error(
                            "Decoder returned "
                            "incorrect key length."
                        );
                    }

                    // --------------------------------
                    // Update complexity counters
                    // --------------------------------

                    sc_passes.fetch_add(
                        actual_passes,
                        std::memory_order_relaxed
                    );

                    equivalent_units.fetch_add(
                        work_units,
                        std::memory_order_relaxed
                    );

                    // --------------------------------
                    // Count block errors
                    // --------------------------------

                    // True secret is used only for
                    // performance evaluation.
                    if (decoded_key != secret_)
                    {
                        errors.fetch_add(
                            1,
                            std::memory_order_relaxed
                        );
                    }

                    completed.fetch_add(
                        1,
                        std::memory_order_relaxed
                    );
                }
            }
            catch (...)
            {
                // Avoid std::terminate() when
                // a worker encounters an exception.
                {
                    std::lock_guard<std::mutex> lock(
                        error_mutex
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
        };


        // ====================================================
        // 10. Run workers and display progress
        // ====================================================

        const std::string label =
            decoder_ == "SCL"
            ? "SCL(L=" +
                std::to_string(list_size_) +
                ")"
            : decoder_;

        run_workers_with_progress(
            threads_,
            worker,
            label,
            p,
            completed,
            errors,
            equivalent_units,
            started
        );

        if (first_exception)
            std::rethrow_exception(first_exception);


        // ====================================================
        // 11. Final simulation statistics
        // ====================================================

        const double elapsed =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now()
                - started
            ).count();

        PolarMCResult out;

        out.blocks = completed.load();
        out.failures = errors.load();

        out.total_sc_passes =
            sc_passes.load();

        out.total_sc_equivalent_units =
            equivalent_units.load();

        out.nlogn_per_pass = nlogn_;

        // Check multiplication overflow.
        if (
            out.total_sc_equivalent_units >
            std::numeric_limits<uint64_t>::max()
            / nlogn_
        )
        {
            throw std::overflow_error(
                "SC-equivalent operation "
                "counter overflow."
            );
        }

        out.total_sc_equivalent_ops =
            out.total_sc_equivalent_units
            * nlogn_;

        out.seconds = elapsed;

        out.bler = out.blocks > 0
            ? static_cast<double>(
                out.failures
              ) / out.blocks
            : 0.0;

        out.frames_per_second =
            elapsed > 0.0
            ? static_cast<double>(
                out.blocks
              ) / elapsed
            : 0.0;

        out.normalized_complexity =
            out.blocks > 0
            ? static_cast<double>(
                out.total_sc_equivalent_units
              ) / out.blocks
            : 0.0;

        // ----------------------------------------
        // Decoder-specific statistics
        // ----------------------------------------

        if (decoder_ == "SCL")
        {
            out.list_size = list_size_;

            // SCL operations are only estimated.
            out.complexity_model = "nominal_L";

            out.mean_extra_flips = 0.0;
            out.max_observed_passes = 0;
        }
        else
        {
            out.list_size = 1;

            out.complexity_model =
                "actual_SC_passes";

            out.mean_extra_flips =
                decoder_ == "DSCF"
                ? out.normalized_complexity - 1.0
                : 0.0;

            out.max_observed_passes =
                maximum_passes.load();
        }

        return out;
    }


    // ========================================================
    // 12. Compatibility helper
    // ========================================================

    bool decoder_name_is_dscf() const
    {
        return decoder_ == "DSCF";
    }
};

} // namespace puf
