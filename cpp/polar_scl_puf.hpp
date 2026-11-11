
#pragma once

// ============================================================
// SHA-256 / CRC16-Aided Polar Successive Cancellation List
// Decoder for SRAM-PUF Key Reconstruction
// ============================================================
//
// Based on AFF3CT's naive SCL decoder.
//
// Supported enrollment schemes:
//
//   1. Syndrome / transform construction
//   2. Conventional code-offset construction
//
// Both schemes provide channel LLRs corresponding to a
// zero-frozen polar codeword.
//
// The decoder is independent of the enrollment scheme.
//
// ------------------------------------------------------------
// Candidate selection
// ------------------------------------------------------------
//
// 1. Perform standard SCL decoding with list size L.
//
// 2. Collect all surviving paths.
//
// 3. Sort paths by increasing path metric.
//
// 4. Extract information bits from each surviving path.
//
// 5. Remove the optional K-bit scrambling mask.
//
// 6. Verify each candidate using the PUBLIC SHA-256
//    or CRC16 tag.
//
// 7. Select the verified candidate with the smallest
//    path metric.
//
// 8. If no candidate passes verification, select
//    the overall minimum-metric path.
//
// IMPORTANT:
//
// The true enrolled secret is never used for selecting
// SCL candidates.
//
// The public verification tag is the only criterion
// used to identify a verified candidate.
//
// One decoder instance must be created per worker thread,
// because AFF3CT's decoding trees are mutable.
//
// ============================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "Module/Decoder/Polar/SCL/Decoder_polar_SCL_naive.hpp"
#include "native_key_checker.hpp"


namespace puf {

// ============================================================
// 1. SCL decoding result
// ============================================================

struct SCLPUFResult
{
    // Reconstructed K-bit secret.
    // Any optional scrambling mask has been removed.
    std::vector<uint8_t> key;

    // True if at least one surviving candidate passes
    // public SHA-256 or CRC16 verification.
    bool verified = false;

    // Configured SCL list size.
    int list_size = 0;

    // Number of surviving paths before final selection.
    int surviving_paths = 0;

    // Number of candidate keys tested against the public tag.
    int candidates_checked = 0;

    // Number of candidates accepted by the public verifier.
    int verified_candidates = 0;

    // Selected AFF3CT path identifier.
    int selected_path = -1;

    // Path metric of the selected candidate.
    // Smaller values indicate better likelihood.
    float path_metric = 0.0f;
};


// ============================================================
// 2. Polar SCL decoder
// ============================================================

class PolarSCLPUF final
    : public aff3ct::module::Decoder_polar_SCL_naive<int, float>
{
private:

    using Base =
        aff3ct::module::Decoder_polar_SCL_naive<int, float>;

    // --------------------------------------------------------
    // Native public-tag verifier
    // --------------------------------------------------------

    NativeKeyChecker checker_;

    // --------------------------------------------------------
    // Polar-code information positions
    // --------------------------------------------------------

    std::vector<int> info_positions_;

    // --------------------------------------------------------
    // Optional K-bit information scrambling mask
    // --------------------------------------------------------

    std::vector<uint8_t> xor_mask_;

    // --------------------------------------------------------
    // Most recent decoding result
    // --------------------------------------------------------

    SCLPUFResult last_result_;

    // Confirm that AFF3CT invoked select_best_path().
    bool selection_called_ = false;


    // ========================================================
    // 3. Validate frozen positions
    // ========================================================
    //
    // AFF3CT represents frozen positions using:
    //
    //   true  = frozen bit
    //   false = information bit
    //
    // ========================================================

    static std::vector<bool> make_frozen(
        int n,
        const std::vector<int>& frozen_positions
    )
    {
        // ----------------------------------------------------
        // Validate code length
        // ----------------------------------------------------

        if (
            n < 2 ||
            (n & (n - 1))
        )
        {
            throw std::invalid_argument(
                "N must be a power of two."
            );
        }

        // ----------------------------------------------------
        // Validate number of frozen positions
        // ----------------------------------------------------

        if (
            frozen_positions.empty() ||
            frozen_positions.size() >=
                static_cast<size_t>(n)
        )
        {
            throw std::invalid_argument(
                "Require 0 < number of frozen bits < N."
            );
        }

        // ----------------------------------------------------
        // Construct frozen-bit mask
        // ----------------------------------------------------

        std::vector<bool> frozen(
            static_cast<size_t>(n),
            false
        );

        for (int i : frozen_positions)
        {
            if (
                i < 0 ||
                i >= n ||
                frozen[static_cast<size_t>(i)]
            )
            {
                throw std::invalid_argument(
                    "Invalid or duplicate frozen position."
                );
            }

            frozen[static_cast<size_t>(i)] = true;
        }

        return frozen;
    }


    // ========================================================
    // 4. Extract information bits from a surviving path
    // ========================================================
    //
    // The extracted information bits are in ascending
    // polar information-position order.
    //
    // If scrambling is enabled, remove the K-bit mask
    // before returning the key.
    //
    // The returned key is ready for public-tag verification.
    //
    // ========================================================

    std::vector<uint8_t> extract_key(
        int path
    ) const
    {
        // ----------------------------------------------------
        // Allocate decoded information-bit buffer
        // ----------------------------------------------------

        std::vector<int> bits(
            static_cast<size_t>(this->K),
            0
        );

        int k = 0;

        // ----------------------------------------------------
        // Extract information bits from AFF3CT path tree
        // ----------------------------------------------------

        this->recursive_store(
            this->polar_trees[path].get_root(),
            bits.data(),
            k
        );

        if (k != this->K)
        {
            throw std::runtime_error(
                "Incorrect number of extracted "
                "information bits."
            );
        }

        // ----------------------------------------------------
        // Convert decoded bits and remove scrambling
        // ----------------------------------------------------

        std::vector<uint8_t> key(
            static_cast<size_t>(this->K)
        );

        for (int i = 0; i < this->K; ++i)
        {
            const uint8_t bit =
                static_cast<uint8_t>(bits[i]);

            if (bit > 1)
            {
                throw std::runtime_error(
                    "Nonbinary SCL output."
                );
            }

            // Remove the optional scrambling mask.
            //
            // For an empty mask, the XOR term is zero.

            key[static_cast<size_t>(i)] =
                bit ^
                (
                    xor_mask_.empty()
                    ? 0
                    : xor_mask_[static_cast<size_t>(i)]
                );
        }

        return key;
    }


protected:

    // ========================================================
    // 5. Override AFF3CT final-path selection
    // ========================================================
    //
    // Default AFF3CT SCL selects the path with the
    // lowest path metric.
    //
    // This implementation performs public-tag-aided
    // candidate selection.
    //
    // Selection rules:
    //
    //   - Among verified candidates, select the one
    //     with the smallest path metric.
    //
    //   - If none verifies, select the overall
    //     minimum-metric candidate.
    //
    // All surviving candidates are checked to collect
    // complete verification statistics.
    //
    // ========================================================

    void select_best_path(
        const size_t frame_id
    ) override
    {
        (void)frame_id;

        selection_called_ = true;

        // ----------------------------------------------------
        // Check surviving paths
        // ----------------------------------------------------

        if (this->active_paths.empty())
        {
            throw std::runtime_error(
                "SCL produced no surviving paths."
            );
        }

        // ----------------------------------------------------
        // Collect surviving path identifiers
        // ----------------------------------------------------

        std::vector<int> paths(
            this->active_paths.begin(),
            this->active_paths.end()
        );

        // ----------------------------------------------------
        // Sort paths by increasing path metric
        //
        // Lower metric = better likelihood.
        //
        // Stable sorting preserves the original order
        // of paths when metrics are equal.
        // ----------------------------------------------------

        std::stable_sort(
            paths.begin(),
            paths.end(),

            [this](int a, int b)
            {
                return
                    this->polar_trees[a]
                        .get_path_metric()
                    <
                    this->polar_trees[b]
                        .get_path_metric();
            }
        );

        // ----------------------------------------------------
        // Record number of surviving paths
        // ----------------------------------------------------

        last_result_.surviving_paths =
            static_cast<int>(paths.size());

        // ----------------------------------------------------
        // Initialize fallback to best-metric path
        // ----------------------------------------------------

        int selected = paths.front();

        std::vector<uint8_t> selected_key;

        bool found_verified = false;

        // ====================================================
        // 6. Public-tag verification of surviving candidates
        // ====================================================

        for (size_t j = 0; j < paths.size(); ++j)
        {
            const int path = paths[j];

            // -----------------------------------------------
            // Extract candidate and remove scrambling mask
            // -----------------------------------------------

            auto candidate = extract_key(path);

            // -----------------------------------------------
            // Verify using public SHA-256 or CRC16 tag
            // -----------------------------------------------

            const bool valid =
                checker_.verify(candidate);

            // -----------------------------------------------
            // Update verification statistics
            // -----------------------------------------------

            ++last_result_.candidates_checked;

            if (valid)
            {
                ++last_result_.verified_candidates;
            }

            // -----------------------------------------------
            // Retain minimum-metric candidate as fallback
            // -----------------------------------------------

            if (j == 0)
            {
                selected_key = candidate;
            }

            // -----------------------------------------------
            // Select first verified candidate
            //
            // Since paths are sorted by increasing metric,
            // this is the best-metric verified candidate.
            // -----------------------------------------------

            if (
                valid &&
                !found_verified
            )
            {
                selected = path;

                selected_key = candidate;

                found_verified = true;
            }
        }

        // ====================================================
        // 7. Store final selected candidate
        // ====================================================

        last_result_.key =
            std::move(selected_key);

        last_result_.verified =
            found_verified;

        last_result_.selected_path =
            selected;

        last_result_.path_metric =
            this->polar_trees[selected]
                .get_path_metric();

        // ----------------------------------------------------
        // Restrict AFF3CT to the selected path
        //
        // The base decoder's _store() operation expects
        // exactly one selected surviving path.
        // ----------------------------------------------------

        this->active_paths.clear();

        this->active_paths.insert(selected);
    }


public:

    // ========================================================
    // 8. Constructor
    // ========================================================

    PolarSCLPUF(
        int n,
        const std::vector<int>& frozen_positions,
        int list_size,
        const std::vector<uint8_t>& expected_tag,
        const std::string& checker = "sha256",
        int hash_bits = 128
    )
        : Base(
              n - static_cast<int>(frozen_positions.size()),
              n,
              list_size,
              make_frozen(n, frozen_positions)
          ),
          checker_(
              checker,
              static_cast<size_t>(
                  n - static_cast<int>(
                      frozen_positions.size()
                  )
              ),
              expected_tag,
              static_cast<size_t>(hash_bits)
          )
    {
        // ----------------------------------------------------
        // Validate SCL list size
        // ----------------------------------------------------

        if (
            list_size < 1 ||
            (list_size & (list_size - 1))
        )
        {
            throw std::invalid_argument(
                "List size must be a positive power of two."
            );
        }

        // ----------------------------------------------------
        // Collect information-bit positions
        // ----------------------------------------------------

        for (int i = 0; i < n; ++i)
        {
            if (
                !this->frozen_bits[
                    static_cast<size_t>(i)
                ]
            )
            {
                info_positions_.push_back(i);
            }
        }

        // ----------------------------------------------------
        // Initialize result metadata
        // ----------------------------------------------------

        last_result_.list_size = list_size;
    }


    // ========================================================
    // 9. Disable copying
    // ========================================================
    //
    // AFF3CT decoding trees are mutable.
    // Each worker must own its decoder instance.
    //
    // ========================================================

    PolarSCLPUF(const PolarSCLPUF&) = delete;

    PolarSCLPUF& operator=(
        const PolarSCLPUF&
    ) = delete;


    // ========================================================
    // 10. Decoder accessors
    // ========================================================

    int n() const
    {
        return this->N;
    }

    int k() const
    {
        return this->K;
    }

    int list_size() const
    {
        return this->L;
    }

    const std::vector<int>& info_positions() const
    {
        return info_positions_;
    }


    // ========================================================
    // 11. Decode one zero-frozen polar codeword
    // ========================================================
    //
    // Parameters:
    //
    //   channel_llrs:
    //       N channel LLR values, after applying the
    //       enrollment offset and optional scrambling.
    //
    //   xor_mask:
    //       Optional K-bit information mask.
    //       Removed before candidate verification.
    //
    // Decoding procedure:
    //
    //   1. Validate input.
    //
    //   2. Load the scrambling mask.
    //
    //   3. Run AFF3CT SCL decoding.
    //
    //   4. Invoke select_best_path() for public-tag-aided
    //      candidate selection.
    //
    //   5. Verify AFF3CT's stored information bits match
    //      the selected candidate after unmasking.
    //
    //   6. Return the reconstructed key and statistics.
    //
    // ========================================================

    SCLPUFResult decode(
        const float* channel_llrs,
        const std::vector<uint8_t>& xor_mask = {}
    )
    {
        // ----------------------------------------------------
        // Validate channel-LLR pointer
        // ----------------------------------------------------

        if (!channel_llrs)
        {
            throw std::invalid_argument(
                "channel_llrs cannot be null."
            );
        }

        // ----------------------------------------------------
        // Validate optional scrambling mask
        // ----------------------------------------------------

        if (
            !xor_mask.empty() &&
            xor_mask.size() !=
                static_cast<size_t>(this->K)
        )
        {
            throw std::invalid_argument(
                "xor_mask must contain K bits."
            );
        }

        for (uint8_t bit : xor_mask)
        {
            if (bit > 1)
            {
                throw std::invalid_argument(
                    "xor_mask must be binary."
                );
            }
        }

        // ----------------------------------------------------
        // Store current frame's scrambling mask
        // ----------------------------------------------------

        xor_mask_ = xor_mask;

        // ----------------------------------------------------
        // Reset decoding result
        // ----------------------------------------------------

        last_result_ = SCLPUFResult{};

        last_result_.list_size = this->L;

        selection_called_ = false;

        // ====================================================
        // 12. Run AFF3CT SCL decoding
        // ====================================================
        //
        // AFF3CT performs:
        //
        //   1. Load channel LLRs.
        //
        //   2. Generate list-decoding paths.
        //
        //   3. Invoke our select_best_path() override.
        //
        //   4. Store selected information bits.
        //
        // ====================================================

        std::vector<int> decoded(
            static_cast<size_t>(this->K),
            0
        );

        this->_decode_siho(
            channel_llrs,
            decoded.data(),
            0
        );

        // ----------------------------------------------------
        // Ensure our path-selection override was called
        // ----------------------------------------------------

        if (!selection_called_)
        {
            throw std::runtime_error(
                "AFF3CT did not invoke "
                "select_best_path()."
            );
        }

        // ====================================================
        // 13. Verify AFF3CT selected-path output
        // ====================================================
        //
        // The information-bit convention used by AFF3CT
        // must match extract_key().
        //
        // The same scrambling mask is removed here
        // before comparing the decoded key.
        //
        // ====================================================

        for (int i = 0; i < this->K; ++i)
        {
            const uint8_t bit =
                static_cast<uint8_t>(
                    decoded[i]
                ) ^
                (
                    xor_mask_.empty()
                    ? 0
                    : xor_mask_[
                        static_cast<size_t>(i)
                    ]
                );

            if (
                bit !=
                last_result_.key[
                    static_cast<size_t>(i)
                ]
            )
            {
                throw std::runtime_error(
                    "SCL selected-path output mismatch."
                );
            }
        }

        return last_result_;
    }
};

} // namespace puf
