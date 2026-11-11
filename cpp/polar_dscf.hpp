
#pragma once

// ============================================================
// Order-One Dynamic Successive Cancellation Flip (DSCF)
// ============================================================
//
// Polar decoder for zero-frozen polar codes.
//
// Based on AFF3CT's naive SC decoder, preserving:
//
//   - SC decoding tree
//   - Polar f/g/h operations
//   - Channel-LLR sign convention
//   - Frozen and information-bit positions
//
// DSCF procedure:
//
//   1. Perform standard SC decoding.
//   2. Verify the decoded information bits.
//   3. If verification fails, rank information-bit positions.
//   4. Flip one selected SC decision per additional attempt.
//   5. Stop when verification succeeds or T trials are exhausted.
//
// Candidate ranking uses the log-domain DSCF metric:
//
//   log(M_i) = -alpha * |L_i|
//              - sum_{j <= i} log(1 + exp(-alpha * |L_j|))
//
// The sum runs over information-bit positions in SC order.
//
// Supported enrollment schemes:
//
//   - Syndrome / transform enrollment
//   - Conventional code-offset enrollment
//
// The enrollment scheme is handled outside this decoder.
//
// Both schemes supply channel LLRs corresponding to a
// zero-frozen polar codeword.
//
// Optional scrambling is removed from the decoded
// information bits before candidate verification.
//
// IMPORTANT:
//
//   The decoder never uses the true enrolled secret
//   to select candidates.
//
//   Candidate verification is performed through the
//   supplied callback, normally using a public SHA-256
//   or CRC16 verification tag.
//
// ============================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Module/Decoder/Polar/SC/Decoder_polar_SC_naive.hpp"


namespace puf {

// ============================================================
// 1. DSCF decoding result
// ============================================================

struct DSCFResult
{
    // Decoded K information bits.
    // The optional scrambling mask has been removed.
    std::vector<uint8_t> key;

    // Leaf LLR values from the retained decoding path.
    std::vector<float> leaf_llrs;

    // Information-bit positions ranked by decreasing
    // DSCF metric, using the initial SC decoding pass.
    std::vector<int> ranked_positions;

    // Log-domain DSCF scores in information-bit order.
    std::vector<float> log_metrics;

    // Whether the supplied verification callback accepted
    // the returned key.
    //
    // Note: In native SC mode, the callback always returns
    // true without checking a hash/CRC.
    bool verified = false;

    // Total SC decoding passes, including the initial pass.
    int attempts = 0;

    // Position flipped in the accepted DSCF attempt.
    //
    // -1: Initial SC estimate retained, or no retry
    //     candidate passed verification.
    int flip_position = -1;
};


// ============================================================
// 2. Polar DSCF decoder
// ============================================================

class PolarDSCF final
    : public aff3ct::module::Decoder_polar_SC_naive<int, float>
{
private:

    using Base =
        aff3ct::module::Decoder_polar_SC_naive<int, float>;

    using Node =
        aff3ct::tools::Binary_node<
            aff3ct::module::Contents_SC<int, float>
        >;

    // --------------------------------------------------------
    // Decoder configuration
    // --------------------------------------------------------

    // Sorted information-bit positions.
    std::vector<int> info_;

    // DSCF metric parameter.
    float alpha_;

    // Maximum number of additional flip attempts.
    int max_trials_;

    // Selected bit position for the current retry.
    //
    // -1 means ordinary SC decoding.
    int flip_position_ = -1;


    // ========================================================
    // 3. Validate and construct frozen-bit vector
    // ========================================================

    static std::vector<bool> make_frozen(
        int n,
        const std::vector<int>& frozen_positions
    )
    {
        // Polar-code length must be a power of two.

        if (n <= 1 || (n & (n - 1)) != 0)
        {
            throw std::invalid_argument(
                "N must be a power of two greater than one."
            );
        }

        std::vector<bool> frozen(
            static_cast<size_t>(n),
            false
        );

        // Validate frozen positions.

        for (int i : frozen_positions)
        {
            if (
                i < 0 ||
                i >= n ||
                frozen[static_cast<size_t>(i)]
            )
            {
                throw std::invalid_argument(
                    "Frozen positions must be unique "
                    "indices in [0,N)."
                );
            }

            frozen[static_cast<size_t>(i)] = true;
        }

        // Require at least one information bit
        // and at least one frozen bit.

        if (
            frozen_positions.empty() ||
            frozen_positions.size() >= static_cast<size_t>(n)
        )
        {
            throw std::invalid_argument(
                "Require 0 < number of frozen positions < N."
            );
        }

        return frozen;
    }


    // ========================================================
    // 4. Recursive SC decoding with optional bit flip
    // ========================================================
    //
    // Uses the same recursive SC tree operations as AFF3CT.
    //
    // At an information leaf:
    //
    //   - Make the standard hard decision.
    //   - Invert the decision if the leaf matches
    //     the selected DSCF flip position.
    //
    // Frozen bits always remain zero.
    //
    // ========================================================

    void recursive_decode(const Node* node) override
    {
        auto* c = node->get_c();

        // ----------------------------------------------------
        // Leaf node
        // ----------------------------------------------------

        if (node->is_leaf())
        {
            const int i = node->get_lane_id();

            int bit =
                !c->is_frozen_bit &&
                aff3ct::tools::h_LLR<int, float>(
                    c->lambda[0]
                );

            // Apply the selected DSCF flip.
            if (
                i == flip_position_ &&
                !c->is_frozen_bit
            )
            {
                bit ^= 1;
            }

            c->s[0] = bit;

            return;
        }

        // ----------------------------------------------------
        // Internal node
        // ----------------------------------------------------

        const int half =
            static_cast<int>(c->lambda.size()) / 2;

        const auto* left = node->get_left();
        const auto* right = node->get_right();

        auto* lc = left->get_c();
        auto* rc = right->get_c();

        // ----------------------------------------------------
        // Left child: f operation
        // ----------------------------------------------------

        for (int j = 0; j < half; ++j)
        {
            lc->lambda[j] =
                aff3ct::tools::f_LLR(
                    c->lambda[j],
                    c->lambda[half + j]
                );
        }

        recursive_decode(left);

        // ----------------------------------------------------
        // Right child: g operation
        // ----------------------------------------------------

        for (int j = 0; j < half; ++j)
        {
            rc->lambda[j] =
                aff3ct::tools::g_LLR(
                    c->lambda[j],
                    c->lambda[half + j],
                    lc->s[j]
                );
        }

        recursive_decode(right);

        // ----------------------------------------------------
        // Combine partial sums
        // ----------------------------------------------------

        for (int j = 0; j < half; ++j)
        {
            c->s[j] =
                lc->s[j] ^ rc->s[j];

            c->s[half + j] =
                rc->s[j];
        }
    }


    // ========================================================
    // 5. Extract decoded key and leaf LLRs
    // ========================================================

    void read_tree(
        std::vector<uint8_t>& key,
        std::vector<float>& llrs
    ) const
    {
        const auto& leaves =
            this->polar_tree.get_leaves();

        key.resize(info_.size());

        llrs.resize(
            static_cast<size_t>(this->N)
        );

        // ----------------------------------------------------
        // Read all leaf LLRs
        // ----------------------------------------------------

        for (int i = 0; i < this->N; ++i)
        {
            llrs[static_cast<size_t>(i)] =
                leaves[static_cast<size_t>(i)]
                    ->get_c()
                    ->lambda[0];
        }

        // ----------------------------------------------------
        // Extract information-bit decisions
        // ----------------------------------------------------

        for (size_t j = 0; j < info_.size(); ++j)
        {
            key[j] = static_cast<uint8_t>(
                leaves[static_cast<size_t>(info_[j])]
                    ->get_c()
                    ->s[0]
            );
        }
    }


    // ========================================================
    // 6. Compute DSCF candidate-ranking metrics
    // ========================================================
    //
    // For information-bit position i:
    //
    //   x_i = -alpha * abs(L_i)
    //
    //   log(M_i) = x_i
    //              - sum_{j <= i} log(1 + exp(x_j))
    //
    // The log-domain calculation avoids directly
    // multiplying small probability terms.
    //
    // The candidates are ranked in descending metric order.
    //
    // ========================================================

    void score_candidates(
        const std::vector<float>& leaf_llrs,
        std::vector<float>& scores,
        std::vector<int>& ranked
    ) const
    {
        scores.resize(info_.size());

        ranked.resize(info_.size());

        std::iota(
            ranked.begin(),
            ranked.end(),
            0
        );

        // ----------------------------------------------------
        // Compute log-domain DSCF metrics
        // ----------------------------------------------------

        float prefix = 0.0f;

        for (size_t j = 0; j < info_.size(); ++j)
        {
            const float x =
                -alpha_ *
                std::fabs(
                    leaf_llrs[static_cast<size_t>(info_[j])]
                );

            // Since x <= 0, this expression is
            // numerically stable for finite LLRs.

            prefix += std::log1p(
                std::exp(x)
            );

            scores[j] = x - prefix;
        }

        // ----------------------------------------------------
        // Rank candidate indices by decreasing score
        // ----------------------------------------------------

        std::stable_sort(
            ranked.begin(),
            ranked.end(),
            [&scores](int a, int b)
            {
                return scores[a] > scores[b];
            }
        );

        // ----------------------------------------------------
        // Convert information-bit order to polar positions
        // ----------------------------------------------------

        for (int& j : ranked)
        {
            j = info_[static_cast<size_t>(j)];
        }
    }


public:

    // ========================================================
    // 7. Constructor
    // ========================================================

    PolarDSCF(
        int n,
        const std::vector<int>& frozen_positions,
        float alpha = 0.3f,
        int max_trials = 10
    )
        : Base(
              n - static_cast<int>(frozen_positions.size()),
              n,
              make_frozen(n, frozen_positions)
          ),
          alpha_(alpha),
          max_trials_(max_trials)
    {
        // ----------------------------------------------------
        // Validate DSCF parameters
        // ----------------------------------------------------

        if (
            !std::isfinite(alpha_) ||
            alpha_ <= 0.0f
        )
        {
            throw std::invalid_argument(
                "alpha must be finite and positive."
            );
        }

        if (max_trials_ < 0)
        {
            throw std::invalid_argument(
                "max_trials must be nonnegative."
            );
        }

        // ----------------------------------------------------
        // Extract information-bit positions
        // ----------------------------------------------------

        for (int i = 0; i < n; ++i)
        {
            if (
                !this->frozen_bits[
                    static_cast<size_t>(i)
                ]
            )
            {
                info_.push_back(i);
            }
        }
    }


    // ========================================================
    // 8. Disable copying
    // ========================================================
    //
    // Each decoder owns a mutable AFF3CT decoding tree.
    // One decoder instance must be used per worker thread.
    //
    // ========================================================

    PolarDSCF(const PolarDSCF&) = delete;

    PolarDSCF& operator=(
        const PolarDSCF&
    ) = delete;


    // ========================================================
    // 9. Accessors
    // ========================================================

    int n() const
    {
        return this->N;
    }

    int k() const
    {
        return this->K;
    }

    const std::vector<int>& info_positions() const
    {
        return info_;
    }


    // ========================================================
    // 10. Decode using SC / order-one DSCF
    // ========================================================
    //
    // Parameters:
    //
    //   channel_llrs:
    //       N channel LLR values after applying the
    //       appropriate enrollment offset.
    //
    //   check:
    //       Callback accepting a decoded candidate key
    //       and returning true when the candidate passes
    //       verification.
    //
    //   xor_mask:
    //       Optional K-bit information mask.
    //       Removed before candidate verification.
    //
    // Behavior:
    //
    //   max_trials = 0:
    //       Standard SC decoding, one pass.
    //
    //   max_trials > 0:
    //       Initial SC followed by at most max_trials
    //       additional single-bit-flip decoding passes.
    //
    // If all candidate verifications fail, the initial
    // SC estimate is returned.
    //
    // ========================================================

    template <class Check>
    DSCFResult decode(
        const float* channel_llrs,
        Check&& check,
        const std::vector<uint8_t>& xor_mask = {}
    )
    {
        // ----------------------------------------------------
        // Validate input
        // ----------------------------------------------------

        if (!channel_llrs)
        {
            throw std::invalid_argument(
                "channel_llrs cannot be null."
            );
        }

        if (
            !xor_mask.empty() &&
            xor_mask.size() != info_.size()
        )
        {
            throw std::invalid_argument(
                "xor_mask must have K bits."
            );
        }

        // ----------------------------------------------------
        // Initialize decoding result and buffers
        // ----------------------------------------------------

        DSCFResult out;

        std::vector<uint8_t> current_key;
        std::vector<float> current_llrs;

        // ----------------------------------------------------
        // Optional removal of information-bit scrambling
        // ----------------------------------------------------

        auto apply_mask =
            [&xor_mask](std::vector<uint8_t>& key)
        {
            if (!xor_mask.empty())
            {
                for (size_t i = 0; i < key.size(); ++i)
                {
                    key[i] ^= xor_mask[i];
                }
            }
        };

        // ====================================================
        // 11. Initial SC decoding pass
        // ====================================================

        flip_position_ = -1;

        // Load channel LLRs into the AFF3CT tree.
        // Root LLRs persist across additional attempts.

        this->_load(channel_llrs);

        recursive_decode(
            this->polar_tree.get_root()
        );

        // Extract decoded information bits and leaf LLRs.

        read_tree(
            current_key,
            current_llrs
        );

        // Remove scrambling before verification.

        apply_mask(current_key);

        out.attempts = 1;

        // Preserve initial SC result as fallback.

        out.key = current_key;
        out.leaf_llrs = current_llrs;

        // Verify initial candidate using the
        // supplied verification callback.

        out.verified =
            static_cast<bool>(
                check(current_key)
            );

        // No additional attempt is needed if the
        // callback accepts the candidate, or if
        // DSCF retries are disabled.

        if (
            out.verified ||
            max_trials_ == 0
        )
        {
            return out;
        }

        // ====================================================
        // 12. Rank candidate flip positions
        // ====================================================

        score_candidates(
            current_llrs,
            out.log_metrics,
            out.ranked_positions
        );

        const int trials = std::min(
            max_trials_,
            static_cast<int>(info_.size())
        );

        // ====================================================
        // 13. Additional DSCF decoding attempts
        // ====================================================

        for (int t = 0; t < trials; ++t)
        {
            // ------------------------------------------------
            // Select next flip position
            // ------------------------------------------------

            flip_position_ =
                out.ranked_positions[
                    static_cast<size_t>(t)
                ];

            // ------------------------------------------------
            // Re-run SC with the selected bit flipped
            // ------------------------------------------------

            recursive_decode(
                this->polar_tree.get_root()
            );

            // ------------------------------------------------
            // Extract the new candidate
            // ------------------------------------------------

            read_tree(
                current_key,
                current_llrs
            );

            // ------------------------------------------------
            // Remove scrambling before verification
            // ------------------------------------------------

            apply_mask(current_key);

            ++out.attempts;

            // ------------------------------------------------
            // Verify candidate using the public tag
            // ------------------------------------------------

            if (
                static_cast<bool>(
                    check(current_key)
                )
            )
            {
                out.key = std::move(current_key);

                out.leaf_llrs =
                    std::move(current_llrs);

                out.verified = true;

                out.flip_position = flip_position_;

                break;
            }
        }

        // ----------------------------------------------------
        // Reset decoder state for future frames
        // ----------------------------------------------------

        flip_position_ = -1;

        return out;
    }
};

} // namespace puf
