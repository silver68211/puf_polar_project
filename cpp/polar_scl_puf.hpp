
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "Module/Decoder/Polar/SCL/Decoder_polar_SCL_naive.hpp"
#include "native_key_checker.hpp"

namespace puf {

struct SCLPUFResult
{
    std::vector<uint8_t> key;

    bool verified = false;

    int list_size = 0;
    int surviving_paths = 0;
    int candidates_checked = 0;
    int verified_candidates = 0;

    int selected_path = -1;
    float path_metric = 0.0f;
};


class PolarSCLPUF final
    : public aff3ct::module::Decoder_polar_SCL_naive<int, float>
{
private:

    using Base =
        aff3ct::module::Decoder_polar_SCL_naive<int, float>;

    NativeKeyChecker checker_;

    std::vector<int> info_positions_;
    std::vector<uint8_t> xor_mask_;

    SCLPUFResult last_result_;

    bool selection_called_ = false;

    // --------------------------------------------------
    // Convert frozen positions into AFF3CT representation
    // --------------------------------------------------

    static std::vector<bool> make_frozen(
        int n,
        const std::vector<int>& frozen_positions)
    {
        if (n < 2 || (n & (n - 1)))
            throw std::invalid_argument(
                "N must be a power of two."
            );

        if (frozen_positions.empty() ||
            frozen_positions.size() >= static_cast<size_t>(n))
            throw std::invalid_argument(
                "Require 0 < number of frozen bits < N."
            );

        std::vector<bool> frozen(
            static_cast<size_t>(n), false
        );

        for (int i : frozen_positions)
        {
            if (i < 0 || i >= n ||
                frozen[static_cast<size_t>(i)])
                throw std::invalid_argument(
                    "Invalid or duplicate frozen position."
                );

            frozen[static_cast<size_t>(i)] = true;
        }

        return frozen;
    }

    // --------------------------------------------------
    // Extract information bits from one surviving path
    // --------------------------------------------------

    std::vector<uint8_t> extract_key(int path) const
    {
        std::vector<int> bits(
            static_cast<size_t>(this->K), 0
        );

        int k = 0;

        this->recursive_store(
            this->polar_trees[path].get_root(),
            bits.data(),
            k
        );

        if (k != this->K)
            throw std::runtime_error(
                "Incorrect number of extracted information bits."
            );

        std::vector<uint8_t> key(
            static_cast<size_t>(this->K)
        );

        for (int i = 0; i < this->K; ++i)
        {
            const uint8_t bit =
                static_cast<uint8_t>(bits[i]);

            if (bit > 1)
                throw std::runtime_error(
                    "Nonbinary SCL output."
                );

            key[static_cast<size_t>(i)] =
                bit ^ (
                    xor_mask_.empty()
                    ? 0
                    : xor_mask_[static_cast<size_t>(i)]
                );
        }

        return key;
    }

protected:

    // --------------------------------------------------
    // Replace AFF3CT's default minimum-metric selection
    // with PUF hash/CRC-aided selection.
    // --------------------------------------------------

    void select_best_path(
        const size_t frame_id) override
    {
        (void)frame_id;

        selection_called_ = true;

        if (this->active_paths.empty())
            throw std::runtime_error(
                "SCL produced no surviving paths."
            );

        // Collect surviving path identifiers
        std::vector<int> paths(
            this->active_paths.begin(),
            this->active_paths.end()
        );

        // Lower path metric means more likely.
        // Stable ordering preserves path-ID order on ties.
        std::stable_sort(
            paths.begin(),
            paths.end(),
            [this](int a, int b)
            {
                return
                    this->polar_trees[a].get_path_metric()
                    <
                    this->polar_trees[b].get_path_metric();
            }
        );

        last_result_.surviving_paths =
            static_cast<int>(paths.size());

        // Default: best-metric candidate
        int selected = paths.front();

        std::vector<uint8_t> selected_key;

        bool found_verified = false;

        // --------------------------------------------------
        // Check surviving candidates in metric order
        // --------------------------------------------------

        for (size_t j = 0; j < paths.size(); ++j)
        {
            const int path = paths[j];

            auto candidate = extract_key(path);

            const bool valid =
                checker_.verify(candidate);

            ++last_result_.candidates_checked;

            if (valid)
                ++last_result_.verified_candidates;

            // Keep the best-metric candidate as fallback
            if (j == 0)
                selected_key = candidate;

            // First verified candidate has the
            // smallest metric among verified candidates
            if (valid && !found_verified)
            {
                selected = path;
                selected_key = candidate;
                found_verified = true;
            }
        }

        last_result_.key =
            std::move(selected_key);

        last_result_.verified =
            found_verified;

        last_result_.selected_path =
            selected;

        last_result_.path_metric =
            this->polar_trees[selected].get_path_metric();

        // AFF3CT's _store() expects one selected path.
        this->active_paths.clear();
        this->active_paths.insert(selected);
    }

public:

    // --------------------------------------------------
    // Constructor
    // --------------------------------------------------

    PolarSCLPUF(
        int n,
        const std::vector<int>& frozen_positions,
        int list_size,
        const std::vector<uint8_t>& expected_tag,
        const std::string& checker = "sha256",
        int hash_bits = 128)
        :
        Base(
            n - static_cast<int>(frozen_positions.size()),
            n,
            list_size,
            make_frozen(n, frozen_positions)
        ),
        checker_(
            checker,
            static_cast<size_t>(
                n - static_cast<int>(frozen_positions.size())
            ),
            expected_tag,
            static_cast<size_t>(hash_bits)
        )
    {
        if (list_size < 1 ||
            (list_size & (list_size - 1)))
            throw std::invalid_argument(
                "List size must be a positive power of two."
            );

        for (int i = 0; i < n; ++i)
        {
            if (!this->frozen_bits[
                    static_cast<size_t>(i)])
                info_positions_.push_back(i);
        }

        last_result_.list_size = list_size;
    }

    // AFF3CT decoder trees are mutable.
    // Do not share an instance between threads.
    PolarSCLPUF(const PolarSCLPUF&) = delete;
    PolarSCLPUF& operator=(const PolarSCLPUF&) = delete;

    int n() const { return this->N; }
    int k() const { return this->K; }
    int list_size() const { return this->L; }

    const std::vector<int>& info_positions() const
    {
        return info_positions_;
    }

    // --------------------------------------------------
    // Decode one zero-frozen polar codeword
    // --------------------------------------------------

    SCLPUFResult decode(
        const float* channel_llrs,
        const std::vector<uint8_t>& xor_mask = {})
    {
        if (!channel_llrs)
            throw std::invalid_argument(
                "channel_llrs cannot be null."
            );

        if (!xor_mask.empty() &&
            xor_mask.size() !=
                static_cast<size_t>(this->K))
            throw std::invalid_argument(
                "xor_mask must contain K bits."
            );

        for (uint8_t bit : xor_mask)
        {
            if (bit > 1)
                throw std::invalid_argument(
                    "xor_mask must be binary."
                );
        }

        xor_mask_ = xor_mask;

        last_result_ = SCLPUFResult{};
        last_result_.list_size = this->L;

        selection_called_ = false;

        // AFF3CT performs:
        //   1. Load channel LLRs
        //   2. SCL path generation
        //   3. select_best_path()
        //   4. Store information bits
        std::vector<int> decoded(
            static_cast<size_t>(this->K), 0
        );

        this->_decode_siho(
            channel_llrs,
            decoded.data(),
            0
        );

        if (!selection_called_)
            throw std::runtime_error(
                "AFF3CT did not invoke select_best_path()."
            );

        // Ensure AFF3CT stored the selected path
        // with the same information-bit convention.
        for (int i = 0; i < this->K; ++i)
        {
            const uint8_t bit =
                static_cast<uint8_t>(decoded[i]) ^
                (
                    xor_mask_.empty()
                    ? 0
                    : xor_mask_[static_cast<size_t>(i)]
                );

            if (bit != last_result_.key[
                    static_cast<size_t>(i)])
                throw std::runtime_error(
                    "SCL selected-path output mismatch."
                );
        }

        return last_result_;
    }
};

} // namespace puf
