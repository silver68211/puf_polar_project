#pragma once

// Order-one dynamic SC-Flip decoder for zero-frozen polar codes.
// Keeps AFF3CT's SC tree, f/g/h functions, and channel-LLR sign convention.
// Candidate ranking matches DSCFlip.py (log-domain DSCF metric).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Module/Decoder/Polar/SC/Decoder_polar_SC_naive.hpp"

namespace puf {

struct DSCFResult {
    std::vector<uint8_t> key;         // decoded information bits; XOR-unmasked if requested
    std::vector<float> leaf_llrs;     // N leaf LLRs from the selected path
    std::vector<int> ranked_positions; // information-bit indices in decreasing metric order
    std::vector<float> log_metrics;   // scores in information-bit order
    bool verified = false;            // independent CRC/hash tag check passed
    int attempts = 0;                 // total SC passes, including initial pass
    int flip_position = -1;           // -1 if initial SC chosen or no candidate validated
};

class PolarDSCF final : public aff3ct::module::Decoder_polar_SC_naive<int, float> {
    using Base = aff3ct::module::Decoder_polar_SC_naive<int, float>;
    using Node = aff3ct::tools::Binary_node<aff3ct::module::Contents_SC<int, float>>;

    std::vector<int> info_;
    float alpha_;
    int max_trials_;
    int flip_position_ = -1;

    static std::vector<bool> make_frozen(int n, const std::vector<int>& frozen_positions) {
        if (n <= 1 || (n & (n - 1)) != 0)
            throw std::invalid_argument("N must be a power of two greater than one");
        std::vector<bool> frozen(static_cast<size_t>(n), false);
        for (int i : frozen_positions) {
            if (i < 0 || i >= n || frozen[static_cast<size_t>(i)])
                throw std::invalid_argument("Frozen positions must be unique indices in [0,N)");
            frozen[static_cast<size_t>(i)] = true;
        }
        if (frozen_positions.empty() || frozen_positions.size() >= static_cast<size_t>(n))
            throw std::invalid_argument("Require 0 < number of frozen positions < N");
        return frozen;
    }

    // AFF3CT's recursive SC update, with one difference at an information leaf:
    // invert the hard decision if that leaf is the selected flip position.
    void recursive_decode(const Node* node) override {
        auto* c = node->get_c();
        if (node->is_leaf()) {
            const int i = node->get_lane_id();
            int bit = !c->is_frozen_bit && aff3ct::tools::h_LLR<int, float>(c->lambda[0]);
            if (i == flip_position_ && !c->is_frozen_bit)
                bit ^= 1;
            c->s[0] = bit;
            return;
        }

        const int half = static_cast<int>(c->lambda.size()) / 2;
        const auto* left = node->get_left();
        const auto* right = node->get_right();
        auto* lc = left->get_c();
        auto* rc = right->get_c();

        for (int j = 0; j < half; ++j)
            lc->lambda[j] = aff3ct::tools::f_LLR(c->lambda[j], c->lambda[half + j]);
        recursive_decode(left);

        for (int j = 0; j < half; ++j)
            rc->lambda[j] = aff3ct::tools::g_LLR(c->lambda[j], c->lambda[half + j], lc->s[j]);
        recursive_decode(right);

        for (int j = 0; j < half; ++j) {
            c->s[j] = lc->s[j] ^ rc->s[j];
            c->s[half + j] = rc->s[j];
        }
    }

    void read_tree(std::vector<uint8_t>& key, std::vector<float>& llrs) const {
        const auto& leaves = this->polar_tree.get_leaves();
        key.resize(info_.size());
        llrs.resize(static_cast<size_t>(this->N));
        for (int i = 0; i < this->N; ++i)
            llrs[static_cast<size_t>(i)] = leaves[static_cast<size_t>(i)]->get_c()->lambda[0];
        for (size_t j = 0; j < info_.size(); ++j)
            key[j] = static_cast<uint8_t>(leaves[static_cast<size_t>(info_[j])]->get_c()->s[0]);
    }

    void score_candidates(const std::vector<float>& leaf_llrs,
                          std::vector<float>& scores,
                          std::vector<int>& ranked) const {
        scores.resize(info_.size());
        ranked.resize(info_.size());
        std::iota(ranked.begin(), ranked.end(), 0);

        // log M_i = -alpha*abs(L_i) - sum_{j<=i} softplus(-alpha*abs(L_j))
        float prefix = 0.0f;
        for (size_t j = 0; j < info_.size(); ++j) {
            const float x = -alpha_ * std::fabs(leaf_llrs[static_cast<size_t>(info_[j])]);
            prefix += std::log1p(std::exp(x)); // x<=0, so numerically stable
            scores[j] = x - prefix;
        }

        // Stable descending sort: ties remain in increasing information-bit order.
        std::stable_sort(ranked.begin(), ranked.end(),
                         [&scores](int a, int b) { return scores[a] > scores[b]; });
        for (int& j : ranked)
            j = info_[static_cast<size_t>(j)];
    }

public:
    PolarDSCF(int n, const std::vector<int>& frozen_positions,
              float alpha = 0.3f, int max_trials = 10)
        : Base(n - static_cast<int>(frozen_positions.size()), n,
               make_frozen(n, frozen_positions)),
          alpha_(alpha), max_trials_(max_trials) {
        if (!(std::isfinite(alpha_) && alpha_ > 0.0f))
            throw std::invalid_argument("alpha must be finite and positive");
        if (max_trials_ < 0)
            throw std::invalid_argument("max_trials must be nonnegative");
        for (int i = 0; i < n; ++i)
            if (!this->frozen_bits[static_cast<size_t>(i)]) info_.push_back(i);
    }

    // Do not copy/share a decoder instance across threads: the AFF3CT tree is mutable.
    PolarDSCF(const PolarDSCF&) = delete;
    PolarDSCF& operator=(const PolarDSCF&) = delete;

    int n() const { return this->N; }
    int k() const { return this->K; }
    const std::vector<int>& info_positions() const { return info_; }

    // check(key) must verify a PUBLIC stored tag, never compare to the secret key.
    // xor_mask (K bits, optional) removes a polar-codeword mask before verification.
    template<class Check>
    DSCFResult decode(const float* channel_llrs, Check&& check,
                      const std::vector<uint8_t>& xor_mask = {}) {
        if (!channel_llrs)
            throw std::invalid_argument("channel_llrs cannot be null");
        if (!xor_mask.empty() && xor_mask.size() != info_.size())
            throw std::invalid_argument("xor_mask must have K bits");

        DSCFResult out;
        std::vector<uint8_t> current_key;
        std::vector<float> current_llrs;

        auto apply_mask = [&xor_mask](std::vector<uint8_t>& key) {
            if (!xor_mask.empty())
                for (size_t i = 0; i < key.size(); ++i)
                    key[i] ^= xor_mask[i];
        };

        flip_position_ = -1;
        this->_load(channel_llrs);                 // root LLRs persist across retries
        recursive_decode(this->polar_tree.get_root());
        read_tree(current_key, current_llrs);
        apply_mask(current_key);
        out.attempts = 1;
        out.key = current_key;                     // keep original SC estimate if all fail
        out.leaf_llrs = current_llrs;
        out.verified = static_cast<bool>(check(current_key));
        if (out.verified || max_trials_ == 0) return out;

        score_candidates(current_llrs, out.log_metrics, out.ranked_positions);
        const int trials = std::min(max_trials_, static_cast<int>(info_.size()));
        for (int t = 0; t < trials; ++t) {
            flip_position_ = out.ranked_positions[static_cast<size_t>(t)];
            recursive_decode(this->polar_tree.get_root());
            read_tree(current_key, current_llrs);
            apply_mask(current_key);
            ++out.attempts;
            if (static_cast<bool>(check(current_key))) {
                out.key = std::move(current_key);
                out.leaf_llrs = std::move(current_llrs);
                out.verified = true;
                out.flip_position = flip_position_;
                break;
            }
        }
        flip_position_ = -1;
        return out;
    }
};

} // namespace puf
