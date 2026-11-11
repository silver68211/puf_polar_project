
#pragma once

// ============================================================
// Native key verification for polar SRAM-PUF reconstruction
// ============================================================
//
// This implementation provides the native C++ equivalent
// of the project's keycheck.py.
//
// Supported verification methods:
//
//   1. SHA-256 (with configurable digest truncation)
//   2. CRC16 (project-specific GF(2) parity construction)
//
// Supported enrollment schemes:
//
//   - Syndrome / transform enrollment
//   - Conventional code-offset enrollment
//
// The verification procedure is identical for both schemes.
// Only the enrolled secret key and its public tag are needed.
//
// SHA-256 input format:
//
//   b"PUF-KEY-CHECK-v1:"
//       + K.to_bytes(4, "big")
//       + np.packbits(key).tobytes()
//
// Key bits are packed MSB-first, with zero padding in the
// final byte when K is not divisible by eight.
//
// CRC16 uses the same generator-matrix construction as
// keycheck.py, rather than a conventional byte-oriented
// CRC implementation.
//
// The enrolled secret is never used for selecting
// decoder candidates. Candidate verification uses only
// the publicly available verification tag.
//
// ============================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


namespace puf {

namespace detail {

// ============================================================
// 1. SHA-256 utilities
// ============================================================

// ------------------------------------------------------------
// Rotate a 32-bit integer to the right.
//
// All rotation counts used by SHA-256 are in [1,31].
// ------------------------------------------------------------

inline uint32_t rotr(uint32_t x, unsigned n)
{
    return (x >> n) | (x << (32 - n));
}


// ============================================================
// 2. SHA-256 implementation
// ============================================================
//
// Input:
//   Arbitrary byte vector.
//
// Output:
//   32-byte SHA-256 digest.
//
// Implements the standard SHA-256 compression function,
// message padding, and digest serialization.
//
// ============================================================

inline std::array<uint8_t, 32> sha256(
    const std::vector<uint8_t>& data
)
{
    // --------------------------------------------------------
    // SHA-256 round constants
    // --------------------------------------------------------

    static constexpr uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

    // --------------------------------------------------------
    // Initial SHA-256 hash state
    // --------------------------------------------------------

    uint32_t h[8] = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19
    };

    // --------------------------------------------------------
    // Message padding
    //
    // Append:
    //   1. A single '1' bit
    //   2. Enough zero bits to reach 448 mod 512
    //   3. Original message length as a 64-bit integer
    // --------------------------------------------------------

    std::vector<uint8_t> m(data);

    const uint64_t bit_count =
        static_cast<uint64_t>(m.size()) * 8;

    m.push_back(0x80);

    while ((m.size() % 64) != 56)
    {
        m.push_back(0);
    }

    for (int i = 7; i >= 0; --i)
    {
        m.push_back(
            static_cast<uint8_t>(
                bit_count >> (i * 8)
            )
        );
    }

    // --------------------------------------------------------
    // Process each 512-bit message block
    // --------------------------------------------------------

    for (size_t off = 0; off < m.size(); off += 64)
    {
        // ----------------------------------------------------
        // Message schedule
        // ----------------------------------------------------

        uint32_t w[64] = {};

        // First 16 words are read directly from the block.

        for (int i = 0; i < 16; ++i)
        {
            const size_t j =
                off + static_cast<size_t>(i) * 4;

            w[i] =
                (uint32_t(m[j]) << 24) |
                (uint32_t(m[j + 1]) << 16) |
                (uint32_t(m[j + 2]) << 8) |
                uint32_t(m[j + 3]);
        }

        // Expand to 64 message-schedule words.

        for (int i = 16; i < 64; ++i)
        {
            const uint32_t a =
                rotr(w[i - 15], 7) ^
                rotr(w[i - 15], 18) ^
                (w[i - 15] >> 3);

            const uint32_t b =
                rotr(w[i - 2], 17) ^
                rotr(w[i - 2], 19) ^
                (w[i - 2] >> 10);

            w[i] =
                w[i - 16] +
                a +
                w[i - 7] +
                b;
        }

        // ----------------------------------------------------
        // Initialize compression working variables
        // ----------------------------------------------------

        uint32_t a = h[0];
        uint32_t b = h[1];
        uint32_t c = h[2];
        uint32_t d = h[3];
        uint32_t e = h[4];
        uint32_t f = h[5];
        uint32_t g = h[6];
        uint32_t z = h[7];

        // ----------------------------------------------------
        // SHA-256 compression rounds
        // ----------------------------------------------------

        for (int i = 0; i < 64; ++i)
        {
            const uint32_t s1 =
                rotr(e, 6) ^
                rotr(e, 11) ^
                rotr(e, 25);

            const uint32_t ch =
                (e & f) ^
                (~e & g);

            const uint32_t t1 =
                z + s1 + ch + k[i] + w[i];

            const uint32_t s0 =
                rotr(a, 2) ^
                rotr(a, 13) ^
                rotr(a, 22);

            const uint32_t maj =
                (a & b) ^
                (a & c) ^
                (b & c);

            const uint32_t t2 = s0 + maj;

            // Update the working state.

            z = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        // ----------------------------------------------------
        // Accumulate compression result
        // ----------------------------------------------------

        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += z;
    }

    // --------------------------------------------------------
    // Serialize hash state as 32 big-endian bytes
    // --------------------------------------------------------

    std::array<uint8_t, 32> out{};

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            out[static_cast<size_t>(i) * 4 + j] =
                static_cast<uint8_t>(
                    h[i] >> (24 - 8 * j)
                );
        }
    }

    return out;
}

} // namespace detail


// ============================================================
// 3. Native key checker
// ============================================================
//
// Used by:
//
//   - Monte Carlo enrollment validation
//   - DSCF candidate verification
//   - SCL candidate selection
//
// Inputs:
//
//   kind       : "sha256" or "crc16"
//   k          : Secret length in bits
//   tag        : Expected public verification tag
//   hash_bits  : Number of SHA-256 digest bits retained
//
// SHA-256 tags are stored as bytes.
// CRC16 tags are stored as individual parity bits.
//
// ============================================================

class NativeKeyChecker
{
private:

    // Verification algorithm.
    std::string kind_;

    // Number of key bits.
    size_t k_;

    // SHA-256 truncation length in bytes.
    size_t digest_bytes_;

    // Public verification tag.
    std::vector<uint8_t> expected_;

    // CRC16 generator rows:
    // One 16-bit parity contribution per key position.
    std::vector<std::array<uint8_t, 16>> crc_rows_;


public:

    // ========================================================
    // 4. Constructor
    // ========================================================

    NativeKeyChecker(
        std::string kind,
        size_t k,
        const std::vector<uint8_t>& tag,
        size_t hash_bits = 128
    )
        : kind_(std::move(kind)),
          k_(k),
          digest_bytes_(hash_bits / 8),
          expected_(tag)
    {
        // ----------------------------------------------------
        // Validate verification method
        // ----------------------------------------------------

        if (
            kind_ != "sha256" &&
            kind_ != "crc16"
        )
        {
            throw std::invalid_argument(
                "Native verifier supports sha256 or crc16 "
                "(not other hash/CRC variants)."
            );
        }

        // ----------------------------------------------------
        // SHA-256 configuration
        // ----------------------------------------------------

        if (kind_ == "sha256")
        {
            if (
                hash_bits == 0 ||
                hash_bits > 256 ||
                hash_bits % 8 != 0 ||
                tag.size() != digest_bytes_
            )
            {
                throw std::invalid_argument(
                    "Invalid SHA-256 tag length."
                );
            }
        }

        // ----------------------------------------------------
        // CRC16 configuration
        // ----------------------------------------------------

        else
        {
            if (tag.size() != 16)
            {
                throw std::invalid_argument(
                    "CRC16 tag must contain "
                    "16 individual bits."
                );
            }

            // ------------------------------------------------
            // Polynomial:
            //
            // P(x) = x^16 + x^12 + x^5 + 1
            //
            // Matches the GF(2) construction used
            // in the project's keycheck.py.
            // ------------------------------------------------

            std::array<uint8_t, 17> poly{};

            for (int exponent : {16, 12, 5, 0})
            {
                poly[
                    static_cast<size_t>(16 - exponent)
                ] = 1;
            }

            // ------------------------------------------------
            // Build CRC generator rows.
            //
            // Each row represents the parity contribution
            // of one information-bit position.
            //
            // These rows are used for fast verification
            // without recomputing polynomial division
            // for every candidate key.
            // ------------------------------------------------

            std::array<uint8_t, 16> rem{};

            rem[0] = 1;

            crc_rows_.resize(k_);

            for (size_t i = 0; i < k_; ++i)
            {
                std::array<uint8_t, 17> extended{};

                // Copy current remainder.

                for (int j = 0; j < 16; ++j)
                {
                    extended[j] = rem[j];
                }

                // Apply polynomial reduction over GF(2).

                if (extended[0])
                {
                    for (int j = 0; j < 17; ++j)
                    {
                        extended[j] ^= poly[j];
                    }
                }

                // Update remainder state.

                for (int j = 0; j < 16; ++j)
                {
                    rem[j] = extended[j + 1];
                }

                // Store rows in the key-bit order.

                crc_rows_[k_ - i - 1] = rem;
            }
        }
    }


    // ========================================================
    // 5. Compute verification tag
    // ========================================================
    //
    // Input:
    //   key: K binary information bits.
    //
    // Output:
    //   SHA-256: truncated digest bytes.
    //   CRC16:   16 individual parity bits.
    //
    // ========================================================

    std::vector<uint8_t> tag(
        const std::vector<uint8_t>& key
    ) const
    {
        // ----------------------------------------------------
        // Validate key length
        // ----------------------------------------------------

        if (key.size() != k_)
        {
            throw std::invalid_argument(
                "Unexpected key length."
            );
        }

        // ====================================================
        // CRC16 tag
        // ====================================================

        if (kind_ == "crc16")
        {
            std::vector<uint8_t> result(16, 0);

            // Combine each active information bit's
            // contribution to the CRC over GF(2).

            for (size_t i = 0; i < k_; ++i)
            {
                if (key[i])
                {
                    for (int j = 0; j < 16; ++j)
                    {
                        result[j] ^= crc_rows_[i][j];
                    }
                }
            }

            return result;
        }

        // ====================================================
        // SHA-256 tag
        // ====================================================

        // ----------------------------------------------------
        // Domain-separation prefix
        // ----------------------------------------------------

        static constexpr char prefix[] =
            "PUF-KEY-CHECK-v1:";

        std::vector<uint8_t> data(
            prefix,
            prefix + sizeof(prefix) - 1
        );

        // ----------------------------------------------------
        // Append key length as a 4-byte big-endian integer
        // ----------------------------------------------------

        const uint32_t len =
            static_cast<uint32_t>(k_);

        for (int i = 3; i >= 0; --i)
        {
            data.push_back(
                static_cast<uint8_t>(
                    len >> (i * 8)
                )
            );
        }

        // ----------------------------------------------------
        // Pack binary key bits into bytes (MSB-first).
        //
        // Matches NumPy's default np.packbits order.
        // ----------------------------------------------------

        for (size_t i = 0; i < k_; i += 8)
        {
            uint8_t byte = 0;

            for (
                size_t b = 0;
                b < 8 && i + b < k_;
                ++b
            )
            {
                byte |= static_cast<uint8_t>(
                    (key[i + b] & 1u) << (7 - b)
                );
            }

            data.push_back(byte);
        }

        // ----------------------------------------------------
        // Compute SHA-256 and truncate the public tag
        // ----------------------------------------------------

        const auto digest = detail::sha256(data);

        return std::vector<uint8_t>(
            digest.begin(),
            digest.begin() +
                static_cast<std::ptrdiff_t>(
                    digest_bytes_
                )
        );
    }


    // ========================================================
    // 6. Verify a candidate secret
    // ========================================================
    //
    // Returns:
    //
    //   true  : Candidate's tag matches the public tag.
    //   false : Candidate's tag does not match.
    //
    // This function does not require the original PUF
    // response or the enrolled secret.
    //
    // Therefore, it works with both enrollment schemes.
    //
    // ========================================================

    bool verify(
        const std::vector<uint8_t>& key
    ) const
    {
        return tag(key) == expected_;
    }
};

} // namespace puf
