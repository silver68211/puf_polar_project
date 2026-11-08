#pragma once
// Native equivalent of the project's keycheck.py for SHA-256 and CRC16.
// Hash input: b"PUF-KEY-CHECK-v1:" + K.to_bytes(4,"big") + np.packbits(key).
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace puf {
namespace detail {
inline uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
inline std::array<uint8_t,32> sha256(const std::vector<uint8_t>& data) {
    static constexpr uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    uint32_t h[8] = { 0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                      0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19 };
    std::vector<uint8_t> m(data);
    const uint64_t bit_count = static_cast<uint64_t>(m.size()) * 8;
    m.push_back(0x80);
    while ((m.size() % 64) != 56) m.push_back(0);
    for (int i=7;i>=0;--i) m.push_back(static_cast<uint8_t>(bit_count >> (i*8)));
    for (size_t off=0; off<m.size(); off+=64) {
        uint32_t w[64] = {};
        for (int i=0;i<16;++i) {
            const size_t j=off+static_cast<size_t>(i)*4;
            w[i] = (uint32_t(m[j])<<24) | (uint32_t(m[j+1])<<16) | (uint32_t(m[j+2])<<8) | uint32_t(m[j+3]);
        }
        for (int i=16;i<64;++i) {
            uint32_t a = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15]>>3);
            uint32_t b = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2]>>10);
            w[i] = w[i-16]+a+w[i-7]+b;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],z=h[7];
        for (int i=0;i<64;++i) {
            const uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            const uint32_t ch=(e&f)^(~e&g);
            const uint32_t t1=z+s1+ch+k[i]+w[i];
            const uint32_t s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            const uint32_t maj=(a&b)^(a&c)^(b&c);
            const uint32_t t2=s0+maj;
            z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;
        h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
    }
    std::array<uint8_t,32> out{};
    for (int i=0;i<8;++i)
        for (int j=0;j<4;++j) out[static_cast<size_t>(i)*4+j] = static_cast<uint8_t>(h[i] >> (24-8*j));
    return out;
}
} // detail

class NativeKeyChecker {
    std::string kind_;
    size_t k_;
    size_t digest_bytes_;
    std::vector<uint8_t> expected_;
    std::vector<std::array<uint8_t,16>> crc_rows_;
public:
    NativeKeyChecker(std::string kind, size_t k, const std::vector<uint8_t>& tag, size_t hash_bits = 128)
        :kind_(std::move(kind)),k_(k),digest_bytes_(hash_bits/8),expected_(tag) {
        if (kind_!="sha256" && kind_!="crc16")
            throw std::invalid_argument("Native verifier supports sha256 or crc16 (not other hash/CRC variants)");
        if (kind_=="sha256") {
            if (hash_bits==0 || hash_bits>256 || hash_bits%8!=0 || tag.size()!=digest_bytes_)
                throw std::invalid_argument("Invalid SHA-256 tag length");
        } else {
            if (tag.size()!=16) throw std::invalid_argument("CRC16 tag must contain 16 individual bits");
            // keycheck.py: build the same GF(2) generator matrix, not a generic CRC convention.
            std::array<uint8_t,17> poly{};
            for (int exponent : {16,12,5,0}) poly[static_cast<size_t>(16-exponent)] = 1;
            std::array<uint8_t,16> rem{};
            rem[0] = 1;
            crc_rows_.resize(k_);
            for (size_t i=0;i<k_;++i) {
                std::array<uint8_t,17> extended{};
                for (int j=0;j<16;++j) extended[j]=rem[j];
                if (extended[0]) for (int j=0;j<17;++j) extended[j]^=poly[j];
                for (int j=0;j<16;++j) rem[j]=extended[j+1];
                crc_rows_[k_-i-1]=rem;
            }
        }
    }
    std::vector<uint8_t> tag(const std::vector<uint8_t>& key) const {
        if (key.size()!=k_) throw std::invalid_argument("Unexpected key length");
        if (kind_=="crc16") {
            std::vector<uint8_t> result(16,0);
            for (size_t i=0;i<k_;++i)
                if (key[i]) for (int j=0;j<16;++j) result[j]^=crc_rows_[i][j];
            return result;
        }
        static constexpr char prefix[]="PUF-KEY-CHECK-v1:";
        std::vector<uint8_t> data(prefix,prefix+sizeof(prefix)-1);
        const uint32_t len=static_cast<uint32_t>(k_);
        for (int i=3;i>=0;--i) data.push_back(static_cast<uint8_t>(len>>(i*8)));
        for (size_t i=0;i<k_;i+=8) {
            uint8_t byte=0;
            for (size_t b=0;b<8 && i+b<k_;++b)
                byte|=static_cast<uint8_t>((key[i+b]&1u)<<(7-b));
            data.push_back(byte);
        }
        const auto digest=detail::sha256(data);
        return std::vector<uint8_t>(digest.begin(),digest.begin()+static_cast<std::ptrdiff_t>(digest_bytes_));
    }
    bool verify(const std::vector<uint8_t>& key) const { return tag(key)==expected_; }
};
} // namespace puf
