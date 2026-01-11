#include "gemm_hopper_emu.h"

#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <array>
#include <limits>

#include <cuda_bf16.h>

namespace {

struct BF16Parts {
    explicit BF16Parts(__nv_bfloat16 number)
        : BF16Parts(((__nv_bfloat16_raw)number).x)
    {}

    explicit BF16Parts(uint16_t number)
        : raw_significand(number & 0x7F)
        , exponent((number >> 7) & 0xFF)
        , sign(number >> 15)
    {}

    uint8_t full_significand() const {
        // The MSB is implicitly 1 for normal numbers (exponent > 0).
        // For zero/subnormals, it's 0.
        return ((exponent == 0 ? 0 : 0x80) | raw_significand);
    }

    uint8_t raw_significand;
    uint8_t exponent;
    uint8_t sign;
};

struct Addend {
    uint32_t significand;
    uint8_t exponent;
    uint8_t sign;
};

}

float MulVecVecHopperEmu(const Vec& vec_a, const Vec& vec_b) {
    std::array<Addend, kK> addends;
    for (size_t ii = 0; ii < vec_a.size(); ++ii) {
        const BF16Parts a_parts{vec_a[ii]};
        const BF16Parts b_parts{vec_b[ii]};

        const auto addend_sign = (uint8_t)(a_parts.sign ^ b_parts.sign);

        const auto addend_exp = (uint8_t)std::clamp<int32_t>(
            (int32_t)a_parts.exponent + (int32_t)b_parts.exponent - 127,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max()
        );

        // Full significands have 8 bits, so the full product will have 16 bits.
        // The 32-bit addend has 5 most significant bits for intermediate carries,
        // so we have to move the MSB of the product (bit 15) to the first free bit.
        const auto addend_significand =
            ((uint32_t)a_parts.full_significand() * (uint32_t)b_parts.full_significand())
            << (31 - 5 - 15)
        ;

        addends[ii] = {
            .significand = addend_significand,
            .exponent = addend_exp,
            .sign = addend_sign,
        };
    }

    std::sort(addends.begin(), addends.end(), [](const auto& a, const auto& b) {
        return a.exponent > b.exponent;
    });

    // FIXME: return the actual value
    return 42.0f;
}