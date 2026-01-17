#include "gemm_hopper_emu.h"

#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

#include <cuda_bf16.h>

#define DEBUG 0

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
    int32_t exponent;
    uint8_t sign;
};

}

#if DEBUG
void PrintBinarySignficand(const Addend& x) {
    printf("%d -> ", x.exponent - 127);
    for (ssize_t ii = 31; ii >= 0; --ii) {
        if (ii == 24 || ii == 26 || ii == 1) {
            printf("|");
        }
        printf("%c", (x.significand & (1 << ii)) ? '1' : '0');
    }
    printf("\n");
}
#endif

float MulVecVecHopperEmu(const Vec& vec_a, const Vec& vec_b) {
    constexpr uint32_t kCarryBits = 5;
    constexpr uint32_t kExtraSignificandBits = 2;
    constexpr int32_t kMinExponent = 0;
    constexpr int32_t kMaxExponent = 255;

    std::array<Addend, kK> addends;
    for (size_t ii = 0; ii < vec_a.size(); ++ii) {
        const BF16Parts a_parts{vec_a[ii]};
        const BF16Parts b_parts{vec_b[ii]};

        const auto addend_sign = (uint8_t)(a_parts.sign ^ b_parts.sign);
        const auto addend_exp = (int32_t)a_parts.exponent + (int32_t)b_parts.exponent - 127;

        // Full significands have 8 bits, so the full product will have 16 bits.
        // The 32-bit addend has kCarryBits most significant bits for intermediate carries,
        // so we have to move the MSB of the product (bit 15) to the first free bit.
        const auto addend_significand =
            ((uint32_t)a_parts.full_significand() * (uint32_t)b_parts.full_significand())
            << (31 - kCarryBits - 15)
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

    Addend result{addends[0]};
#if DEBUG
    PrintBinarySignficand(result);
#endif
    for (size_t ii = 1; ii < addends.size(); ++ii) {
        // The below follows the section 7.3 from "Handbook of Floating-Point Arithmetic"
        //
        // We know that the exponent of `other` is not greater than the exponent
        // of the result, so we can immediately align the significand of `other`.
        auto& other = addends[ii];
        other.significand >>= std::min(31, result.exponent - other.exponent);
#if DEBUG
        PrintBinarySignficand(other);
#endif
        if (result.sign == other.sign) {
            // Perform an addition. This can't make the result negative, and won't overflow,
            // since we have enough most significant bits to store the carries.
            result.significand += other.significand;
        } else {
            // Perform a subtraction.
            if (result.significand >= other.significand) {
                result.significand -= other.significand;
            } else {
                // If the result is negative, we should swap the order
                // of the subtracion and the sign of the result.
                result.significand = other.significand - result.significand;
                result.sign ^= 1;
            }
        }
    }

#if DEBUG
    printf("===\n");
    PrintBinarySignficand(result);
#endif

    while ((result.significand & 0xfc'00'00'00u) != 0) {
        if (result.exponent < kMaxExponent) {
            ++result.exponent;
            result.significand >>= 1;
        } else {
            throw std::logic_error("overflow");
        }
    }
    while ((result.significand & 0x02'00'00'00u) == 0) {
        if (result.exponent > 1) {
            --result.exponent;
            result.significand <<= 1;
        } else {
            result.exponent = 0;
            break;
        }
    }

    // Compose the final result out of the individual components.
    return std::bit_cast<float, uint32_t>(
        (result.sign << 31) |
        (result.exponent << 23) |
        // Get rid of the extra bits and the implicit one, if any.
        ((result.significand >> kExtraSignificandBits) & 0x7F'FFFF)
    );
}