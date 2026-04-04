#include "gemm_hopper_emu.h"

#include <cstdint>
#include <cstdio>

#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <stdexcept>

#include <cuda_bf16.h>

namespace {

constexpr uint32_t kCarryBits = 5;
constexpr uint32_t kExtraSignificandBits = 2;

constexpr int32_t kMinExponent = -126;
constexpr int32_t kMaxExponent = 127;

constexpr uint32_t kBF16SignificandBits = 7;
constexpr uint32_t kFloatSignificandBits = 23;

int8_t GetUnbiasedExponent(int32_t biased_exponent) {
    return std::max(kMinExponent, biased_exponent - kMaxExponent);
}

uint8_t GetBiasedExponent(uint32_t number, uint32_t significand_bits) {
    return (number >> significand_bits) & 0xFF;
}

uint32_t GetFullSignificand(uint32_t number, uint32_t significand_bits) {
    const auto raw_significand = number & ((1L << significand_bits) - 1);
    // The MSB is implicitly 1 for normal numbers (exponent > 0).
    // For zero/subnormals, it's 0.
    const auto biased_exp = GetBiasedExponent(number, significand_bits);
    return (biased_exp == 0 ? 0 : (1U << significand_bits)) | raw_significand;
}

struct BF16Parts {
    explicit BF16Parts(__nv_bfloat16 number)
        : BF16Parts(((__nv_bfloat16_raw)number).x)
    {}

    explicit BF16Parts(uint16_t number)
        : full_significand(
            GetFullSignificand(number, kBF16SignificandBits)
        )
        , unbiased_exponent(
            GetUnbiasedExponent(
                GetBiasedExponent(number, kBF16SignificandBits)
            )
        )
        , sign(number >> 15)
    {
    }

    uint8_t full_significand;
    int8_t unbiased_exponent;
    uint8_t sign;
};

struct Addend {
    uint32_t full_significand;
    int32_t unbiased_exponent;
    uint8_t sign;

    void Align(int32_t max_exp) {
        full_significand >>= std::min(31, max_exp - unbiased_exponent);
    }
};

Addend FloatToAddend(float float_number) {
    const auto number = std::bit_cast<uint32_t>(float_number);
    return Addend {
        .full_significand =
            GetFullSignificand(number, kFloatSignificandBits) << kExtraSignificandBits,
        .unbiased_exponent = GetUnbiasedExponent(
            GetBiasedExponent(number, kFloatSignificandBits)
        ),
        .sign = (uint8_t)(number >> 31),
    };
}

}

#ifdef DEBUG
void PrintBinarySignficand(const Addend& x) {
    printf("%d -> %c", x.unbiased_exponent, x.sign ? '-' : '+');
    for (ssize_t ii = 31; ii >= 0; --ii) {
        if (ii == 24 || ii == 26 || ii == 1) {
            printf("|");
        }
        printf("%c", (x.full_significand & (1 << ii)) ? '1' : '0');
    }
    printf("\n");
}
#endif

float MulVecVecHopperEmu(float c, const Vec& vec_a, const Vec& vec_b) {
    std::array<Addend, kK> addends;
    addends[0] = FloatToAddend(c);
    // https://github.com/north-numerical-computing/MATLAB-tensor-core/blob/1c2522d4e248f46b426638b0af7d13beb1563ef9/models/tools/Generic_BFMA_TC.m#L418-L421
    auto max_exp = -133;
    for (size_t ii = 0; ii < vec_a.size(); ++ii) {
        const BF16Parts a_parts{vec_a[ii]};
        const BF16Parts b_parts{vec_b[ii]};

        const auto addend_sign = (uint8_t)(a_parts.sign ^ b_parts.sign);
        const auto addend_exp = (int32_t)a_parts.unbiased_exponent + (int32_t)b_parts.unbiased_exponent;

        // Full significands have 8 bits, so the full product will have 16 bits.
        // The 32-bit addend has kCarryBits most significant bits for intermediate carries,
        // so we have to move the MSB of the product (bit 15) to the first free bit.
        const auto addend_significand =
            ((uint32_t)a_parts.full_significand * (uint32_t)b_parts.full_significand)
            << (31 - kCarryBits - 15)
        ;

        max_exp = std::max(max_exp, addend_exp);
        addends[ii] = {
            .full_significand = addend_significand,
            .unbiased_exponent = addend_exp,
            .sign = addend_sign,
        };
    }

    auto result = FloatToAddend(c);
    max_exp = std::max(max_exp, result.unbiased_exponent);
    result.Align(max_exp);
    for (size_t ii = 0; ii < addends.size(); ++ii) {
        // The below follows the section 7.3 from "Handbook of Floating-Point Arithmetic"
        //
        // We know that the exponent of `other` is not greater than the exponent
        // of the result, so we can immediately align the significand of `other`.
        auto& other = addends[ii];
        other.Align(max_exp);
#ifdef DEBUG
        PrintBinarySignficand(other);
#endif
        if (result.sign == other.sign) {
            // Perform an addition. This can't make the result negative, and won't overflow,
            // since we have enough most significant bits to store the carries.
            result.full_significand += other.full_significand;
        } else {
            // Perform a subtraction.
            if (result.full_significand >= other.full_significand) {
                result.full_significand -= other.full_significand;
            } else {
                // If the result is negative, we should swap the order
                // of the subtracion and the sign of the result.
                result.full_significand = other.full_significand - result.full_significand;
                result.sign ^= 1;
            }
        }
    }

    result.unbiased_exponent = max_exp;

#ifdef DEBUG
    printf("===\n");
    PrintBinarySignficand(result);
#endif

    while (result.unbiased_exponent < kMinExponent) {
        ++result.unbiased_exponent;
        result.full_significand >>= 1;
    }

    while (result.unbiased_exponent > kMaxExponent) {
        --result.unbiased_exponent;
        result.full_significand <<= 1;
    }

#ifdef DEBUG
    printf("exp = %d\n", result.unbiased_exponent);
    PrintBinarySignficand(result);
#endif

    while ((result.full_significand & 0xfc'00'00'00u) != 0) {
        if (result.unbiased_exponent < kMaxExponent) {
            ++result.unbiased_exponent;
            result.full_significand >>= 1;
        } else {
            throw std::logic_error("overflow");
        }
    }
    while ((result.full_significand & 0x02'00'00'00u) == 0) {
        if (result.unbiased_exponent > kMinExponent) {
            --result.unbiased_exponent;
            result.full_significand <<= 1;
        } else {
            result.unbiased_exponent = kMinExponent - 1;
            break;
        }
    }

    // Compose the final result out of the individual components.
    return std::bit_cast<float>(
        (result.sign << 31) |
        ((result.unbiased_exponent + kMaxExponent) << 23) |
        // Get rid of the extra bits and the implicit one, if any.
        ((result.full_significand >> kExtraSignificandBits) & 0x7F'FFFF)
    );
}