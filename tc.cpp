#include "common.h"
#include "gemm_avx512.h"
#include "gemm_hopper.h"
#include "gemm_hopper_emu.h"
#include "gemm_hopper_emu_2.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <random>
#include <tuple>
#include <stdexcept>
#include <utility>

#include <cstdint>
#include <cstdio>
#include <err.h>

#include <cuda_bf16.h>

using Rng = std::mt19937_64;

#define SUBNORMALS 0

float GenSign(Rng& rng) {
    std::bernoulli_distribution sign;
    return sign(rng) ? 1.0f : -1.0f;
}

template <typename Gen>
void GenVec(Rng& rng, Gen& gen, Vec& out) {
    for (size_t ii = 0; ii < out.size(); ++ii) {
        out[ii] = GenSign(rng) * gen(rng);
    }
}

void PermuteVecPair(Rng& rng, Vec& vec1, Vec& vec2) {
    for (size_t ii = 1; ii < vec1.size(); ++ii) {
        std::uniform_int_distribution<size_t> gen{0, ii};
        const size_t jj = gen(rng);
        std::swap(vec1[ii], vec1[jj]);
        std::swap(vec2[ii], vec2[jj]);
    }
}

template <typename Gen>
std::tuple<MatA, MatB, float> GenInput(Rng& rng, Gen& gen, bool accumulate) {
    MatA res_a;
    MatB res_b;

    // First 64 vectors of both matrices are permutations of the first vector.
    GenVec(rng, gen, res_a[0]);
    GenVec(rng, gen, res_b[0]);
    for (size_t ii = 1; ii < res_a.size(); ++ii) {
        res_a[ii] = res_a[0];
        res_b[ii] = res_b[0];

        PermuteVecPair(rng, res_a[ii], res_b[ii]);
    }

    // The remaining vectors of B are just random vectors.
    for (size_t ii = res_a.size(); ii < res_b.size(); ++ii) {
        GenVec(rng, gen, res_b[ii]);
    }

    return std::make_tuple(res_a, res_b, accumulate ? GenSign(rng) * gen(rng) : 0.0f);
}

std::tuple<MatA, MatB, float> GenInput(Rng::result_type seed, uint32_t mode, bool accumulate) {
    Rng rng{seed};
    switch (mode) {
    case 0:
        {
            std::lognormal_distribution<float> gen{0.0f, 2.0f};
            return GenInput(rng, gen, accumulate);
        }
    case 1:
        {
            std::normal_distribution<float> gen{0.0f, 1e-19f};
            return GenInput(rng, gen, accumulate);
        }
    case 2:
        {
            std::normal_distribution<float> gen{0.0f, 1e-20f};
            return GenInput(rng, gen, accumulate);
        }
    case 3:
        {
            std::normal_distribution<float> gen{0.0f, 1e-21f};
            return GenInput(rng, gen, accumulate);
        }
    case 4:
        {
            std::normal_distribution<float> gen{0.0f, 1e-35f};
            return GenInput(rng, gen, accumulate);
        }
    case 5:
        {
            std::uniform_int_distribution gen{std::numeric_limits<int>::min()};
            MatA res_a;
            MatB res_b;
            for (size_t ii = 0; ii < res_b.size(); ++ii) {
                if (ii < res_a.size()) {
                    auto& vec_a = res_a[ii];
                    for (size_t jj = 0; jj < vec_a.size(); ++jj) {
                        vec_a[jj] = std::bit_cast<float>(gen(rng));
                    }
                }
                auto& vec_b = res_b[ii];
                for (size_t jj = 0; jj < vec_b.size(); ++jj) {
                    vec_b[jj] = std::bit_cast<float>(gen(rng));
                }
            }
            return std::make_tuple(res_a, res_b, accumulate ? std::bit_cast<float>(gen(rng)) : 0.0f);
        }
    default:
        throw std::runtime_error("Unknown mode");
    }
}

void CheckRandomInput(Rng::result_type seed, bool is_verbose, uint32_t mode, bool accumulate) {
    const auto [mat_a, mat_b, cc] = GenInput(seed, mode, accumulate);
    const auto hopper_out = MulMatMatHopper(cc, mat_a, mat_b);

    for (size_t aa = 0; aa < mat_a.size(); ++aa) {
        for (size_t bb = 0; bb < mat_b.size(); ++bb) {
            const auto& vec_a = mat_a[aa];
            const auto& vec_b = mat_b[bb];
            const auto avx512_res = MulVecVecAvx512(cc, vec_a, vec_b);

            if (!std::isfinite(avx512_res)) {
                continue;
            }

            const auto hopper_res = hopper_out[aa][bb];
            const auto hopper_emu_res = MulVecVecHopperEmu(cc, vec_a, vec_b);
            const auto hopper_emu_res_2 = MulVecVecHopperEmu2(cc, (const uint16_t*)vec_a.data(), (const uint16_t*)vec_b.data());
            printf(
                "A[%zu]*B[%zu]: AVX512 = %a (%1.8e), HOPPER = %a (%1.8e)\nHOPPER EMULATION   = %a (%1.8e)\nHOPPER EMULATION 2 = %a (%1.8e)\n",
                aa, bb,
                avx512_res, avx512_res,
                hopper_res, hopper_res,
                hopper_emu_res, hopper_emu_res,
                hopper_emu_res_2, hopper_emu_res_2
            );

            if (is_verbose != 0) {
                printf("A_hex = [");
                for (size_t ii = 0; ii < vec_a.size(); ++ii) {
                    printf("%s0x%04hX", (ii ? ", " : ""), ((__nv_bfloat16_raw)vec_a[ii]).x);
                }
                printf("]\n");
                printf("B_hex = [");
                for (size_t ii = 0; ii < vec_b.size(); ++ii) {
                    printf("%s0x%04hX", (ii ? ", " : ""), ((__nv_bfloat16_raw)vec_b[ii]).x);
                }
                printf("]\n");
                printf("C_hex = 0x%08X\n", std::bit_cast<uint32_t>(cc));
            }

            if (!std::isnan(hopper_res) && hopper_res != hopper_emu_res) {
                errx(1, "detected a mismatch between the device output and its emulation");
            }
            if (hopper_emu_res != hopper_emu_res_2) {
                errx(1, "detected a mismatch between two different emulation types");
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        errx(1, "usage: %s SEED VERBOSE", argv[0]);
    }
    const auto seed = std::stoull(argv[1]);
    const auto is_verbose = std::stoull(argv[2]);

    for (uint32_t acc = 0; acc < 2; ++acc) {
        for (uint32_t mode = 0; mode < 6; ++mode) {
            printf("MODE = %d, ACCUMULATE = %d\n", mode, acc);
            CheckRandomInput(seed, is_verbose, mode, (bool)acc);
        }
    }
}
