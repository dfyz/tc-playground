#include "common.h"
#include "gemm_avx512.h"
#include "gemm_hopper.h"
#include "gemm_hopper_emu.h"

#include <algorithm>
#include <random>
#include <tuple>
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

void GenVec(Rng& rng, Vec& out) {
#if SUBNORMALS
    std::normal_distribution<float> gen{0.0f, 1e-19};
#else
    std::lognormal_distribution<float> gen{0.0f, 2.0f};
#endif
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

std::tuple<MatA, MatB, float> GenInput(Rng::result_type seed) {
    Rng rng{seed};
    MatA res_a;
    MatB res_b;

    // First 64 vectors of both matrices are permutations of the first vector.
    GenVec(rng, res_a[0]);
    GenVec(rng, res_b[0]);
    for (size_t ii = 1; ii < res_a.size(); ++ii) {
        res_a[ii] = res_a[0];
        res_b[ii] = res_b[0];

        PermuteVecPair(rng, res_a[ii], res_b[ii]);
    }

    // The remaining vectors of B are just random vectors.
    for (size_t ii = res_a.size(); ii < res_b.size(); ++ii) {
        GenVec(rng, res_b[ii]);
    }

    std::lognormal_distribution<float> cc_gen{0.0f, 2.0f};
    // FIXME: make the emulator work with non-zero C.
    return std::make_tuple(res_a, res_b, /*GenSign(rng) * cc_gen(rng)*/0.0f);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        errx(1, "usage: %s SEED VERBOSE", argv[0]);
    }
    const auto seed = std::stoull(argv[1]);
    const auto is_verbose = std::stoull(argv[2]);
    const auto [mat_a, mat_b, cc] = GenInput(seed);
    const auto hopper_out = MulMatMatHopper(cc, mat_a, mat_b);

    for (size_t aa = 0; aa < mat_a.size(); ++aa) {
        for (size_t bb = 0; bb < mat_b.size(); ++bb) {
            const auto& vec_a = mat_a[aa];
            const auto& vec_b = mat_b[bb];
            const auto avx512_res = MulVecVecAvx512(cc, vec_a, vec_b);
            const auto hopper_res = hopper_out[aa][bb];
            const auto hopper_emu_res = MulVecVecHopperEmu(cc, vec_a, vec_b);
            printf(
                "A[%zu]*B[%zu]: AVX512 = %a (%1.8e), HOPPER = %a (%1.8e), HOPPER EMULATION = %a (%1.8e)\n",
                aa, bb,
                avx512_res, avx512_res,
                hopper_res, hopper_res,
                hopper_emu_res, hopper_emu_res
            );

            if (is_verbose != 0) {
                printf("A_hex = {");
                for (size_t ii = 0; ii < vec_a.size(); ++ii) {
                    printf("%s'%04hX'", (ii ? ", " : ""), ((__nv_bfloat16_raw)vec_a[ii]).x);
                }
                printf("}\n");
                printf("B_hex = {");
                for (size_t ii = 0; ii < vec_b.size(); ++ii) {
                    printf("%s'%04hX'", (ii ? ", " : ""), ((__nv_bfloat16_raw)vec_b[ii]).x);
                }
                printf("}\n");
                uint32_t c_int;
                memcpy(&c_int, &cc, sizeof(uint32_t));
                printf("C_hex = '%08X'\n", c_int);
            }

            if (!std::isnan(hopper_res) && hopper_res != hopper_emu_res) {
                errx(1, "detected a mismatch between the device output and its emulation");
            }
        }
    }
}
