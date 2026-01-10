#include "common.h"
#include "gemm_avx512.h"
#include "gemm_hopper.h"

#include <algorithm>
#include <random>
#include <utility>

#include <cstdint>
#include <cstdio>
#include <err.h>

#include <cuda_bf16.h>

using Rng = std::mt19937_64;

void GenVec(Rng& rng, Vec& out) {
    std::lognormal_distribution<float> gen{0.0f, 5.0f};
    for (size_t ii = 0; ii < out.size(); ++ii) {
        out[ii] = gen(rng);
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

std::pair<MatA, MatB> GenInput(Rng::result_type seed) {
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

    return std::make_pair(res_a, res_b);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        errx(1, "usage: %s SEED VERBOSE", argv[0]);
    }
    const auto seed = std::stoull(argv[1]);
    const auto is_verbose = std::stoull(argv[2]);
    const auto [mat_a, mat_b] = GenInput(seed);
    const auto hopper_out = MulMatMatHopper(mat_a, mat_b);

    for (size_t aa = 0; aa < mat_a.size(); ++aa) {
        for (size_t bb = 0; bb < mat_b.size(); ++bb) {
            const auto& vec_a = mat_a[aa];
            const auto& vec_b = mat_b[bb];
            const auto avx512_res = MulVecVecAvx512(vec_a, vec_b);
            const auto hopper_res = (*hopper_out)(aa, bb);
            printf(
                "A[%zu]*B[%zu]: AVX512 = %a (%1.8e), HOPPER = %a (%1.8e)\n",
                aa, bb,
                avx512_res, avx512_res,
                hopper_res, hopper_res
            );
            if (is_verbose != 0) {
                printf("\tINPUTS = ");
                for (size_t ii = 0; ii < vec_a.size(); ++ii) {
                    const auto a_float = (float)vec_a[ii];
                    const auto b_float = (float)vec_b[ii];
                    printf("%s%g * %g", ii ? " + " : "", a_float, b_float);
                }
                printf("\n");
            }
        }
    }
}
