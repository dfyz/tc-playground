#include <algorithm>
#include <array>
#include <random>
#include <utility>

#include <cstdint>
#include <err.h>

#include <cuda.h>
#include <cuda_bf16.h>

#include <immintrin.h>

void CheckCu(CUresult res, const char* reason) {
    if (res != CUDA_SUCCESS) {
        const char* err;
        const auto err_res = cuGetErrorString(res, &err);
        errx(1, "failed to %s: %s", reason, err_res == CUDA_SUCCESS ? err : "N/A");
    }
}

using Rng = std::mt19937_64;
constexpr Rng::result_type kSeed = 31337;

using Vec16 = std::array<__nv_bfloat16, 16>;
using MatA  = std::array<Vec16, 64>;
using MatB  = std::array<Vec16, 256>;

void GenVec(Rng& rng, Vec16& out) {
    std::lognormal_distribution<float> gen{0.0f, 5.0f};
    for (size_t ii = 0; ii < out.size(); ++ii) {
        out[ii] = gen(rng);
    }
}

void PermuteVecPair(Rng& rng, Vec16& vec1, Vec16& vec2) {
    for (size_t ii = 1; ii < vec1.size(); ++ii) {
        std::uniform_int_distribution<size_t> gen{0, ii};
        const size_t jj = gen(rng);
        std::swap(vec1[ii], vec1[jj]);
        std::swap(vec2[ii], vec2[jj]);
    }
}

std::pair<MatA, MatB> GenInput() {
    Rng rng{kSeed};
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

float MulVecVecRef(const auto& vec_a, const auto& vec_b) {
    __m512bh a_pack = (__m512bh)_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)vec_a.data()));
    __m512bh b_pack = (__m512bh)_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)vec_b.data()));
    __m512 packed_prods = _mm512_dpbf16_ps(_mm512_setzero(), a_pack, b_pack);
    alignas(64) float prods[16];
    _mm512_store_ps(prods, packed_prods);
    float res = 0.0f;
    for (auto x : prods) {
        res += x;
    }
    return res;
}

int main() {
    const auto [mat_a, mat_b] = GenInput();

    for (size_t aa = 0; aa < mat_a.size(); ++aa) {
        for (size_t bb = 0; bb < mat_b.size(); ++bb) {
            const auto& vec_a = mat_a[aa];
            const auto& vec_b = mat_b[bb];
            const auto ref_res = MulVecVecRef(vec_a, vec_b);

            printf("A[%zu]*B[%zu] = %a (%1.8e) = ", aa, bb, ref_res, ref_res);
            for (size_t ii = 0; ii < vec_a.size(); ++ii) {
                const auto a_float = (float)vec_a[ii];
                const auto b_float = (float)vec_b[ii];
                printf("%s%g * %g", ii ? " + " : "", a_float, b_float);
            }
            printf("\n");
        }
    }

    CheckCu(cuInit(0), "initialize CUDA");

    CUdevice device;
    CheckCu(cuDeviceGet(&device, 0), "get CUDA device #0");

    CUcontext ctx;
    CheckCu(cuCtxCreate(&ctx, nullptr, 0, device), "get CUDA context");

    CUmodule module;
    CheckCu(cuModuleLoad(&module, "tc.cubin"), "load the compiled PTX");

    CUfunction run_tc;
    CheckCu(cuModuleGetFunction(&run_tc, module, "run_tc"), "find the run_tc kernel");

    std::array<void*, 3> args = {
        // FIXME: use real data
        nullptr, // A matrix
        nullptr, // B matrix
        nullptr, // output matrix
    };
    CheckCu(
        cuLaunchKernel(
            run_tc,
            1, 1, 1,   // grid dims
            128, 1, 1, // block dims
            0,         // don't need dynamic smem
            nullptr,   // use the default stream
            args.data(),
            nullptr   // nothing extra
        ),
        "launch the run_tc kernel"
    );

    CheckCu(cuCtxSynchronize(), "wait for run_tc completion");

    CheckCu(cuModuleUnload(module), "unload the compiled PTX");
    CheckCu(cuCtxDestroy(ctx), "tear down the CUDA context");
}
