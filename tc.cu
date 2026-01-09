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

constexpr size_t kK = 16;
constexpr size_t kM = 64;
constexpr size_t kN = 256;

using Vec = std::array<__nv_bfloat16, kK>;
using MatA  = std::array<Vec, kM>;
using MatB  = std::array<Vec, kN>;

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

CUdeviceptr GetDevicePointer(void* host_pointer) {
    CUdeviceptr res;
    CheckCu(cuMemHostGetDevicePointer(&res, host_pointer, 0), "get the device pointer");
    return res;
}

class SmemInput {
public:
    SmemInput(size_t rows)
        // Without swizzling, we skip exactly `rows` 128-bit elements.
        : lbo_(rows * 16)
    {
        CheckCu(cuMemHostAlloc(
            (void**)&data_, rows * kK * sizeof(__nv_bfloat16), CU_MEMHOSTALLOC_DEVICEMAP),
            "allocate shared memory input"
        );
    }

    ~SmemInput() {
        CheckCu(cuMemFreeHost(data_), "free shared memory input");
    }

    __nv_bfloat16& operator()(size_t row, size_t col) {
        const size_t off =
            row * 8 +
            (col / 8) * (lbo_ / sizeof(__nv_bfloat16)) +
            (col % 8);
        ;
        return *(data_ + off);
    }

    template <size_t Rows>
    void CopyFrom(const std::array<Vec, Rows>& mat) {
        for (size_t row = 0; row < mat.size(); ++row) {
            const auto& vec = mat[row];
            for (size_t col = 0; col < vec.size(); ++col) {
                (*this)(row, col) = vec[col];
            }
        }
    }

    CUdeviceptr device_data() const {
        return GetDevicePointer(data_);
    }

    uint64_t desc() const {
        return ((lbo_ >> 4) << 16) | ((sbo_ >> 4) << 32);
    }

private:
    // 9.7.15.5.1.2.1.3: "T = 128 / sizeof-elements-in-bits [...]
    // represents scale factor which normalizes matrix element types to 128-bits."
    const size_t t_ = 128 / sizeof(__nv_bfloat16);
    // 9.7.15.5.1.9:
    // "The offset from the first 8 rows to the next 8 rows"
    // Without swizzling, it's always 128 bytes (8 128-bit elements).
    const size_t sbo_ = 128;
    // 9.7.15.5.1.8: "the offset from the first column
    // to the second columns [sic] of the 8x2 tile
    // in the 128-bit element type normalized matrix"
    const size_t lbo_;

    __nv_bfloat16* data_;
};

class GemmOutput {
public:
    GemmOutput(size_t rows, size_t cols)
        : rows_(rows)
        , cols_(cols)
    {
        CheckCu(
            cuMemHostAlloc((void**)&data_, rows * cols * sizeof(float), CU_MEMHOSTALLOC_DEVICEMAP),
            "allocate global memory output"
        );
    }

    ~GemmOutput() {
        CheckCu(cuMemFreeHost(data_), "free global memory output");
    }

    float operator()(size_t row, size_t col) const {
        // 9.7.15.5.1.1.1, "Accumulator D" with `.f32`
        const size_t tid =
            32 * (row / 16) +
            4 * (row % 8) +
            (col % 8) / 2
        ;

        const size_t off =
            4 * (col / 8) +
            2 * ((row / 8) % 2) +
            col % 2
        ;

        return data_[(kN / 2) * tid + off];
    }

    CUdeviceptr device_data() const {
        return GetDevicePointer(data_);
    }

private:
    size_t rows_;
    size_t cols_;
    float* data_;
};

void ComputeDeviceOut(const MatA& mat_a, const MatB& mat_b, GemmOutput& device_out) {
    CUmodule module;
    CheckCu(cuModuleLoad(&module, "tc.cubin"), "load the compiled PTX");

    CUfunction run_tc;
    CheckCu(cuModuleGetFunction(&run_tc, module, "run_tc"), "find the run_tc kernel");

    SmemInput smem_a(kM);
    SmemInput smem_b(kN);

    smem_a.CopyFrom(mat_a);
    smem_b.CopyFrom(mat_b);

    auto a_device_data = smem_a.device_data();
    auto b_device_data = smem_b.device_data();
    auto a_desc = smem_a.desc();
    auto b_desc = smem_b.desc();
    auto out_device_data = device_out.device_data();

    std::array<void*, 5> args = {
        &a_device_data, &b_device_data,
        &a_desc, &b_desc,
        &out_device_data,
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
}

class CuContext {
public:
    CuContext() {
        CheckCu(cuDeviceGet(&device_, 0), "get CUDA device #0");
        CheckCu(cuCtxCreate(&ctx_, nullptr, 0, device_), "get CUDA context");
    }

    ~CuContext() {
        CheckCu(cuCtxDestroy(ctx_), "tear down the CUDA context");
    }

private:
    CUdevice device_;
    CUcontext ctx_;
};

int main() {
    CheckCu(cuInit(0), "initialize CUDA");
    CuContext ctx;
    const auto [mat_a, mat_b] = GenInput();

    GemmOutput device_out{kM, kN};
    ComputeDeviceOut(mat_a, mat_b, device_out);

    for (size_t aa = 0; aa < mat_a.size(); ++aa) {
        for (size_t bb = 0; bb < mat_b.size(); ++bb) {
            const auto& vec_a = mat_a[aa];
            const auto& vec_b = mat_b[bb];
            const auto host_res = MulVecVecRef(vec_a, vec_b);
            const auto device_res = device_out(aa, bb);
            printf(
                "A[%zu]*B[%zu] -> %a (%1.8e) HOST vs. %a (%1.8e) DEVICE\n",
                aa, bb,
                host_res, host_res,
                device_res, device_res
            );
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
