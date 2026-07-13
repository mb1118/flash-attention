// 3rd-gen TensorCore: warp-level mma.sync (Ampere sm_80+, portable).
// Single warp computes D[16x8] = A[16x16] * B[16x8] in fp16 with fp32 accumulate.
// Compiled for both sm_80 and sm_120a to test portability of the instruction.
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define M 16
#define N 8
#define K 16

__global__ void mma_kernel(const half* A, const half* B, float* D) {
    // A: row-major [M][K]; B: logical [K][N] (row-major storage B[k*N+n]).
    int lane = threadIdx.x;          // 0..31, one warp
    int g = lane >> 2;               // groupID 0..7
    int t = lane & 3;                // threadID_in_group 0..3

    auto Ae = [&](int r, int c) { return A[r * K + c]; };
    auto Be = [&](int r, int c) { return B[r * N + c]; };

    half2 a0 = __halves2half2(Ae(g,   2*t),   Ae(g,   2*t+1));
    half2 a1 = __halves2half2(Ae(g+8, 2*t),   Ae(g+8, 2*t+1));
    half2 a2 = __halves2half2(Ae(g,   2*t+8), Ae(g,   2*t+9));
    half2 a3 = __halves2half2(Ae(g+8, 2*t+8), Ae(g+8, 2*t+9));
    half2 b0 = __halves2half2(Be(2*t,   g),   Be(2*t+1, g));
    half2 b1 = __halves2half2(Be(2*t+8, g),   Be(2*t+9, g));

    uint32_t ra0 = *reinterpret_cast<uint32_t*>(&a0);
    uint32_t ra1 = *reinterpret_cast<uint32_t*>(&a1);
    uint32_t ra2 = *reinterpret_cast<uint32_t*>(&a2);
    uint32_t ra3 = *reinterpret_cast<uint32_t*>(&a3);
    uint32_t rb0 = *reinterpret_cast<uint32_t*>(&b0);
    uint32_t rb1 = *reinterpret_cast<uint32_t*>(&b1);

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));

    D[(g)   * N + (2*t)]   = c0;
    D[(g)   * N + (2*t+1)] = c1;
    D[(g+8) * N + (2*t)]   = c2;
    D[(g+8) * N + (2*t+1)] = c3;
}

int main() {
    half hA[M*K], hB[K*N];
    float hD[M*N], ref[M*N];
    for (int i = 0; i < M*K; i++) hA[i] = __float2half((float)((i % 7) - 3) * 0.5f);
    for (int i = 0; i < K*N; i++) hB[i] = __float2half((float)((i % 5) - 2) * 0.25f);
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += __half2float(hA[m*K+k]) * __half2float(hB[k*N+n]);
            ref[m*N+n] = acc;
        }

    half *dA, *dB; float *dD;
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB)); cudaMalloc(&dD, sizeof(hD));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    mma_kernel<<<1, 32>>>(dA, dB, dD);
    cudaError_t launch = cudaGetLastError();
    cudaError_t sync = cudaDeviceSynchronize();
    if (launch != cudaSuccess || sync != cudaSuccess) {
        printf("LAUNCH_FAIL: %s / %s\n",
               cudaGetErrorString(launch), cudaGetErrorString(sync));
        return 2;
    }
    cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);

    float max_err = 0.f;
    for (int i = 0; i < M*N; i++) {
        float e = fabsf(hD[i] - ref[i]);
        if (e > max_err) max_err = e;
    }
    if (max_err < 1e-2f) {
        printf("RUN_OK CORRECT max_err=%.4g\n", max_err);
        return 0;
    } else {
        printf("RUN_OK WRONG max_err=%.4g (D[0]=%.3f ref[0]=%.3f)\n", max_err, hD[0], ref[0]);
        return 1;
    }
}
