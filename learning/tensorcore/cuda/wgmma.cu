// 4th-gen TensorCore: Hopper warpgroup MMA (wgmma.mma_async), sm_90a only.
// Minimal kernel that EMITS a real wgmma instruction (operands from shared memory).
// Goal is a feasibility probe, not a numerically-correct GEMM:
//   - does it compile for sm_90a?
//   - does ptxas accept it for sm_120a? (expected: no)
//   - does the sm_90a build launch on this SM120 GPU? (expected: no)
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Build a GMMA shared-memory matrix descriptor (swizzle=0, no-swizzle layout).
__device__ __forceinline__ uint64_t make_desc(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr & 0x3FFFF) >> 4);   // start address >> 4
    desc |= (uint64_t)((16u >> 4)) << 16;        // leading byte offset >> 4
    desc |= (uint64_t)((16u >> 4)) << 32;        // stride byte offset >> 4
    return desc;
}

// m64n8k16 f16*f16 -> f32. 128 threads = 1 warpgroup.
__global__ void wgmma_kernel(const half* A, const half* B, float* D) {
    __shared__ half sA[64 * 16];
    __shared__ half sB[16 * 8];
    int tid = threadIdx.x;
    for (int i = tid; i < 64 * 16; i += blockDim.x) sA[i] = A[i];
    for (int i = tid; i < 16 * 8;  i += blockDim.x) sB[i] = B[i];
    __syncthreads();

    uint64_t descA = make_desc(sA);
    uint64_t descB = make_desc(sB);

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 "
        "{%0,%1,%2,%3}, %4, %5, 1, 1, 1, 0, 0;\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "l"(descA), "l"(descB));
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    // Each of 128 threads writes its 4 accumulators.
    D[tid * 4 + 0] = d0; D[tid * 4 + 1] = d1;
    D[tid * 4 + 2] = d2; D[tid * 4 + 3] = d3;
}

int main() {
    const int MA = 64 * 16, MB = 16 * 8, MD = 128 * 4;
    half hA[MA], hB[MB]; float hD[MD];
    for (int i = 0; i < MA; i++) hA[i] = __float2half(0.1f);
    for (int i = 0; i < MB; i++) hB[i] = __float2half(0.1f);

    half *dA, *dB; float *dD;
    cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB)); cudaMalloc(&dD, sizeof(hD));
    cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);

    wgmma_kernel<<<1, 128>>>(dA, dB, dD);
    cudaError_t launch = cudaGetLastError();
    cudaError_t sync = cudaDeviceSynchronize();
    if (launch != cudaSuccess || sync != cudaSuccess) {
        printf("LAUNCH_FAIL: %s / %s\n",
               cudaGetErrorString(launch), cudaGetErrorString(sync));
        return 2;
    }
    printf("RUN_OK (wgmma executed; numerical correctness not checked)\n");
    return 0;
}
