// 5th-gen TensorCore: Blackwell datacenter UMMA (tcgen05.mma) with Tensor Memory,
// sm_100a only. Minimal kernel that EMITS tcgen05.alloc + tcgen05.mma.
// Feasibility probe, not a correct GEMM:
//   - does it compile for sm_100a?
//   - does ptxas accept it for sm_120a? (expected: no)
//   - does the sm_100a build launch on this SM120 GPU? (expected: no)
#include <cstdio>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__device__ __forceinline__ uint64_t make_desc(const void* smem_ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (uint64_t)((addr & 0x3FFFF) >> 4);
    desc |= (uint64_t)((16u >> 4)) << 16;
    desc |= (uint64_t)((16u >> 4)) << 32;
    return desc;
}

__global__ void tcgen05_kernel(const half* A, const half* B, float* D) {
    __shared__ half sA[64 * 16];
    __shared__ half sB[16 * 64];
    __shared__ uint32_t tmem_slot;   // tcgen05.alloc writes the TMEM base address here

    int tid = threadIdx.x;
    for (int i = tid; i < 64 * 16; i += blockDim.x) sA[i] = A[i % (64 * 16)];
    for (int i = tid; i < 16 * 64; i += blockDim.x) sB[i] = B[i % (16 * 64)];
    __syncthreads();

    // One warp allocates 32 columns of Tensor Memory.
    if (tid < 32) {
        uint32_t slot_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&tmem_slot));
        asm volatile(
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
            :: "r"(slot_addr), "r"(32u) : "memory");
    }
    __syncthreads();

    uint32_t d_tmem = tmem_slot;
    uint64_t a_desc = make_desc(sA);
    uint64_t b_desc = make_desc(sB);
    uint32_t idesc  = 0;   // instruction descriptor (dummy; not executed on this GPU)

    if (tid == 0) {
        asm volatile(
            "{\n\t .reg .pred PRED_enable_input_d;\n\t"
            "setp.ne.b32 PRED_enable_input_d, %4, 0;\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, PRED_enable_input_d;\n\t"
            "}\n"
            :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(0u) : "memory");
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
                     :: "l"((uint64_t)0) : "memory");
    }
    if (tid == 0) D[0] = 1.0f;   // marker: kernel body reached
}

int main() {
    const int MA = 64 * 16, MB = 16 * 64, MD = 64 * 64;
    half *hA = new half[MA], *hB = new half[MB];
    for (int i = 0; i < MA; i++) hA[i] = __float2half(0.1f);
    for (int i = 0; i < MB; i++) hB[i] = __float2half(0.1f);

    half *dA, *dB; float *dD;
    cudaMalloc(&dA, MA * sizeof(half)); cudaMalloc(&dB, MB * sizeof(half));
    cudaMalloc(&dD, MD * sizeof(float));
    cudaMemcpy(dA, hA, MA * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, MB * sizeof(half), cudaMemcpyHostToDevice);

    tcgen05_kernel<<<1, 128>>>(dA, dB, dD);
    cudaError_t launch = cudaGetLastError();
    cudaError_t sync = cudaDeviceSynchronize();
    if (launch != cudaSuccess || sync != cudaSuccess) {
        printf("LAUNCH_FAIL: %s / %s\n",
               cudaGetErrorString(launch), cudaGetErrorString(sync));
        return 2;
    }
    printf("RUN_OK (tcgen05 executed; numerical correctness not checked)\n");
    return 0;
}
