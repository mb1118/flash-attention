# 不同 TensorCore 版本在 SM120 上的可行性实测

在本机(NVIDIA RTX PRO 1000 Blackwell Laptop,**SM120 / compute capability 12.0**)上,
分别用**裸 CUDA C++ 内联 PTX** 和 **CuTeDSL(FA4)** 两条路径,给每一代 TensorCore
写最小示例并实际编译、运行,评估其可行性。

> 环境:CUDA 13.2(nvcc/ptxas V13.2.78),torch 2.12.1+cu132,nvidia-cutlass-dsl 4.5.2,flash-attn 2.8.4。
> 复现脚本与源码见 `learning/tensorcore/`(`bash run.sh`)。

## 结论矩阵(实测)

| TensorCore 代际 | 指令 | 编译 sm_XXa | 编译 sm_120a | 在 SM120 运行 | 结果正确 |
|---|---|:---:|:---:|:---:|:---:|
| 3rd gen · Ampere | `mma.sync.m16n8k16` | ✅ (sm_80) | ✅ | ✅ | ✅ max_err=0 |
| 4th gen · Hopper | `wgmma.mma_async` | ✅ (sm_90a) | ❌ ptxas 拒绝 | ❌ no kernel image | — |
| 5th gen · Blackwell DC | `tcgen05.mma` + TMEM | ✅ (sm_100a) | ❌ ptxas 拒绝 | ❌ no kernel image | — |
| 5th gen · Blackwell 消费级 | `mma.sync`(SM80 式) | ✅ (sm_120a) | ✅ | ✅ | ✅ max_err=0 |

**一句话结论**:本机 SM120 上,只有 **warp 级 `mma.sync`(3rd-gen,可移植)** 这条 TensorCore 路径可行;
Hopper 的 `wgmma` 与数据中心 Blackwell 的 `tcgen05` 都**无法编译到 sm_120,也无法在本卡启动**。
SM120 的原生 TensorCore 编程模型就是 SM80 那套 `mma.sync`,而不是 wgmma / tcgen05。

## 关键发现

### 1. 可行性分界在「指令」,不在「编译 flag」
`nvcc -arch=sm_90a` 默认会同时嵌入 sm_90a 的 SASS **和 PTX**(`cuobjdump` 可见 `ptx code: arch = sm_90 / sm_90a`)。
因此把只用 `mma.sync` 的核按 `sm_90a` 编译后,拿到 SM120 上竟也能跑对——因为驱动把可移植的 PTX
**JIT** 成了 SM120 SASS。真正决定能否在 SM120 运行的,是核里**用了什么指令**:

- `mma.sync` 是基础 ISA,PTX 能 JIT 到 sm_120 → 可行。
- `wgmma` / `tcgen05` 是架构专属指令,sm_120 的 JIT/ptxas 直接拒绝 → 不可行。

### 2. ptxas 对 sm_120 的明确拒绝
把 wgmma / tcgen05 核直接编译到 `sm_120a` 时,ptxas 报错(节选):

```
error : Instruction 'wgmma.mma_async with floating point types' not supported on .target 'sm_120'
error : Instruction 'tcgen05.mma' not supported on .target 'sm_120'
error : Feature '.cta_group::1' not supported on .target 'sm_120'
error : Feature '.kind::f16'   not supported on .target 'sm_120'
```

连 `.cta_group` / `.kind` 这些 tcgen05 的子特性都不被 sm_120 接受,说明 SM120 根本没有
数据中心 Blackwell 的 UMMA / Tensor Memory(TMEM)那套硬件。

### 3. 架构专属 SASS 在 SM120 上「no kernel image」
把 wgmma(sm_90a)、tcgen05(sm_100a)编成纯 SASS 后在 SM120 运行:

```
LAUNCH_FAIL: no kernel image is available for execution on the device
```

即便强行只保留 PTX(`code=compute_90a`)让驱动去 JIT,也会因指令不被 sm_120 支持而同样报
`no kernel image`。

### 4. 与 FA4 源码互相印证
FA4 的 `flash_attn/cute/interface.py` 按 SM 分派 MMA 代际,注释直接写明:

```python
# SM80/SM120: uses SM80 MMA, 128 threads (4 warps)
if arch // 10 in [8, 12]:
    num_threads = 128
```

即 **SM90 → wgmma(`FlashAttentionForwardSm90`)、SM100/110 → tcgen05(`FlashAttentionForwardSm100`)、
SM80/SM120 → SM80 `mma.sync`(`flash_fwd` / `*_sm120`)**,与上面的实测完全一致。

- 在本机跑 FA4 前向(自动走 SM120 路径),对照 PyTorch SDPA:`max_err ≈ 1e-3`,**正确**。
- 用 `FLASH_ATTENTION_ARCH=90a / 100a` 强制走 Hopper / Blackwell-DC 路径,FA4 直接断言拒绝:
  `Only SM 9.x is supported` / `Only SM 10.x and 11.x are supported`——它自己就知道这些
  wgmma / tcgen05 核不能在 SM120 上跑。

## 实现说明

- **可运行的两代(Ampere / SM120)**:`cuda/mma_sync.cu` 是一个完整正确的 `m16n8k16` fp16→fp32
  单 warp GEMM,按 CPU 参考校验(`max_err=0`),分别以 `sm_80` 和 `sm_120a` 编译。
- **不可运行的两代(Hopper / Blackwell-DC)**:`cuda/wgmma.cu`、`cuda/tcgen05.cu` 是**最小可行性探针**
  ——只需真实发射对应指令、能为其原生 arch 通过 ptxas 即可;因为它们在本卡根本无法启动,
  数值正确性无从验证,也无需验证。
- **CuTeDSL 侧**:`cute_fa4_sm120.py`(跑通原生路径)、`cute_force_arch.py`(强制越代路径观察失败)。
