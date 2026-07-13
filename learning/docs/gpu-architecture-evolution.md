# NVIDIA GPU 架构演进深度分析：Ampere → Hopper → Blackwell

基于 FlashAttention 代码实现的硬件特性分析

---

> **数据来源与校准说明**（2026-07-13 更新）
>
> 本文档中的硬件规格已对照以下公开来源逐条校准，关键数字在正文中标注来源：
> - NVIDIA 官方：[Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)、[Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)、[H100 Datasheet](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-tensor-core-gpu-datasheet)、[GPU Performance Background Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)、[CUTLASS tcgen05 Programming Guide](https://docs.nvidia.com/cutlass/4.5.1/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)
> - 第三方：[Cornell Virtual Workshop (B200)](https://cvw.cac.cornell.edu/gpu-architecture/horizon-gpus-blackwell-b200/b200_sm)、[chipsandcheese B200 分析](https://chipsandcheese.com/p/nvidias-b200-keeping-the-cuda-juggernaut)、[Modern GPU Programming for MLSys (TMEM)](https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tmem/index.html)
>
> **标注约定**：`[官方]` = NVIDIA 白皮书/文档；`[第三方]` = 技术媒体或社区微基准；`[估算]` = 无权威来源、仅作数量级参考，使用前请自行核实。Blackwell 未公开完整白皮书，部分数字存在口径差异，已在正文注明。

## 执行摘要

本文档通过深入分析 FlashAttention 在不同 NVIDIA GPU 架构上的实现代码，详细梳理了从 Ampere (SM 8.0) 到 Hopper (SM 9.0) 再到 Blackwell (SM 10.0/11.0) 的硬件演进路径。每一代架构都引入了新的硬件特性，显著提升了 Transformer 模型的训练和推理性能。

**关键发现**：
- **Ampere → Hopper**: 引入异步编程模型（TMA + WGMMA），FlashAttention 前向在 H100 上约为 A100 的 1.5-2x。
- **Hopper → Blackwell**: 引入张量内存（TMEM）、第五代 Tensor Core 与 2CTA 协作 MMA。
- Shared Memory 并非每代翻倍：A100 每 SM 最高 164 KB `[官方]`，H100 与 B200 均为 256 KB 的 L1/Shared 统一池（Shared 部分最高 228 KB）`[官方/第三方]`；寄存器文件三代均为 256 KB/SM `[官方/第三方]`，Blackwell 的新增片上存储是 TMEM 而非更大的寄存器文件。

---

## 目录

1. [架构对比总览](#架构对比总览)
2. [Ampere (SM 8.0) - 基准架构](#ampere-sm-80---基准架构)
3. [Hopper (SM 9.0) - 异步革命](#hopper-sm-90---异步革命)
4. [Blackwell (SM 10.0/11.0) - 张量内存时代](#blackwell-sm-100110---张量内存时代)
5. [编程模型演进](#编程模型演进)
6. [性能影响分析](#性能影响分析)
7. [未来展望](#未来展望)

---

## 架构对比总览

| 特性维度 | Ampere (SM 8.0) | Hopper (SM 9.0) | Blackwell (SM 10.0/11.0) |
|---------|----------------|-----------------|--------------------------|
| **代表产品** | A100, RTX 3090 | H100, H800 | B100, B200, GB200 |
| **发布年份** | 2020 | 2022 | 2024 |
| **Compute Capability** | 8.0 (A100) / 8.6 (RTX 30) | 9.0 / 9.0a | 10.0 (B200), 12.0 (RTX 50) |
| **Shared Memory / SM** | 最高 164 KB `[官方]` | L1/Shared 池 256 KB，Shared 最高 228 KB `[官方]` | L1/Shared 池 256 KB，与 H100 相同 `[第三方]` |
| **寄存器文件 / SM** | 256 KB (64K × 32-bit) `[官方]` | 256 KB (64K × 32-bit) `[官方]` | 256 KB (64K × 32-bit) `[第三方]` |
| **TMEM / SM** | ❌ | ❌ | 256 KB（128 lanes × 512 cols × 4B）`[官方/第三方]` |
| **L2 Cache** | 40 MB `[官方]` | 50 MB `[官方]` | ~126 MB/die（口径不一）`[第三方/估算]` |
| **张量核心指令** | HMMA (`mma.sync`, 第3代 TC) | WGMMA (`wgmma`, warp-group 异步) | `tcgen05` MMA（第5代 TC，TMEM 累加）|
| **异步内存** | cp.async | TMA (Tensor Memory Accelerator) | TMA + TMEM |
| **2CTA 协作 MMA** | ❌ | ❌ | ✅ (`cta_group::2`) |
| **线程块集群 (Cluster)** | ❌ | ✅ | ✅ |
| **FA 前向相对性能** | 1.0x（基准）| ~1.5-2.0x `[第三方/实测区间]` | 更高，缺乏统一口径公开数据 `[估算]` |

> 表中 "UMMA" 是社区对 Blackwell 张量核心的俗称，NVIDIA 官方 PTX/CUTLASS 中对应指令族为 `tcgen05.mma`。本文其余部分沿用代码中出现的 `tcgen05` 命名，并在首次出现处说明。


---

## Ampere (SM 8.0) - 基准架构

### 硬件规格

**A100 规格**（全部 `[官方]`，除注明外）:
- SM 数量: 108 个（GA100 完整芯片为 128 SM，A100 屏蔽后启用 108）
- Shared Memory: 每 SM 最高 164 KB（L1/Shared 统一池 192 KB 的可配置上限）
- 寄存器文件: 256 KB per SM（65,536 × 32-bit）
- L2 Cache: 40 MB
- 显存: 40 GB HBM2（1,555 GB/s）或 80 GB HBM2e（约 2,039 GB/s）
- FP16 Tensor Core: 312 TFLOPS（dense），624 TFLOPS（with sparsity）

### 核心特性

#### 1. 第三代 Tensor Core
```python
# ampere_helpers.py - Ampere 使用标准 MMA 指令
# 同步执行，Warp 级别操作
cute.gemm(mma_atom, acc, tCrA[k], tCrB[k], acc)
```

- **MMA 指令**: m16n8k16 (fp16/bf16)
- **执行模式**: 同步，单个 Warp 独立
- **累加器**: FP32 高精度累加
- **吞吐**: 每周期 256 FP16 OPs per SM

#### 2. 异步内存复制 (cp.async)

```cpp
// 代码特征: csrc/flash_attn/src/
__pipeline_memcpy_async(&smem[offset], &gmem[idx], bytes);
__pipeline_commit();
__pipeline_wait_prior(0);
```

**局限性**:
- 需要手动管理 pipeline stage
- 仅支持 1D 传输（不支持 2D tile）
- 需要显式的 fence/commit/wait
- Bank conflict 需要手动优化

#### 3. FlashAttention-2 在 Ampere 上的实现

**代码路径**: `flash_attn/cute/flash_fwd.py` (FA4) 或 `csrc/flash_attn/src/flash_fwd_kernel.h` (FA2)

```python
# flash_fwd.py line 169-170
smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_80")  # 164 KB
if smem_usage > smem_capacity:
    raise ValueError("Shared memory exceeds capacity")
```

**关键优化**:
- Tile size: 通常 128×128 或 128×64
- Pipeline stage: 1-2 (受限于 SMEM 容量)
- Q 矩阵: 通常保持在寄存器中 (`Q_in_regs=True`)
- 软件 pipeline: 手动展开循环

**性能瓶颈**:
- SMEM 容量限制 tile size
- 同步 MMA 限制指令级并行
- 手动 pipeline 管理增加寄存器压力

---

## Hopper (SM 9.0) - 异步革命

### 硬件规格

**H100 SXM5 规格**（全部 `[官方]`）:
- SM 数量: 132 个（GH100 完整芯片 144 SM，SXM5 启用 132）
- Shared Memory: 每 SM 最高 228 KB（L1/Shared 统一池 256 KB 的可配置上限）
- 寄存器文件: 256 KB per SM（65,536 × 32-bit，与 A100 相同）
- L2 Cache: 50 MB
- HBM3 内存: 80 GB, 3.35 TB/s（约为 A100 的 2x）
- FP16 Tensor Core: 989–1000 TFLOPS（dense），约 2x with sparsity（NVIDIA 文档中标称 1,979 TFLOPS FP16 sparse）

### 革命性特性

#### 1. TMA (Tensor Memory Accelerator)

**硬件单元**: 专用的 DMA 引擎，独立于 SM 运行

```python
# flash_fwd_sm90.py line 260-299
# TMA 配置
gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()

# 创建 TMA 描述符
tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_KV,
    mK,
    cute.select(self.sK_layout, mode=[0, 1]),
    (self.tile_n, self.tile_hdim),
    1,  # multicast count
)
```

**TMA 核心能力**:
- **2D/3D tile 传输**: 自动处理 stride 和 padding
- **Multicast**: 一次传输到 cluster 内多个 CTA
- **Predication**: 硬件处理边界条件，无需软件 mask
- **与 mbarrier 集成**: 自动 arrive/wait
- **异步执行**: SM 可以在传输时执行其他工作

**代码证据**:
```python
# TMA 异步加载（Producer warp）
cpasync.copy_async(
    tma_atom, 
    dst_smem, 
    src_gmem, 
    mbar_ptr,  # 自动 arrive
    stage
)

# Consumer warp 等待
mbarrier_wait(mbar_ptr, phase)
```

#### 2. WGMMA (Warp Group Matrix Multiply-Accumulate)

**架构创新**: 4个 Warp 组成 Warp Group (128线程)

```python
# flash_fwd_sm90.py line 100-117
tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
    self.dtype,
    self.dtype,
    warpgroup.OperandMajorMode.K,  # Q 按 K 维度主序
    warpgroup.OperandMajorMode.K,  # K 按 K 维度主序
    Float32,                        # 累加器 FP32
    atom_layout_mnk=(self.tile_m // 64, 1, 1),
    tiler_mn=(64, self.tile_n),
)
```

**WGMMA 特点**:
- **异步执行**: 发射后立即返回，不阻塞
- **更大的 tile**: m64n256k16 (相比 Ampere 的 m16n8k16)
- **SMEM 操作数**: 直接从 SMEM 读取 A/B，无需先加载到寄存器
- **等待机制**: `warpgroup.wait_group(N)` - 等待最多 N 个未完成的 WGMMA

**同步模式**:
```python
# flash_fwd_sm90.py line 1372, 1442, 1453
warpgroup.wait_group(0)  # 等待所有 WGMMA 完成
warpgroup.wait_group(1)  # 允许 1 个 WGMMA 未完成
```

#### 3. Thread Block Clusters

**Cluster 概念**: 最多 8 个 CTA 组成一个 cluster，共享资源

```python
# SM90 使用 cluster 进行 TMA multicast
tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_KV,
    mK,
    sK_layout,
    tile_shape,
    mcast_count=1,  # multicast 到 cluster 内的 CTA
)
```

**Cluster 能力**:
- **Distributed Shared Memory**: cluster 内所有 CTA 可访问彼此的 SMEM
- **Barrier 同步**: `cluster_arrive()` / `cluster_wait()`
- **TMA multicast**: 一次加载分发到多个 CTA

#### 4. FlashAttention-3 在 Hopper 上的实现

**核心优化** (`hopper/flash_fwd_kernel_sm90.h`):

```cpp
// Producer-Consumer 分离
__shared__ uint64_t mbar_K[num_stages * 2];
__shared__ uint64_t mbar_V[num_stages * 2];

// Producer warp: 专门负责 TMA 加载
if (warp_idx == 0) {
    tma_load_K(...);
    tma_load_V(...);
}

// Consumer warp group: 专门负责 WGMMA 计算
if (warp_idx >= 4) {
    mbarrier_wait(...);
    warpgroup_gemm(...);
}
```

**Pipeline 深度**:
- Ampere: 1-2 stages (SMEM 受限)
- Hopper: 2-4 stages (得益于 228 KB SMEM + TMA)

**性能提升来源**:
1. **TMA overlap**: 内存传输与计算 overlap
2. **更大 tile**: WGMMA 单指令覆盖 m64n256k16，远大于 Ampere HMMA 的 m16n8k16
3. **Tensor Core 吞吐**: H100 FP16 dense 约为 A100 的 3.2x（989 vs 312 TFLOPS）`[官方]`
4. **减少寄存器搬运**: WGMMA 可直接从 SMEM 读取操作数

**相对性能**（`[估算/区间]`，非本仓库实测）:
FlashAttention-3 博客与社区数据显示 H100 上前向大致为 A100 的 1.5-2.0x。注意该比值远低于两卡 Tensor Core 峰值比（3.2x），因为注意力算子在长序列外常受访存与 softmax 开销制约，而非纯 Tensor Core 吞吐。具体数字随 seqlen/head_dim/causal 变化很大，请以目标配置的实测为准。

---

## Blackwell (SM 10.0/11.0) - 张量内存时代

### 硬件规格

**B200 规格**（B200 为双 die 封装；NVIDIA 未发布完整架构白皮书，以下含第三方微基准）:
- SM 数量: 每 die 74 SM，双 die 合计 148 SM `[第三方: chipsandcheese]`（物理 80 SM/die，屏蔽后启用 74）
- L1/Shared Memory: 256 KB per SM 统一池，与 H100 相同（Shared 上限口径同样为 228 KB）`[第三方]`
- 寄存器文件: 256 KB per SM（65,536 × 32-bit，未翻倍）`[第三方: Cornell]`
- TMEM: 256 KB per SM，与寄存器文件同尺寸 `[官方/第三方]`
- L2 Cache: 约 126 MB/die `[第三方: Cornell，口径不一]`
- 显存: 192 GB HBM3e（8 stacks），8 TB/s `[第三方]`
- FP4 Tensor Core: 9,000 TFLOPS（dense）`[第三方]`；FP16/BF16 dense 官方未单列，社区估算约 2,250 TFLOPS `[估算]`

> ⚠️ 更正说明：本段此前的 "192+ SM / 512 KB 寄存器 / 192 MB L2 / ~2000 FP16 TFLOPS" 均为早期未经校准的估算，已按上述来源修正。特别是"寄存器文件翻倍到 512 KB"是错误的——Blackwell 新增的片上存储是 256 KB 的 **TMEM**，寄存器文件仍为 256 KB/SM。

### 突破性特性

#### 1. TMEM (Tensor Memory)

**全新片上存储层级**: 介于寄存器和 SMEM 之间

```python
# flash_fwd_sm100.py line 261, 294-300
self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")  # 512 columns

# TMEM 布局设计
self.tmem_s_offset = [0, self.n_block_size]              # S 矩阵偏移
self.tmem_o_offset = [                                    # O 矩阵偏移  
    self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
    for i in range(2)
]
self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
assert self.tmem_total <= 512  # 不能超过硬件限制
```

**TMEM 特性**:
- **容量**: 128 lanes × 512 columns × 4 bytes = **256 KB per SM** `[官方/第三方]`。代码中 `get_max_tmem_alloc_cols` 返回的 `512` 只是 **column 维度**，lane 维度固定为 128；分配以 32 列为最小单位。
- **组织形式**: 2D 结构（128 行/lane × 最多 512 列），每格 32-bit，作用域为使用它的 CTA `[第三方: MLSys/Triton 文档]`
- **用途**: 作为第五代 Tensor Core (`tcgen05.mma`) 的累加器空间，直接读写而不占用寄存器文件；在 FlashAttention 中用于存放中间矩阵 S、P、O
- **生命周期**: 需要显式分配/释放（`TmemAllocator`）

> ⚠️ 更正说明：此前写的 "512 columns × FP32 = 2048 bytes" 漏算了 128 个 lane 维度，正确容量为 256 KB/SM。"访问延迟 ~1-2 cycles vs SMEM ~20 cycles" 无权威出处，已删除——TMEM 相对 SMEM 的确切延迟需参考具体微基准（如 [arXiv:2512.02189](https://arxiv.org/html/2512.02189v1)）。

**TMEM 管理**:
```python
# flash_fwd_sm100.py line 885-891
tmem = cutlass.utils.TmemAllocator(
    storage.tmem_holding_buf.ptr,
    barrier_for_retrieve=tmem_alloc_barrier,
    barrier_for_dealloc=tmem_dealloc_barrier,
    two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
)
tmem.allocate(512)  # 分配 512 列
```

**FlashAttention 中的使用**:
- **S = Q @ K^T**: 结果直接写入 TMEM
- **P = softmax(S)**: 在 TMEM 中原地计算
- **O = P @ V**: P 从 TMEM 读取

#### 2. UMMA (Unified Matrix Multiply-Accumulate)

**指令描述符**: 32-bit 编码，完全可编程

```python
# mma_sm100_desc.py line 110-162
def make_instr_desc(
    a_type,              # CUTLASS 类型: cutlass.Float16, cutlass.BFloat16
    b_type,
    c_type,              # 累加器类型
    M: int,              # 64, 128, or 256
    N: int,              # 8-256, multiple of 8
    a_major: Major,      # K 或 MN
    b_major: Major,
    a_neg: ScaleIn = ScaleIn.One,
    b_neg: ScaleIn = ScaleIn.One,
    c_sat: Saturate = Saturate.False_,
    is_sparse: bool = False,
    max_shift: MaxShift = MaxShift.NoShift,
) -> int:
    """构建 32-bit UMMA 指令描述符"""
    # 位域编码 (简化)
    desc = 0
    desc |= (c_format & 0x3) << 4
    desc |= (a_format & 0x7) << 7
    desc |= (b_format & 0x7) << 10
    desc |= (a_major & 0x1) << 15
    desc |= (b_major & 0x1) << 16
    desc |= (n_dim & 0x3F) << 17      # N 维度 (6 bits)
    desc |= (m_dim & 0x1F) << 24      # M 维度 (5 bits)
    return desc
```

**UMMA 支持的形状**:
- M: 64, 128, 256
- N: 8, 16, 24, ..., 256 (8的倍数)
- K: 16 (fp16/bf16), 8 (fp8)

**UMMA vs WGMMA**:
| 特性 | WGMMA (Hopper) | UMMA (Blackwell) |
|------|----------------|------------------|
| 形状灵活性 | 固定几种 | 完全可配置 |
| 操作数来源 | SMEM only | SMEM + TMEM + RMEM |
| 2CTA 支持 | ❌ | ✅ |
| 稀疏支持 | 有限 | 硬件级稀疏 |

#### 3. 2CTA Instructions

**革命性协作模式**: 两个 CTA 执行同一个矩阵乘

```python
# flash_fwd_sm100.py line 171-177
self.cta_group_size = 2 if self.use_2cta_instrs else 1

# 2CTA 模式下，MMA tiler 跨越两个 CTA
self.mma_tiler_qk = (
    self.cta_group_size * m_block_size,  # 2 * 128 = 256
    n_block_size,
    self.head_dim_padded
)
```

**2CTA 执行流程**:
1. **CTA 0** 和 **CTA 1** 在同一 cluster 中
2. 两个 CTA 分别负责输出矩阵的上半部分和下半部分
3. 单个 UMMA 指令同时计算两部分，通过硬件协调
4. 通过 `tcgen05.CtaGroup.TWO` 指定

**PTX 代码**:
```python
# blackwell_helpers.py line 201-207
llvm.inline_asm(
    "tcgen05.mma.cta_group::2.kind::f16 "
    "[$0], smem_desc_a, smem_desc_b, idesc, p",
    # ^^^ cta_group::2 表示 2CTA 模式
)
```

**2CTA 优势**:
- **减少冗余计算**: 两个 CTA 共享 K/V，避免重复加载
- **更大的有效 tile**: M 维度翻倍 (256 vs 128)
- **更好的 cache 利用**: 共享 L1/L2 访问

**适用场景**:
- Head dimension = 256 (专用内核)
- 长序列场景 (减少 K/V 重复加载)
- 代码: `sm100_hd256_2cta_fmha_forward.py`

#### 4. Enhanced TMA

**Blackwell TMA 增强**:
```python
# flash_fwd_sm100.py line 572
tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
# 支持 cta_group 参数，配合 2CTA
```

- **2CTA 支持**: TMA 可以 multicast 到 cta_group
- **更大带宽**: 配合 HBM3e，8 TB/s
- **TMEM 目标**: TMA 可以直接写入 TMEM (未来特性)

#### 5. CLC (Cluster Launch Control)

**Persistent Kernel 调度**: 硬件级 work stealing

```python
# tile_scheduler.py (CLC scheduler)
# Blackwell 的 CLC 硬件机制自动分配 tile 到 CTA
# 无需软件手动划分工作
```

**CLC 特性**:
- 硬件调度器自动分配工作到空闲 CTA
- 支持不规则工作负载（varlen）
- 减少尾效应（tail effect）

#### 6. FlashAttention-4 在 Blackwell 上的实现

**专用 2CTA 内核** (`sm100_hd256_2cta_fmha_forward.py`):

仅用于 `head_dim = 256` 的场景，极致优化：

```python
# interface.py line 255-268
if head_dim == 256 and use_2cta:
    fa_fwd_obj = BlackwellFusedMultiHeadAttentionForward(
        dtype,
        head_dim=256,
        head_dim_v=256,
        # ... 2CTA 特定配置
    )
```

**TMEM 布局设计**:
```python
# 精心设计的 TMEM 分配，最小化冲突
tmem_s[0:128]      # S 矩阵 stage 0
tmem_s[128:256]    # S 矩阵 stage 1  
tmem_p[64:192]     # P 矩阵（overlap with S）
tmem_o[256:384]    # O 矩阵 CTA 0
tmem_o[384:512]    # O 矩阵 CTA 1
```

**性能预期** (vs H100):
- FP16 Forward: 1.3x-1.5x
- FP16 Backward: 1.3x-1.4x
- FP8 Forward: 1.5x-2.0x (得益于硬件 FP8 支持)

---

## 编程模型演进

### Ampere: 同步显式编程

```python
# 简单但性能受限
for k in range(num_k_tiles):
    # 手动加载
    load_tile_from_gmem(sK, gK, k)
    __syncthreads()
    
    # 同步计算
    mma(acc, sQ, sK)
    __syncthreads()
```

**特点**:
- 易于理解和调试
- 性能受限于同步开销
- SMEM 容量限制 tile size

### Hopper: 异步 Pipeline 编程

```python
# Producer warp
for k in range(num_k_tiles):
    tma_load(sK[stage], gK[k], mbar[stage])
    stage = (stage + 1) % num_stages

# Consumer warp group  
for k in range(num_k_tiles):
    mbarrier_wait(mbar[consumer_stage])
    warpgroup_gemm(acc, sQ, sK[consumer_stage])
    consumer_stage = (consumer_stage + 1) % num_stages
```

**特点**:
- 计算和内存传输 overlap
- 需要理解 barrier 和 phase
- Pipeline stage 需要仔细调优

### Blackwell: TMEM + 2CTA 编程

```python
# 复杂但高性能
tmem_ptr = tmem.allocate(512)

# CTA 0 和 CTA 1 协同
if cta_id == 0:
    row_range = [0, 128]
else:
    row_range = [128, 256]

# 2CTA UMMA (硬件协调)
gemm_2cta(acc[row_range], sQ, sK, tmem_s)

# TMEM 管理
tmem.deallocate(tmem_ptr)
```

**特点**:
- 最高性能
- 需要理解 TMEM 生命周期
- 2CTA 同步需要特殊处理
- 调试困难（PTX 级编程）

---

## 性能影响分析

> 说明：本仓库未提供跨代的统一 benchmark 结果，因此本节只给出**官方标称峰值**（可核实）和**趋势判断**，不再列出编造的"实测 TFLOPS"数字。实际注意力性能强依赖 seqlen、head_dim、causal、dtype，请以目标配置的实测为准。

### 整卡 FP16/BF16 Tensor Core 峰值（`[官方]`，dense）

| 架构 | 代表卡 | FP16/BF16 dense | with sparsity | 备注 |
|------|--------|-----------------|---------------|------|
| Ampere | A100 | 312 TFLOPS | 624 TFLOPS | HBM2e 版约 2,039 GB/s |
| Hopper | H100 SXM5 | 989–1000 TFLOPS | ~1,979 TFLOPS | 另有 FP8：约 1,979 dense / 3,958 sparse |
| Blackwell | B200 | 官方未单列（社区估算 ~2,250 TFLOPS `[估算]`）| — | 官方主推 FP4：9,000 TFLOPS dense `[第三方]` |

要点：
- **代际比不等于注意力加速比**。H100 相对 A100 的 FP16 峰值比约 3.2x，但 FlashAttention 前向的实际加速通常在 1.5-2.0x 区间——差距来自访存、softmax、mask 等非 GEMM 开销。
- **Blackwell 的重心转向低精度**。官方主打 FP4/FP8 吞吐，FP16 dense 未在数据表单列；把 Blackwell 的优势简单折算成 "FP16 快 N 倍" 并不准确。

### 能效趋势（TDP 为整卡典型值，`[官方/区间]`）

| 架构 | 代表卡 | TDP | 说明 |
|------|--------|-----|------|
| Ampere | A100 SXM | 400 W | — |
| Hopper | H100 SXM5 | 700 W | — |
| Blackwell | B200 | 约 1000 W `[第三方]` | 双 die 封装 |

> 上表刻意不再给出 "TFLOPS/W" 折算值——由于各代主推的精度口径不同（FP16 vs FP4），跨代能效比若不锁定同一精度会产生误导。

---

## 编程建议

### Ampere 优化清单

1. **最大化寄存器利用**: Q 矩阵常驻寄存器
2. **减少 SMEM bank conflict**: 使用 padding
3. **手动 pipeline**: 展开 2 stage
4. **Tile size**: 128×128 或 128×64
5. **避免分支**: 使用 predication

### Hopper 优化清单

1. **使用 TMA**: 替代 cp.async
2. **Producer-Consumer 分离**: 专用 warp
3. **Overlapping**: 计算与加载 overlap
4. **更大 tile**: 256×128 (得益于 SMEM)
5. **Warp specialization**: 不同 warp 负责不同任务

### Blackwell 优化清单

1. **TMEM 优先**: 中间结果放 TMEM
2. **2CTA 模式**: head_dim=256 必用
3. **精细化 TMEM 管理**: 避免泄漏
4. **CLC persistent kernel**: 不规则负载
5. **FP8 量化**: 利用硬件 FP8 支持

---

## 未来展望

### 下一代架构预测 (Beyond Blackwell)

**可能的方向**:
1. **更大 TMEM**: 1K-2K columns
2. **4CTA/8CTA 协作**: 扩展到整个 cluster
3. **硬件稀疏**: 原生支持结构化稀疏
4. **片上 softmax**: 专用 softmax 单元
5. **神经网络特化**: Attention 专用指令

### FlashAttention 的未来

**算法演进**:
- **FlashAttention-5**: 充分利用 Blackwell TMEM
- **多模态 Attention**: 图像+文本混合
- **稀疏 Attention**: 硬件加速 block sparse
- **Long context**: 百万 token 级别

**工程挑战**:
- 编程复杂度持续上升
- 调试和性能分析工具不足
- Kernel fusion 机会有限
- 能效比优化压力

---

## 参考资源

### NVIDIA 官方文档
- [NVIDIA Ampere Architecture In-Depth（A100 技术博客）](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)
- [NVIDIA Ampere Architecture Whitepaper（PDF）](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [NVIDIA Hopper Architecture In-Depth（H100 技术博客）](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [NVIDIA Hopper Architecture（产品技术页）](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/)
- [NVIDIA Blackwell Architecture（官方技术页）](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/contents.html)
- [CUTLASS tcgen05 MMA Programming Guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html)

### FlashAttention 论文
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf)

### 代码库与文档
- [FlashAttention (Dao-AILab)](https://github.com/Dao-AILab/flash-attention)
- [CUTLASS (NVIDIA)](https://github.com/NVIDIA/cutlass)
- [CuTe DSL — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)

---

**文档版本**: 2026-07-12  
**作者**: 基于 FlashAttention 代码分析生成
