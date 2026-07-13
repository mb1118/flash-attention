# Flash Attention 版本架构全景梳理

本文档详细梳理 flash-attention 项目中各个版本的实现和 GPU 平台支持情况。

## 概述

本项目包含 **4 个主要版本**的 Flash Attention 实现，各自针对不同的 GPU 架构优化：

1. **FlashAttention-2 (FA2)** - 成熟稳定的 CUDA C++ 实现
2. **FlashAttention-3 (FA3)** - Hopper 深度优化版本（已初步支持 Blackwell）
3. **FlashAttention-4 (FA4)** - Python + CuTeDSL 最新实现
4. **ROCm Backend** - AMD GPU 支持

---

## 1. FlashAttention-2 (FA2)

### 基本信息
- **代码位置**: `csrc/flash_attn/src/`
- **实现语言**: CUDA C++ (使用 CUTLASS 库)
- **包名**: `flash-attn`
- **论文**: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"

### 支持的 GPU 平台

| GPU 架构 | Compute Capability | 代表型号 |
|---------|-------------------|---------|
| **Ampere** | SM 8.0 (A100) / SM 8.6 (GA10x) | A100 为 8.0；RTX 3090、A6000 为 8.6 |
| **Ada Lovelace** | SM 8.9 | RTX 4090, L40S |
| **Hopper** | SM 9.0 | H100, H800 |

### 关键特性

- **数据类型**: fp16, bf16
- **Head Dimension**: 32, 64, 96, 128, 192, 256
- **高级特性**:
  - Dropout
  - Causal masking
  - Sliding window (local attention)
  - ALiBi (Attention with Linear Biases)
  - Paged KV cache
  - Softcapping
  - MQA/GQA (Multi-Query/Grouped-Query Attention)

### 架构实现

```cpp
// flash_fwd_launch_template.h
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
```

编译时生成针对不同架构的内核实例化：
```
csrc/flash_attn/src/
├── flash_fwd_hdim{64,96,128,192,256}_{fp16,bf16}_{causal,}_sm80.cu
├── flash_bwd_hdim{64,96,128,192,256}_{fp16,bf16}_{causal,}_sm80.cu
└── ... (对应的 backward 内核)
```

### 编译配置

```bash
# setup.py 中的架构配置
FLASH_ATTN_CUDA_ARCHS="80;90;100;110;120"

# 生成的编译标志
-gencode arch=compute_80,code=sm_80
-gencode arch=compute_90,code=sm_90
```

---

## 2. FlashAttention-3 (FA3)

### 基本信息
- **代码位置**: `hopper/`
- **实现语言**: CUDA C++ (CUTLASS 3.x, 使用 CuTe DSL)
- **包名**: `flash-attn-3`
- **状态**: Beta release
- **论文/博客**: [FlashAttention-3 博客](https://tridao.me/blog/2024/flash3/)

### 支持的 GPU 平台

| GPU 架构 | Compute Capability | 代表型号 | 优化程度 |
|---------|-------------------|---------|---------|
| **Hopper** | SM 9.0a | H100, H800 | 专门优化（主力目标） |
| Ampere | SM 8.0 | A100, RTX 3090 | 可选支持，性能非最优 |
| Blackwell | SM 10.0a | B100, B200 | 初步支持（`hopper/setup.py` 已有 `compute_100a` 编译规则及 SM100 实例，如 `flash_fwd_hdim128_bf16_sm100.cu`） |

### 系统要求

- **CUDA 版本**: >= 12.3 (强烈推荐 CUDA 12.8)
- **目标硬件**: H100/H800 GPU

### 关键特性

- **针对 Hopper 架构深度优化**:
  - TMA (Tensor Memory Accelerator) 异步加载
  - WGMMA (Warp Group Matrix Multiply Accumulate)
  - Cluster scheduling
  - Persistent kernels

- **数据类型**:
  - fp16/bf16: forward + backward
  - fp8: forward only

- **Head Dimension**: 64, 96, 128, 192, 256

- **高级特性**:
  - Pack-GQA 优化
  - Paged KV cache
  - Split-KV for long sequences
  - Softcapping
  - Varlen (variable length sequences)

### 架构模板

```cpp
// hopper/flash.h
template <int Arch, typename T, int kHeadDim, int kHeadDimV, 
          bool Split, bool PagedKVNonTMA, bool Has_softcap, bool PackGQA>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);

template <int Arch, typename T, int kHeadDim, bool Has_softcap>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
```

### 编译配置

```python
# hopper/setup.py
# SM90 文件编译标志
cuda_post_cflags = ['-gencode', 'arch=compute_90a,code=sm_90a']

# SM80 文件编译标志（可选）
cuda_post_cflags_sm80 = ['-gencode', 'arch=compute_80,code=sm_80']
```

### 安装方式

```bash
# 标准安装
cd hopper
python setup.py install

# 使用 uv 安装
pip install flash-attn-3

# 或通过 pyproject.toml
[tool.uv.sources]
flash-attn-3 = { git = "https://github.com/Dao-AILab/flash-attention", subdirectory = "hopper" }
```

---

## 3. FlashAttention-4 (FA4)

### 基本信息
- **代码位置**: `flash_attn/cute/`
- **实现语言**: Python + CuTeDSL (JIT 编译)
- **包名**: `flash-attn-4`
- **状态**: 活跃开发
- **依赖**: `nvidia-cutlass-dsl>=4.5.2`

### 支持的 GPU 平台

| GPU 架构 | Compute Capability | 代表型号 | 支持级别 |
|---------|-------------------|---------|---------|
| **Ampere** | SM 8.0 | A100, RTX 3090 | 仅 forward（backward 未支持，见下） |
| **Hopper** | SM 9.0 | H100, H800 | 完整优化 |
| **Blackwell** | SM 10.0, 11.0 | B100, B200, GB200 | 最新特性 |
| **Blackwell GeForce** | SM 12.0 | RTX 50 系列, DGX Spark | forward + backward |

> **注**: FA4 的 backward 入口 `_flash_attn_bwd` 断言 `arch // 10 in [9, 10, 11, 12]`，因此在真实 SM80 (Ampere) 硬件上**只能跑 forward**。`FlashAttentionBackwardSm80` 主要作为 `FlashAttentionBackwardSm120` 的基类存在（SM120 复用 SM80 代码路径）。

### 关键特性

#### 编程模型
- **完全 Python 实现**: 使用 CuTeDSL 编写内核
- **JIT 编译**: 运行时编译到 PTX/CUBIN
- **可扩展性**: 易于添加自定义 attention 变体

#### 数据类型与维度
- **数据类型**:
  - fp16, bf16: forward + backward
  - fp8 (e4m3fn / e5m2): **仅 forward，且仅 SM100**（`requires_grad` 时抛 `NotImplementedError`；输出被强制为 bf16）
- **Head Dimension**（依架构而定）:
  - SM100/110 通用: 8-128（需被 alignment 整除）
  - SM90: 8-256
  - DeepSeek MLA: 192/128 (专门优化)
  - hd256 专用内核: 256/256（head_dim == head_dim_v == 256）
  - MLA absorbed shape: 512

#### 高级特性

1. **用户自定义修改函数**
   - `score_mod`: 自定义 attention score 修改（如 softcapping, ALiBi）
   - `mask_mod`: 自定义 mask 逻辑

2. **稀疏性支持**
   - Block sparsity
   - Learnable sink tokens
   - Sparse KV attention

3. **长序列优化**
   - Split-KV: 将 K/V 分块处理超长序列
   - Paged KV cache with TMA
   - Variable length sequences (varlen)

4. **架构特定优化**
   - **SM100/110 (Blackwell)**:
     - 2CTA instructions（通用路径用于 hd 舍入到 128/192 的 non-causal 等情形；hd256 走独立专用内核）
     - UMMA-based GEMM
     - Enhanced tensor memory
   - **Cluster Launch Control (CLC)**: Persistent scheduling

### 架构分发逻辑

```python
# flash_attn/cute/interface.py
def _get_device_arch():
    """运行时检测 GPU 架构
    
    可通过环境变量覆盖:
      FLASH_ATTENTION_ARCH=sm_100 (内核选择)
      CUTE_DSL_ARCH=sm_100 (JIT 编译目标)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)

# 运行时架构选择
arch = _get_device_arch()  # 返回: 80, 90, 100, 110, 120

if arch // 10 == 8:  # SM 8.x (Ampere)
    fa_fwd = FlashAttentionForwardSm80(...)
elif arch // 10 == 9:  # SM 9.x (Hopper)
    fa_fwd = FlashAttentionForwardSm90(...)
elif arch // 10 in [10, 11]:  # SM 10.x/11.x (Blackwell)
    # hd256 专用内核与通用 2CTA 是两套独立机制：
    #   - use_dedicated_hd256_kernel = (head_dim == 256 and head_dim_v == 256)
    #   - 通用 use_2cta_instrs 面向 head_dim 舍入到 128/192 的 non-causal 等情形（不含 256）
    if head_dim == 256 and head_dim_v == 256:
        # 专用 hd256 2CTA 内核（最优性能）
        fa_fwd = BlackwellFusedMultiHeadAttentionForward(...)
    else:
        fa_fwd = FlashAttentionForwardSm100(...)  # 内部按条件启用通用 2CTA
elif arch // 10 == 12:  # SM 12.x (Blackwell GeForce)
    fa_fwd = FlashAttentionForwardSm120(...)  # 复用 SM80 MMA 路径
```

### 核心内核文件

#### Forward Kernels
```
flash_attn/cute/
├── flash_fwd.py                          # SM80 (Ampere) forward
├── flash_fwd_sm90.py                     # SM90 (Hopper) forward
├── flash_fwd_sm100.py                    # SM100/110 (Blackwell) forward
├── flash_fwd_sm120.py                    # SM120 (Blackwell GeForce) forward
├── sm100_hd256_2cta_fmha_forward.py      # 专用 2CTA 内核 (hdim=256)
├── flash_fwd_mla_sm100.py                # DeepSeek MLA 优化
└── flash_fwd_combine.py                  # Split-KV 合并
```

#### Backward Kernels
```
flash_attn/cute/
├── flash_bwd.py                          # SM80 backward（SM120 的基类；SM80 硬件不直接使用）
├── flash_bwd_sm90.py                     # SM90 backward
├── flash_bwd_sm100.py                    # SM100/110 backward
├── flash_bwd_sm120.py                    # SM120 backward
├── sm100_hd256_2cta_fmha_backward.py     # 专用 2CTA backward (hdim=256)
├── flash_bwd_mla_sm100.py                # MLA backward
├── flash_bwd_mla_dk_sm100.py             # MLA backward (dK)
├── flash_bwd_mla_dq_dqv_sm100.py         # MLA backward (dQ/dQV)
├── flash_bwd_preprocess.py               # Backward 预处理
└── flash_bwd_postprocess.py              # Backward 后处理
```

#### 辅助模块
```
flash_attn/cute/
├── softmax.py                            # Online softmax 实现
├── mask.py                               # Attention mask 抽象
├── block_info.py                         # Tile 维度管理
├── seqlen_info.py                        # 序列长度追踪
├── pipeline.py                           # Pipeline 状态管理
├── tile_scheduler.py                     # Tile 调度策略
├── pack_gqa.py                           # GQA packing 优化
├── paged_kv.py                           # Paged KV cache
├── ampere_helpers.py                     # SM80 辅助
├── blackwell_helpers.py                  # SM100 UMMA 辅助
├── mma_sm100_desc.py                     # SM100 MMA 描述符
├── named_barrier.py                      # 命名 barrier
├── block_sparsity.py                     # Block sparse 支持
├── block_sparse_utils.py                 # Block sparse 辅助
└── compute_block_sparsity.py             # Block sparse mask 计算
```

> **注**: 目录下没有 `hopper_helpers.py`——SM90 的 WGMMA/流水线相关辅助内联在 `flash_fwd_sm90.py` / `flash_bwd_sm90.py` / `pipeline.py` 中。

### 编译与缓存

```python
# JIT 编译流程
# 1. 生成缓存 key（基于配置参数）
compile_key = (dtype, head_dim, head_dim_v, causal, arch, ...)

# 2. 检查缓存
if compile_key not in _flash_attn_fwd.compile_cache:
    # 3. 编译内核
    kernel = fa_fwd.compile(...)
    _flash_attn_fwd.compile_cache[compile_key] = kernel

# 4. 执行内核
kernel.launch(...)
```

#### 缓存配置

```bash
# 环境变量
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1  # 启用磁盘缓存
# 缓存位置: /tmp/${USER}/flash_attention_cute_dsl_cache/

# 调试选项
CUTE_CUBIN_PATH=/path/to/dump              # 导出 CUBIN/SASS
CUTE_DSL_KEEP_PTX=1                        # 保留 PTX
CUTE_DSL_PTXAS_PATH=/custom/ptxas          # 自定义 ptxas 编译器
CUTE_DSL_LINEINFO=1                        # 添加行号信息
```

### 测试工作流

#### 快速两阶段测试
```bash
# Pass 1: 并行编译（无需 GPU 内存）
FLASH_ATTENTION_FAKE_TENSOR=1 \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
pytest -n 64 -x tests/cute/test_flash_attn.py

# Pass 2: 执行测试（使用缓存的编译结果）
FLASH_ATTENTION_FAKE_TENSOR=0 \
FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1 \
pytest -x tests/cute/test_flash_attn.py
```

### 安装方式

```bash
# 标准安装
pip install flash-attn-4

# CUDA 13 优化版本
pip install "flash-attn-4[cu13]"

# 开发安装
pip install -e "flash_attn/cute[dev]"
```

---

## 4. ROCm Backend (AMD GPU 支持)

### 基本信息
- **代码位置**: 
  - CK Backend: `csrc/composable_kernel/`, `csrc/flash_attn_ck/`
  - Triton Backend: `third_party/aiter/`
- **实现语言**: HIP / Triton
- **包名**: `flash-attn` (with ROCm support)

### 支持的 AMD GPU 平台

#### Composable Kernel (CK) Backend（默认）

| GPU 系列 | 代表型号 | 架构 |
|---------|---------|------|
| **MI200x** | MI210, MI250, MI250X | CDNA 2 |
| **MI300x** | MI300A, MI300X | CDNA 3 |
| **MI355x** | MI355X | CDNA 4 |
| **RDNA 3** | RX 7900 XTX, RX 7900 XT | Gaming/Consumer |
| **RDNA 4** | RX 9000 series | Gaming/Consumer |

**系统要求**:
- ROCm >= 6.0
- 数据类型: fp16, bf16
- Head dimension: 最大 256 (forward + backward)

#### Triton Backend（可选）

| GPU 系列 | 代表型号 | 架构 |
|---------|---------|------|
| **CDNA** | MI200, MI300 | Data Center |
| **RDNA** | RX 7000, RX 9000 | Gaming/Consumer |

**系统要求**:
- 数据类型: fp16, bf16, fp32
- 特性: causal, varlen, MQA/GQA, dropout, rotary, ALiBi, paged attention, FP8

### 安装方式

#### CK Backend (默认)
```bash
# 标准 ROCm 安装
pip install flash-attn --no-build-isolation
```

#### Triton Backend
```bash
# 启用 Triton 后端
cd flash-attention
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" pip install --no-build-isolation .

# 运行时环境变量
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"

# 可选：启用自动调优
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE="TRUE"

# 可选：手动配置
export FLASH_ATTENTION_FWD_TRITON_AMD_CONFIG_JSON='{"BLOCK_M":128,"BLOCK_N":64,"waves_per_eu":1}'
```

---

## 版本对比总结

| 特性 | FA2 | FA3 | FA4 | ROCm |
|------|-----|-----|-----|------|
| **实现语言** | CUDA C++ | CUDA C++ (CuTe) | Python + CuTeDSL | HIP/Triton |
| **主要目标 GPU** | Ampere/Ada/Hopper | Hopper（初步支持 Blackwell） | Ampere/Hopper/Blackwell | AMD CDNA/RDNA |
| **CUDA 要求** | >= 12.0 | >= 12.3 (推荐 12.8) | CuTeDSL 依赖 | ROCm >= 6.0 |
| **开发状态** | 成熟稳定 | Beta | 活跃开发 | 稳定 |
| **编译方式** | AOT | AOT | JIT | AOT/JIT |
| **自定义性** | 低 | 低 | 高 | 中 |
| **性能** | 优秀 | Hopper 上最优 | 优秀（新架构最优） | 良好 |
| **生产就绪** | ✅ | ⚠️ Beta | ✅ | ✅ |

### 适用场景建议

1. **生产环境，Ampere/Ada GPU**
   - 推荐: FA2
   - 原因: 成熟稳定，广泛验证

2. **H100 推理/训练，追求极致性能**
   - 推荐: FA3
   - 原因: 针对 Hopper 深度优化

3. **H100/B200，需要自定义 attention**
   - 推荐: FA4
   - 原因: Python 可编程，支持 score_mod/mask_mod

4. **Blackwell (B100/B200/GB200)**
   - 推荐: FA4
   - 原因: 对 SM100/110 提供完整特性支持（FA3 仅初步/部分 SM100 支持）

5. **AMD MI300X/RDNA**
   - 必选: ROCm Backend
   - CK Backend: 默认选择
   - Triton Backend: 需要更多特性或调优

---

## 编译架构配置参考

### FA2/FA3 (setup.py)

```bash
# 指定编译架构
export FLASH_ATTN_CUDA_ARCHS="80;90;100;110;120"

# 生成的编译标志
-gencode arch=compute_80,code=sm_80
-gencode arch=compute_90,code=sm_90
-gencode arch=compute_100f,code=sm_100  # CUDA >= 12.9
-gencode arch=compute_110f,code=sm_110  # CUDA >= 13.0
-gencode arch=compute_120f,code=sm_120  # CUDA >= 12.8
```

### FA4 (运行时 JIT)

```bash
# 运行时架构选择（影响内核路径）
export FLASH_ATTENTION_ARCH=sm_100

# JIT 编译目标（影响 PTX/CUBIN 生成）
export CUTE_DSL_ARCH=sm_100

# CPU 编译测试（无 GPU）
export FLASH_ATTENTION_ARCH=sm_80
export CUTE_DSL_ARCH=sm_80
```

---

## 性能基准参考

FlashAttention-3 官方博客（https://tridao.me/blog/2024/flash3/）与论文报告：在 H100 SXM5 上，FA3 前向 FP16 可达约 **740-750 TFLOPS**，相较 FA2（H100 上约 350-500 TFLOPS，未充分利用 Hopper 特性）有显著提升；FP8 前向可接近 **1.2 PFLOPS**。

> ⚠️ 本文档不复制具体的分序列长度基准表——之前版本中按序列长度列出的 TFLOPS 数字是未经核实的估算。请直接参阅上述官方博客/论文的原始图表，或在目标硬件上用 `hopper/benchmark_attn.py` / `flash_attn/cute` 的 benchmark 脚本实测。

**FA4 / Blackwell**: 本仓库未提供可直接引用的官方 Blackwell 基准数字，此处不作量化预测。

---

## 参考资源

### 论文
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://tridao.me/publications/flash3/flash3.pdf)

### 代码与文档
- [FlashAttention (Dao-AILab)](https://github.com/Dao-AILab/flash-attention)
- [FlashAttention-3 博客](https://tridao.me/blog/2024/flash3/)
- [CUTLASS (NVIDIA)](https://github.com/NVIDIA/cutlass)
- [CuTe DSL — NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)

### 调试指南
- `AI/DEBUG_2CTA.md` - 2CTA 内核调试
- `AI/RACECHECK_TMA_HAZARD.md` - TMA 竞态检测
- `AI/CLC_TRACE_DEBUG.md` - CLC 调度可视化

---

**文档版本**: 2026-07-13  
**维护**: 根据代码梳理生成
