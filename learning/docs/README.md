# 学习笔记

本目录收录围绕 flash-attention 的学习笔记。

## 文档索引

- [architecture-overview.md](architecture-overview.md) — Flash Attention 各版本(FA2/FA3/FA4/ROCm)实现与 GPU 平台支持全景梳理。
- [gpu-architecture-evolution.md](gpu-architecture-evolution.md) — NVIDIA GPU 架构演进(Ampere → Hopper → Blackwell)深度分析。
- [tensorcore-feasibility-sm120.md](tensorcore-feasibility-sm120.md) — 各代 TensorCore(mma.sync / wgmma / tcgen05)在本机 SM120 上的可行性实测。

## 仓库远端

本仓库 fork 自上游,远端配置如下:

| 远端 | 地址 | 用途 |
| --- | --- | --- |
| `origin` | `git@github.com:mb1118/flash-attention.git` | 个人 fork,日常推送 |
| `upstream` | `https://github.com/Dao-AILab/flash-attention.git` | 官方上游,只读同步 |

首次配置 upstream:

```bash
git remote add upstream https://github.com/Dao-AILab/flash-attention.git
```

## 同步上游 main

拉取上游最新提交并快进本地 `main`:

```bash
git fetch upstream
git checkout main
git merge --ff-only upstream/main
```

`--ff-only` 保证只在能快进时更新,避免在 `main` 上产生额外的合并提交;若失败说明本地 `main` 有分叉提交,需要先处理。

将同步后的 `main` 推到自己的 fork:

```bash
git push origin main
```

同步前可先确认领先/落后情况(输出为 `落后<TAB>领先`,`0	0` 表示已是最新):

```bash
git fetch upstream
git rev-list --left-right --count main...upstream/main
```
