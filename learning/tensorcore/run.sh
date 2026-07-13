#!/usr/bin/env bash
# Reproduce the TensorCore feasibility probe on this SM120 GPU.
# Each case prints: COMPILE state, and (if built) the on-device RUN state.
set -u
export PATH=/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")"
PY=/home/mabing/workspace/pyvenv/flash-attn/bin/python
sep() { echo "-------------------------------------------------------"; }

echo "### Raw CUDA C++ / inline-PTX cases ###"

sep; echo "[1] Ampere mma.sync  -> compile sm_80, run on SM120"
nvcc -arch=sm_80 -o cuda/mma_sm80 cuda/mma_sync.cu && echo "COMPILE_OK" && ./cuda/mma_sm80

sep; echo "[2] SM120 native mma.sync -> compile sm_120a, run on SM120"
nvcc -arch=sm_120a -o cuda/mma_sm120 cuda/mma_sync.cu && echo "COMPILE_OK" && ./cuda/mma_sm120

sep; echo "[3] Hopper wgmma -> compile sm_90a (SASS-only), run on SM120"
nvcc -gencode arch=compute_90a,code=sm_90a -o cuda/wgmma_sm90a cuda/wgmma.cu && echo "COMPILE_OK" && ./cuda/wgmma_sm90a
echo "    (try compiling wgmma for sm_120a:)"
nvcc -arch=sm_120a -o /dev/null cuda/wgmma.cu 2>&1 | grep -m1 "not supported" || echo "    unexpectedly compiled"

sep; echo "[4] Blackwell tcgen05 -> compile sm_100a (SASS-only), run on SM120"
nvcc -gencode arch=compute_100a,code=sm_100a -o cuda/tcgen05_sm100 cuda/tcgen05.cu && echo "COMPILE_OK" && ./cuda/tcgen05_sm100
echo "    (try compiling tcgen05 for sm_120a:)"
nvcc -arch=sm_120a -o /dev/null cuda/tcgen05.cu 2>&1 | grep -m1 "not supported" || echo "    unexpectedly compiled"

echo; echo "### CuTeDSL / FA4 cases ###"
sep; echo "[5] FA4 forward on native SM120 path"
$PY cute_fa4_sm120.py 2>&1 | grep -E "device arch|RESULT|max_err"
sep; echo "[6] Force FA4 SM90 (wgmma) and SM100 (tcgen05) paths on SM120"
for A in 90a 100a; do $PY cute_force_arch.py $A 2>&1 | grep -E "^RESULT"; done
sep
