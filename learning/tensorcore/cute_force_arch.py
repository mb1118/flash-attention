"""Force FA4's per-arch CuTeDSL kernels onto this SM120 GPU via FLASH_ATTENTION_ARCH.

  arch=90a  -> SM90 path  (wgmma / warpgroup MMA)
  arch=100a -> SM100 path (tcgen05 / UMMA + TMEM)

Both emit TensorCore instructions that do not exist on SM120, so compilation
or launch should fail. Run once per arch (set by argv) in a fresh process.
"""
import os, sys, traceback

force = sys.argv[1]                       # e.g. "90a" or "100a"
os.environ["FLASH_ATTENTION_ARCH"] = force

import torch
from flash_attn.cute.interface import flash_attn_func

b, s, h, d = 1, 256, 4, 64
q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)

print(f"=== forcing FLASH_ATTENTION_ARCH={force} on real SM120 ===")
try:
    out = flash_attn_func(q, k, v, causal=True)
    if isinstance(out, (tuple, list)):
        out = out[0]
    torch.cuda.synchronize()
    print(f"RESULT arch={force}: UNEXPECTED SUCCESS, out shape {tuple(out.shape)}")
except Exception as e:
    msg = str(e).strip().splitlines()
    head = " | ".join(msg[:3]) if msg else repr(e)
    print(f"RESULT arch={force}: FAILED ({type(e).__name__}): {head[:300]}")
