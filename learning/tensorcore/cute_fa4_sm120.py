"""CuTeDSL side of the TensorCore feasibility probe.

FA4 dispatches the MMA generation by SM (interface.py):
  SM90        -> wgmma (warpgroup MMA)      via FlashAttentionForwardSm90
  SM100/110   -> tcgen05 (UMMA + TMEM)      via FlashAttentionForwardSm100
  SM80/SM120  -> SM80 mma.sync (portable)   via flash_fwd base / *_sm120

This script runs the SM120 native path end-to-end and checks correctness,
proving the SM120 TensorCore path is feasible through CuTeDSL on this GPU.
"""
import torch
from flash_attn.cute.interface import _get_device_arch, flash_attn_func

arch = _get_device_arch()
print(f"device arch = {arch}  ({torch.cuda.get_device_name(0)})")
print(f"selected FA4 forward class family = SM{ (arch//10)*10 }")

torch.manual_seed(0)
b, s, h, d = 2, 512, 8, 64
q = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
k = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)
v = torch.randn(b, s, h, d, device="cuda", dtype=torch.bfloat16)

out = flash_attn_func(q, k, v, causal=True)
if isinstance(out, (tuple, list)):
    out = out[0]  # (out, lse, ...) -> out
torch.cuda.synchronize()

# reference via PyTorch SDPA
qt, kt, vt = (x.transpose(1, 2) for x in (q, k, v))
ref = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=True)
ref = ref.transpose(1, 2)

max_err = (out.float() - ref.float()).abs().max().item()
print(f"FA4 forward on SM{arch}: out shape {tuple(out.shape)}, max_err vs SDPA = {max_err:.4g}")
print("RESULT:", "CORRECT" if max_err < 2e-2 else f"WRONG (max_err={max_err})")
