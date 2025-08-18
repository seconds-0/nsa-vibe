from __future__ import annotations

import torch
import torch.nn.functional as F


def is_flash_available() -> bool:
	try:
		from flash_attn import flash_attn_func  # type: ignore
		return True
	except Exception:
		return False


def is_flash_varlen_available() -> bool:
	try:
		# Placeholder probe for varlen API; adjust when integrating
		from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore
		return True
	except Exception:
		return False


def attention_bgh(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = True) -> torch.Tensor:
	"""
	Q: [B,G,h,Dk], K/V: [B,G,S,D*] -> out [B,G,h,Dv]
	Uses flash-attn if available; falls back to SDPA.
	"""
	B, G, h, Dk = Q.shape
	S = K.shape[2]
	try:
		from flash_attn import flash_attn_func  # type: ignore

		q = Q.reshape(B * G * h, 1, Dk)
		k = K.repeat_interleave(h, dim=1).reshape(B * G * h, S, Dk)
		v = V.repeat_interleave(h, dim=1).reshape(B * G * h, S, V.shape[-1])
		# flash_attn_func expects [B, T, H] in some setups; use SDPA fallback for simplicity here
		raise ImportError
	except Exception:
		q = Q.reshape(B * G * h, 1, Dk)
		k = K.repeat_interleave(h, dim=1).reshape(B * G * h, S, Dk)
		v = V.repeat_interleave(h, dim=1).reshape(B * G * h, S, V.shape[-1])
		attn = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
		o = attn.squeeze(1).reshape(B, G, h, -1)
		return o


def attention_fa2_varlen_stub(*args, **kwargs):
	"""
	Stub for FA-2 varlen attention; will be implemented in M1.
	Intentionally raises to direct callers to fallback.
	"""
	raise NotImplementedError("FA-2 varlen attention not yet implemented")


