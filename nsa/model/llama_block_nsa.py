import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nsa.core.nsa_attention import NSAAttention
from nsa.cache.kv_cache import NSA_KV
from nsa.core.block_index import build_block_meta


class RMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6) -> None:
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B,S,dim]
		rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
		return (x * rms) * self.weight


class MLP(nn.Module):
	def __init__(self, dim: int, hidden_mult: int = 4) -> None:
		super().__init__()
		h = hidden_mult * dim
		self.fc1 = nn.Linear(dim, h, bias=False)
		self.fc2 = nn.Linear(h, dim, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.fc2(F.silu(self.fc1(x)))


class LlamaBlockNSA(nn.Module):
	def __init__(
		self,
		dim: int,
		n_heads: int,
		n_kv_groups: int,
		d_k: int,
		d_v: int,
		l: int = 32,
		d: int = 16,
		l_sel: int = 64,
		n_sel: int = 16,
		w: int = 512,
	) -> None:
		super().__init__()
		self.norm1 = RMSNorm(dim)
		self.attn = NSAAttention(
			dim=dim,
			n_heads=n_heads,
			n_kv_groups=n_kv_groups,
			d_k=d_k,
			d_v=d_v,
			l=l,
			d=d,
			l_sel=l_sel,
			n_sel=n_sel,
			w=w,
		)
		self.norm2 = RMSNorm(dim)
		self.mlp = MLP(dim)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B,S,dim]
		B, S, dim = x.shape
		res = x
		xn = self.norm1(x)
		# Build empty KV cache for prefill
		device = x.device
		G = self.attn.n_kv_groups
		Dk = self.attn.d_k
		Dv = self.attn.d_v
		zeros_k = torch.zeros((B, G, 0, Dk), device=device, dtype=x.dtype)
		zeros_v = torch.zeros((B, G, 0, Dv), device=device, dtype=x.dtype)
		meta = build_block_meta(seq_len=0, l=self.attn.l, d=self.attn.d, l_sel=self.attn.l_sel, n_sel=self.attn.n_sel, w=self.attn.w)
		kv = NSA_KV(
			K_sel=zeros_k.clone(),
			V_sel=zeros_v.clone(),
			K_win=zeros_k.clone(),
			V_win=zeros_v.clone(),
			K_cmp_raw_seq=zeros_k.clone(),
			V_cmp_raw_seq=zeros_v.clone(),
			K_cmp=zeros_k.clone(),
			V_cmp=zeros_v.clone(),
			win_ptr=torch.zeros((B, G), dtype=torch.int64, device=device),
			cmp_emit_next=torch.zeros((B, G), dtype=torch.int64, device=device),
			meta=meta,
			reads_pred=torch.zeros((0,), dtype=torch.int64, device=device),
			reads_act_total=torch.zeros((0,), dtype=torch.int64, device=device),
			reads_act_sel=torch.zeros((0,), dtype=torch.int64, device=device),
			reads_act_cmp=torch.zeros((0,), dtype=torch.int64, device=device),
			reads_act_win=torch.zeros((0,), dtype=torch.int64, device=device),
		)
		out, _kv = self.attn(xn, kv=kv, prefill=True)
		x = res + out
		res2 = x
		x = self.mlp(self.norm2(x))
		return res2 + x


class _EmptyKVLike:
	pass

