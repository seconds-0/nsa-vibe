import torch
from nsa.attn.fa2_contracts import check_cu_seqlens


def validate_varlen_packing(
    qkv_unpad: torch.Tensor,
    cu: torch.Tensor,
    B: int,
    max_seqlen: int,
    expect_packed_dim: int = 3,
) -> bool:
    assert qkv_unpad.is_cuda, "varlen path is CUDA-only"
    assert qkv_unpad.dtype in (torch.float16, torch.bfloat16)
    assert qkv_unpad.is_contiguous(), "qkv_unpad must be contiguous"
    nnz = qkv_unpad.shape[0]
    assert check_cu_seqlens(cu, nnz, B), "invalid cu_seqlens"
    assert max_seqlen > 0 and max_seqlen <= nnz, "max_seqlen out of range"
    if qkv_unpad.dim() == 4:
        assert (
            qkv_unpad.shape[1] == 3
        ), "need qkv-packed (nnz,3,H,D) for qkvpacked kernels"
    return True

