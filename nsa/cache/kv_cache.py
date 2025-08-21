from dataclasses import dataclass

import torch

from nsa.core.block_index import BlockMeta


@dataclass
class NSA_KV:
    K_sel: torch.Tensor  # [B,G,S,Dk]
    V_sel: torch.Tensor  # [B,G,S,Dv]
    K_win: torch.Tensor  # [B,G,S,Dk]
    V_win: torch.Tensor  # [B,G,S,Dv]
    # raw token-level seq for compressed branch
    K_cmp_raw_seq: torch.Tensor  # [B,G,S,Dk]
    V_cmp_raw_seq: torch.Tensor  # [B,G,S,Dv]
    K_cmp: torch.Tensor  # [B,G,S_cmp,Dk]
    V_cmp: torch.Tensor  # [B,G,S_cmp,Dv]
    win_ptr: torch.Tensor  # [B,G]
    cmp_emit_next: torch.Tensor  # [B,G]
    meta: BlockMeta
    reads_pred: torch.Tensor  # [T] per decode step predicted total reads
    reads_act_total: torch.Tensor  # [T]
    reads_act_sel: torch.Tensor  # [T]
    reads_act_cmp: torch.Tensor  # [T]
    reads_act_win: torch.Tensor  # [T]

    def update_selection_raw(self, K: torch.Tensor, V: torch.Tensor) -> None:
        self.K_sel = torch.cat([self.K_sel, K], dim=2)
        self.V_sel = torch.cat([self.V_sel, V], dim=2)

    def update_window(self, K: torch.Tensor, V: torch.Tensor, w: int) -> None:
        self.K_win = torch.cat([self.K_win, K], dim=2)
        self.V_win = torch.cat([self.V_win, V], dim=2)
        # keep last w tokens
        if self.K_win.shape[2] > w:
            self.K_win = self.K_win[:, :, -w:, :]
            self.V_win = self.V_win[:, :, -w:, :]

    def update_compressed(
        self, K_raw_cmp: torch.Tensor, V_raw_cmp: torch.Tensor, l: int, d: int
    ) -> None:
        # M0 prefill path: rebuild fully using avg-pool ϕ handled upstream
        self.K_cmp = K_raw_cmp
        self.V_cmp = V_raw_cmp

    def append_cmp_raw(self, K_raw_tok: torch.Tensor, V_raw_tok: torch.Tensor) -> None:
        self.K_cmp_raw_seq = torch.cat([self.K_cmp_raw_seq, K_raw_tok], dim=2)
        self.V_cmp_raw_seq = torch.cat([self.V_cmp_raw_seq, V_raw_tok], dim=2)

    def append_reads_pred(self, value: int) -> None:
        v = torch.tensor([value], dtype=torch.int64, device=self.K_sel.device)
        self.reads_pred = torch.cat([self.reads_pred, v], dim=0) if self.reads_pred.numel() else v

    def append_reads_actual(self, total: int, sel: int, cmp: int, win: int) -> None:
        dev = self.K_sel.device

        def cat_or_set(t: torch.Tensor, val: int) -> torch.Tensor:
            v = torch.tensor([val], dtype=torch.int64, device=dev)
            return torch.cat([t, v], dim=0) if t.numel() else v

        self.reads_act_total = cat_or_set(self.reads_act_total, total)
        self.reads_act_sel = cat_or_set(self.reads_act_sel, sel)
        self.reads_act_cmp = cat_or_set(self.reads_act_cmp, cmp)
        self.reads_act_win = cat_or_set(self.reads_act_win, win)
