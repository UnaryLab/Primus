###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import torch
from torch.nn.attention.flex_attention import BlockMask
from torchtitan.models.llama3.model.model import Attention as TTAttention
from torchtitan.models.llama3.model.model import apply_rotary_emb

AttentionMasksType = dict[str, BlockMask] | BlockMask


class Attention(TTAttention):
    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
    ):
        bs, seqlen, _ = x.shape
        with torch.autograd.profiler.record_function("q_ip"):
            xq = self.wq(x)
        with torch.autograd.profiler.record_function("k_ip"):
            xk = self.wk(x)
        with torch.autograd.profiler.record_function("v_ip"):
            xv = self.wv(x)

        with torch.autograd.profiler.record_function("qkv_t"):
            xq = xq.view(bs, seqlen, -1, self.head_dim)
            xk = xk.view(bs, seqlen, -1, self.head_dim)
            xv = xv.view(bs, seqlen, -1, self.head_dim)

        with torch.autograd.profiler.record_function("qkv_re"):
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        with torch.autograd.profiler.record_function("attn_fa"):
            output = self.inner_attention(xq, xk, xv)

        with torch.autograd.profiler.record_function("attn_or"):
            output = output.contiguous().view(bs, seqlen, -1)
        with torch.autograd.profiler.record_function("attn_op"):
            return self.wo(output)
