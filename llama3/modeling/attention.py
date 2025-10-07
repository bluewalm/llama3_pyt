# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# 
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. 
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement. 

import math
import torch
from typing import Optional
from llama3.modeling import Linear
from llama3.modeling import apply_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int, n_heads: int) -> torch.Tensor:
    return torch.repeat_interleave(x, dim=2, repeats=n_rep, output_size=n_heads)


class AttentionLayer(torch.nn.Module):
    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, dropout: float):
        super().__init__()
        n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        self.n_kv_heads = n_kv_heads
        self.n_heads = n_heads
        self.n_rep = n_heads // n_kv_heads
        assert dim % n_heads == 0
        self.head_dim = dim // n_heads
        self.wq = Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = Linear(n_heads * self.head_dim, dim, bias=False)
        self.dropout = dropout
    
    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()
    
    def forward(self,
              x: torch.Tensor,
              k_cache: torch.Tensor,
              v_cache: torch.Tensor,
              rope_cache: torch.Tensor,
              mask: Optional[torch.Tensor]):
        
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, rope_cache=rope_cache)
        
        xk = torch.cat((k_cache, xk), dim=1)
        xv = torch.cat((v_cache, xv), dim=1)
        
        # repeat k/v heads if n_kv_heads < n_heads 
        keys = repeat_kv(
            xk, self.n_rep, self.n_heads
        )  # (bs, cache_len + seqlen, n_heads, self.head_dim) 
        values = repeat_kv(
            xv, self.n_rep, self.n_heads
        )  # (bs, cache_len + seqlen, n_heads, self.head_dim) 
        
        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, self.head_dim) 
        keys = keys.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, self.head_dim) 
        values = values.transpose(1, 2)  # (bs, n_heads, cache_len + seqlen, self.head_dim) 
        dropout = self.dropout if self.training else 0.0
        output = torch.nn.functional.scaled_dot_product_attention(
                                                xq, keys, values,
                                                mask, dropout_p=dropout)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        output = self.wo(output)
        return output, xk, xv

