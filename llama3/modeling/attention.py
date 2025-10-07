# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# 
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. 
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
import torch
from typing import Optional
from bluewalm.softplus_attention import QueryProjection, KeyProjection, ValueProjection, OutProjection, attention_operator
from llama3.modeling import apply_rotary_emb


class AttentionLayer(torch.nn.Module):
    def __init__(self, dim: int, core_dim: int, dropout: float):
        super().__init__()
        self.core_dim = core_dim
        # assert dim % n_heads == 0
        self.wq = QueryProjection(dim, core_dim)
        self.wk = KeyProjection(dim, core_dim)
        self.wv = ValueProjection(dim, core_dim)
        self.wo = OutProjection(dim, core_dim)
        self.dropout = torch.nn.Dropout(dropout)
    
    def reset_parameters(self):
        self.wq.reset_parameters()
        self.wk.reset_parameters()
        self.wv.reset_parameters()
        self.wo.reset_parameters()
    
    def forward(self,
              x: torch.Tensor,
              k_cache: torch.Tensor,
              v_cache: torch.Tensor,
              rope_cache: torch.Tensor):
        
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq, xk = apply_rotary_emb(xq, xk, rope_cache=rope_cache)
        
        # reuse attention keys and values by concatenating to the current ones 
        xk = torch.cat((k_cache, xk), dim=2)
        xv = torch.cat((v_cache, xv), dim=2)
        
        # xq, xk and xv are contiguous here 
        scores = attention_operator(xq, xk, xv)
        scores = self.dropout(scores)
        
        output = self.wo(scores)
        return output, xk, xv

