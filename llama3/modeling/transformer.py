# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# 
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. 
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement. 

import math
import torch
from typing import Optional, Tuple
from llama3.modeling import ModelArgs
from llama3.modeling import AttentionLayer
from llama3.modeling import build_rope_cache
from llama3.modeling import TokenEmbedding
from llama3.modeling import Linear, FeedForward


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))
    
    def reset_parameters(self):
        with torch.no_grad():
            self.weight.fill_(1.0)
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(torch.nn.Module):
    def __init__(self, args : ModelArgs):
        super().__init__()
        self.attention = AttentionLayer(args.dim, args.n_heads, args.n_kv_heads, args.dropout)
        self.combinator = FeedForward(args.dim, 4 * args.dim, args.multiple_of, args.ffn_dim_multiplier)
        self.ln_attention = RMSNorm(args.dim, eps=args.norm_eps)
        self.ln_combinator = RMSNorm(args.dim, eps=args.norm_eps)
    
    def reset_parameters(self):
        self.attention.reset_parameters()
        self.combinator.reset_parameters()
        self.ln_attention.reset_parameters()
        self.ln_combinator.reset_parameters()
    
    def forward(
        self,
        x: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        y, k_cache, v_cache = self.attention(self.ln_attention(x), k_cache, v_cache, rope_cache, mask)
        h = x + y
        out = h + self.combinator(self.ln_combinator(h))
        return out, k_cache, v_cache


class Transformer(torch.nn.Module):
    def __init__(self, args : ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.max_seq_len = args.max_seq_len
        self.rope_theta = args.rope_theta
        self.use_scaled_rope = args.use_scaled_rope
        
        self.tok_embeddings = TokenEmbedding(args.vocab_size, args.dim)
        self.dropout_embedding = torch.nn.Dropout(args.dropout)
        
        layers = [TransformerBlock(args) for _ in range(args.n_layers)]
        self.layers = torch.nn.ModuleList(layers)
        
        self.ln_head = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = Linear(args.dim, args.vocab_size, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.tok_embeddings.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.ln_head.reset_parameters()
        self.output.reset_parameters()
        self.rope_cache = build_rope_cache(
            self.dim // self.n_heads,
            self.max_seq_len * 2,
            self.rope_theta,
            self.use_scaled_rope,
        )
    
    def get_empty_k_cache(self, batch_size, cache_len):
        shape = (len(self.layers), batch_size, cache_len, self.n_kv_heads, self.dim // self.n_heads)
        device=self.output.weight.device
        dtype=self.output.weight.dtype
        return torch.empty(shape, device=device, dtype=dtype)
    
    def get_empty_v_cache(self, batch_size, cache_len):
        shape = (len(self.layers), batch_size, cache_len, self.n_kv_heads, self.dim // self.n_heads)
        device=self.output.weight.device
        dtype=self.output.weight.dtype
        return torch.empty(shape, device=device, dtype=dtype)
    
    def forward(self, input: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor):
        _, seqlen = input.shape
        h = self.tok_embeddings(input)
        self.rope_cache = self.rope_cache.to(h.device)
        start_pos = k_cache.size(2)
        rope_cache = self.rope_cache[start_pos : start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            # When performing key-value caching, we compute the attention scores 
            # only for the new sequence. Thus, the matrix of scores is of size 
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for 
            # j > cache_len + i, since row i corresponds to token cache_len + i. 
            mask = torch.full((seqlen, seqlen + start_pos), -1e4, device=input.device)
            mask = mask.triu(start_pos + 1).type_as(h)
        
        updated_k_cache = []
        updated_v_cache = []
        for layer, k_c, v_c in zip(self.layers, k_cache, v_cache):
            h, k_c, v_c = layer(h, k_c, v_c, rope_cache, mask)
            updated_k_cache.append(k_c)
            updated_v_cache.append(v_c)
        # tensorify 
        updated_k_cache = torch.stack(updated_k_cache)
        updated_v_cache = torch.stack(updated_v_cache)
        # layer normalization 
        h = self.ln_head(h)
        output = self.output(h)
        return output, updated_k_cache, updated_v_cache

