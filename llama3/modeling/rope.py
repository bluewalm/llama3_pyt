# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# 
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. 
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
import torch
from typing import Tuple


def apply_scaling(freqs: torch.Tensor, max_seq_len: int):
    # values obtained from grid search 
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    
    low_freq_wavelen = max_seq_len / low_freq_factor
    high_freq_wavelen = max_seq_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (max_seq_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def build_rope_cache(core_dim: int, max_seq_len: int, theta: float = 10000.0, use_scaled: bool = False):
    """ builds rope cache """
    theta = 1.0 / (theta ** (torch.arange(0, core_dim, 2)[: (core_dim // 2)].float() / core_dim))
    seq_idx = torch.arange(max_seq_len, device=theta.device, dtype=theta.dtype)
    if use_scaled:
        freqs = apply_scaling(theta, max_seq_len)
    freqs = torch.outer(seq_idx, theta)
    # rope_cache includes both the cos and sin components and so the output shape is
    # [max_seq_len, core_dim // 2, 2]
    rope_cache = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    return rope_cache


def apply_rotation(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input tensor with shape
                [b, s, core_dim]
            rope_cache (Tensor): rope cache with shape
                [s, core_dim // 2, 2]
                assumed to be the appropriate segment 
                of the rope cache starting from position zero 
                going up to maximal sequence length 
        
        Returns:
            Tensor: output tensor with RoPE applied
        
        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - core_dim: the core dimension of the attention layer
        """
        # input tensor has shape [b, s, core_dim]
        bsz, seq_len, core_dim = x.shape
        
        # extract relevant values from cache
        # rope_cache = rope_cache[:seq_len]
        
        # reshape input; the last dimension is used for computing the output.
        # tensor has shape [b, s, core_dim // 2, 2]
        xshaped = x.reshape(bsz, seq_len, -1, 2)
        
        # reshape the cache for broadcasting
        # tensor has shape [b, s, core_dim // 2, 2] if packed samples,
        # otherwise has shape [1, s, core_dim // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), xshaped.size(2), 2)
        # move rope cache to the same device and dtype as x 
        dtype = x.dtype
        device = x.device
        rope_cache = rope_cache.to(dtype=dtype, device=device)
        
        # tensor has shape [b, s, core_dim // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        
        # tensor has shape [b, s, core_dim]
        x_out = x_out.view(bsz, seq_len, core_dim)  # flatten(2)
        return x_out


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ call this with the corresponding attention values and rope cache """
    xq = apply_rotation(xq, rope_cache)
    xk = apply_rotation(xk.transpose(2,1), rope_cache).transpose(2,1)
    return xq, xk

