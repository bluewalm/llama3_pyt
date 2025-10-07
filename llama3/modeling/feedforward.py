# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# 
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved. 
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement. 

import math
import torch
import torch.nn.functional as F
from typing import Optional


class FLinear(torch.nn.Linear):
    def __repr__(self):
        return "Linear(" + self.extra_repr() + ")"
    
    def extra_repr(self):
        s = 'in_features={in_features}, out_features={out_features}'
        precision = {torch.float32 : 'fp32', torch.float16 : 'fp16', torch.bfloat16 : 'bf16'}
        precision = precision[self.weight.dtype]
        s += ', precision={precision}'
        size = self.weight.nelement() * self.weight.element_size()
        size = round(size / 1024**2, 4)
        s += ', size=' + str(size) + ' MB'
        return s.format(**self.__dict__, precision=precision)


class Linear(FLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            self.weight.normal_(0.0, 0.02)


class FeedForward(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        self.w1 = FLinear(dim, hidden_dim, bias=False)
        self.w2 = FLinear(hidden_dim, dim, bias=False)
        self.w3 = FLinear(dim, hidden_dim, bias=False)
    
    def reset_parameters(self):
        self.w1.reset_parameters()
        self.w2.reset_parameters()
        self.w3.reset_parameters()
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

