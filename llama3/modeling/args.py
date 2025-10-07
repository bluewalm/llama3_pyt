# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 

from typing import Optional
from dataclasses import dataclass, fields


@dataclass
class ModelArgs:
    dim: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    multiple_of: int
    ffn_dim_multiplier: float
    dropout: float
    vocab_size: int
    
    def __init__(self, **kwargs):
        field_names = [field.name for field in fields(self)]
        for k, v in kwargs.items():
            if k in field_names:
                setattr(self, k, v)
        # set the defaults (important) 
        if getattr(kwargs, 'dropout', None) is None:
            setattr(self, 'dropout', 0.1)
        if getattr(kwargs, 'n_kv_heads', None) is None:
            setattr(self, 'n_kv_heads', self.n_heads)
        if getattr(kwargs, 'n_kv_heads', None) is None:
            setattr(self, 'n_kv_heads', self.n_heads)
        if getattr(kwargs, 'multiple_of', None) is None:
            setattr(self, 'multiple_of', 256)  # make SwiGLU hidden layer size multiple of large power of 2 
        if getattr(kwargs, 'ffn_dim_multiplier', None) is None:
            setattr(self, 'ffn_dim_multiplier', None)
        # asserts 
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0

