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

import torch


class TokenEmbedding(torch.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    
    def reset_parameters(self):
        with torch.no_grad():
            self.weight.normal_(0.0, 0.02)
    
    def __repr__(self):
        return "TokenEmbedding(" + self.extra_repr() + ")"
    
    def extra_repr(self):
        s = 'vocab_size={num_embeddings}, dim={embedding_dim}'
        precision = {torch.float32 : 'fp32', torch.float16 : 'fp16', torch.bfloat16 : 'bf16'}
        precision = precision[self.weight.dtype]
        s += ', precision={precision}'
        size = self.weight.nelement() * self.weight.element_size()
        size = round(size / 1024**2, 4)
        s += ', size=' + str(size) + ' MB'
        return s.format(**self.__dict__, precision=precision)

