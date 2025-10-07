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

import os
import torch
from typing import List, Optional
# 
from llama3.utils import print_model


class Generator:
    def __init__(self, specification):
        self.specification = specification
        self.specification.batch_size = 1
    
    def checkpoint_name(self, checkpoint):
        path, filename = os.path.split(checkpoint)
        root, ext = os.path.splitext(filename)
        ckpt_model = root + "_model" + ext
        ckpt_model = os.path.join(path, ckpt_model)
        return ckpt_model
    
    def initialize(self, from_checkpoint: Optional[str] = None):
        # prepare tokenizer 
        self.specification.prepare_tokenizer()
        
        # construct a model 
        model = self.specification.construct_model().cuda()
        
        # prepare the caches 
        self.specification.prepare_empty_caches(model)
        
        # restore last training states from checkpoint 
        if from_checkpoint:
            ckpt_model = self.checkpoint_name(from_checkpoint)
            
            model_state_dict = torch.load(ckpt_model, mmap=True, weights_only=True, map_location='cuda')
            model.load_state_dict(model_state_dict)
            
            # because the checkpoint data allocates quite a lot of GPU 
            # memory, we need to free the memories explicitly 
            del model_state_dict
            torch.cuda.empty_cache()
        
        # cast the model 
        if self.specification.amp_fp16:
            model.half()
        if self.specification.amp_bf16 or self.specification.bf16:
            model.bfloat16()
        
        # allow tf32 
        torch.backends.cudnn.allow_tf32 = bool(self.specification.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.specification.allow_tf32)
        
        model.eval()
        
        # print the model 
        print_model(model)
        
        self.model = model
    
    def generate(self, context: str) -> str:
        words = self.specification.encode_context(context)
        current = words
        k_cache = self.specification.k_cache()
        v_cache = self.specification.v_cache()
        while len(words) < self.specification.max_seq_len:
            # predict the next word token from the given context 
            probs, k_cache, v_cache = self._predict_probs(current, k_cache, v_cache)
            next_word = self._sample_from_top_p(probs)
            # change the context to the predicted word 
            words.append(next_word)
            current = [next_word]
        decoded_tokens = self.specification.decode_tokens(words)
        return decoded_tokens
    
    @torch.inference_mode()
    def _predict_probs(self,
                    words: List[int],
                    k_cache: torch.Tensor,
                    v_cache: torch.Tensor):
        # put it into tensor 
        x = torch.tensor(words, dtype=torch.int64, device='cuda')
        # add dim for batch_size=1 
        x = x.unsqueeze(0)
        # run inference 
        logits, k_cache, v_cache = self.model(x, k_cache, v_cache)
        # convert to float (in case it is half) 
        logits = logits.float()
        logits = logits[0, -1, :].softmax(-1)
        return logits, k_cache, v_cache
    
    def _sample_from_top_p(self, probs: torch.Tensor) -> int:
        probs, indices = probs.sort(descending=True)
        mask = probs.cumsum(-1) > self.specification.nucleus_prob
        mask[0] = False
        probs.masked_fill_(mask, 0)
        # sample from filtered distribution 
        return indices[probs.multinomial(1)[0]].item()

