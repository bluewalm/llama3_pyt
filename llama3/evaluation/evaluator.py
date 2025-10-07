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
import tqdm
from typing import Dict, Optional
# 
from llama3.data import Dataloader
from llama3.utils import Recorder, print_model, pprint, cycle


class Evaluator:
    def __init__(self, specification):
        self.specification = specification
    
    def checkpoint_name(self, checkpoint):
        path, filename = os.path.split(checkpoint)
        root, ext = os.path.splitext(filename)
        ckpt_model = root + "_model" + ext
        ckpt_model = os.path.join(path, ckpt_model)
        return ckpt_model
    
    def evaluate(self, from_checkpoint: Optional[str] = None):
        # prepare tokenizer 
        self.specification.prepare_tokenizer()
        
        # prepare loss function 
        self.specification.prepare_loss_function()
        
        # prepare dataset 
        eval_dataset = self.specification.prepare_dataset()
        
        # prepare dataloader 
        eval_dataloader = self.specification.prepare_dataloader(eval_dataset, self.specification.batch_size)
        
        # cycle through them in an infinite loop 
        if 0 < self.specification.total_steps:
            # cycle through them in an infinite loop 
            eval_dataloader = cycle(eval_dataloader)
        else:
            self.specification.total_steps = float('inf')
        
        # construct a model 
        model = self.specification.construct_model().cuda()
        
        # prepare the caches 
        self.specification.prepare_empty_caches(model)
        
        # creates the recorder 
        recorder = Recorder()
        
        start_step = 0
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
        elif self.specification.amp_bf16 or self.specification.bf16:
            model.bfloat16()
        
        # allow tf32 
        torch.backends.cudnn.allow_tf32 = bool(self.specification.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.specification.allow_tf32)
        
        # print the model 
        print_model(model)
        
        # create tqdm iterator in master process to show the progress of training 
        rng = range(start_step + 1, self.specification.total_steps + 1)
        total = self.specification.total_steps
        desc = self.specification.description
        pbar = tqdm.tqdm(rng, total=total, desc=desc, dynamic_ncols=True)
        pbar.update(start_step)
        for step in pbar:
            if step > self.specification.total_steps:
                break
            # clear CUDA cache which is used for evaluation 
            torch.cuda.empty_cache()
            # make a measurement on a single batch 
            try:
                recorder.record(self._eval_step(eval_dataloader, model), step=step, scope='eval')
            except StopIteration:
                break
        # close progress bar 
        pbar.close()
        # print the accuracy 
        metrics = recorder.averages
        pprint("average accuracy over eval dataset", **metrics)
    
    @torch.inference_mode()
    def _eval_step(self, dataloader: Dataloader, model: torch.nn.Module) -> Dict[str, float]:
        # fetch the data 
        data = next(dataloader)
        data = {k : v.cuda() for k, v in data.items()}
        # run the model on the data and get the metrics 
        model.eval()
        metrics = self.specification.eval_objective(data, model)
        return {k: v.item() for k, v in metrics.items()}

