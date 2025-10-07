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
from datetime import datetime
from typing import Dict, Optional
# 
from llama3.data import Dataloader
from llama3.utils import Recorder, print_model, print_memory_stats, cycle


class Trainer:
    def __init__(self, specification):
        self.specification = specification
        self.dtype, self.enabled = None, False
        if self.specification.amp_fp16:
            self.dtype, self.enabled = torch.float16, True
        if self.specification.amp_bf16:
            self.dtype, self.enabled = torch.bfloat16, True
    
    def checkpoint_names(self, checkpoint):
        path, filename = os.path.split(checkpoint)
        root, ext = os.path.splitext(filename)
        ckpt_model = root + "_model" + ext
        ckpt_model = os.path.join(path, ckpt_model)
        ckpt_optimizer = root + "_optimizer" + ext
        ckpt_optimizer = os.path.join(path, ckpt_optimizer)
        ckpt_other = root + "_other" + ext
        ckpt_other = os.path.join(path, ckpt_other)
        return ckpt_model, ckpt_optimizer, ckpt_other
    
    def train(self, from_checkpoint: Optional[str] = None):
        # prepare rng 
        if self.specification.seed is not None:
            torch.manual_seed(self.specification.seed)
            # torch.use_deterministic_algorithms(True)
        
        # prepare tokenizer 
        self.specification.prepare_tokenizer()
        
        # prepare loss function 
        self.specification.prepare_loss_function()
        
        # prepare datasets 
        train_dataset, eval_dataset = self.specification.prepare_datasets()
        
        # prepare dataloaders 
        train_dataloader, eval_dataloader = self.specification.prepare_dataloaders(train_dataset, eval_dataset, self.specification.batch_size)
        
        # cycle through them in an infinite loop 
        train_dataloader = cycle(train_dataloader)
        eval_dataloader = cycle(eval_dataloader)
        
        # construct a model 
        model = self.specification.construct_model().cuda()
        
        # prepare the caches 
        self.specification.prepare_empty_caches(model)
        
        # create an optimizer and learning rate scheduler 
        optimizer, scheduler = self.specification.create_optimizer(model.parameters())
        # create the GradScaler once at the beginning of training 
        scaler = torch.amp.GradScaler('cuda', enabled=self.specification.amp_fp16)
        # creates the recorder 
        recorder = Recorder()
        
        start_step = 0
        # restore last training states from checkpoint 
        if from_checkpoint:
            ckpt_model, ckpt_optimizer, ckpt_other = self.checkpoint_names(from_checkpoint)
            
            model_state_dict = torch.load(ckpt_model, mmap=True, weights_only=True, map_location='cuda')
            model.load_state_dict(model_state_dict)
            
            optim_state_dict = torch.load(ckpt_optimizer, mmap=True, weights_only=True, map_location='cuda')
            optimizer.load_state_dict(optim_state_dict)
            
            other_state_dict = torch.load(ckpt_other, mmap=True, weights_only=False, map_location='cuda')
            start_step = other_state_dict['step']
            recorder = other_state_dict['recorder']
            scheduler.load_state_dict(other_state_dict['scheduler'])
            scaler.load_state_dict(ckpt['scaler'])
            
            # because the checkpoint data allocates quite a lot of GPU 
            # memory, we need to free the memories explicitly 
            del model_state_dict
            del optim_state_dict
            del other_state_dict
            torch.cuda.empty_cache()
        
        # cast the model when bfloat16 precision flag is set 
        if self.specification.bf16:
            model.bfloat16()
        
        # allow tf32 
        torch.backends.cudnn.allow_tf32 = bool(self.specification.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.specification.allow_tf32)
        
        # print the model 
        print_model(model)
        
        # measure time
        start_time = datetime.now()
        
        # create tqdm iterator in master process to show the progress of training 
        rng = range(start_step + 1, self.specification.total_steps + 1)
        total = self.specification.total_steps
        desc = self.specification.description
        pbar = tqdm.tqdm(rng, total=total, desc=desc, dynamic_ncols=True)
        pbar.update(start_step)
        for step in pbar:
            # clear CUDA cache which is used for training 
            torch.cuda.empty_cache()
            
            recorder.record(self._train_step(train_dataloader, model, optimizer, scheduler, scaler, step), step=step, scope='train')
            
            # clear CUDA cache which is used for evaluation 
            torch.cuda.empty_cache()
            
            if step % self.specification.eval_steps == 0:
                recorder.record(self._eval_step(eval_dataloader, model), step=step, scope='eval')
                
                # set postfix str of the progress bar 
                postfix_str = recorder.format(self.specification.log_format)
                pbar.set_postfix_str(postfix_str)
            
            # save training states to checkpoint file 
            if step % self.specification.save_steps == 0:
                ckpt_model, ckpt_optimizer, ckpt_other = self.checkpoint_names(self.specification.to_checkpoint)
                
                model_state_dict = model.state_dict()
                
                optim_state_dict = optimizer.state_dict()
                
                other_state_dict = {'step' : step, 
                                    'recorder' : recorder, 
                                    'scheduler' : scheduler.state_dict(), 
                                    'scaler' : scaler.state_dict()}
                
                torch.save(model_state_dict, ckpt_model)
                torch.save(optim_state_dict, ckpt_optimizer)
                torch.save(other_state_dict, ckpt_other)
                
                # because the checkpoint data allocates quite a lot of GPU 
                # memories, we free the memory explicitly 
                del model_state_dict
                del optim_state_dict
                del other_state_dict
                torch.cuda.empty_cache()
        
        # save trained model weights and metrics recorded during the training 
        ckpt_model, ckpt_optimizer, ckpt_other = self.checkpoint_names(self.specification.to_checkpoint)
        
        model_state_dict = model.state_dict()
        
        optim_state_dict = optimizer.state_dict()
        
        other_state_dict = {'step' : step,
                            'recorder' : recorder,
                            'scheduler' : scheduler.state_dict(), 
                            'scaler' : scaler.state_dict()}
        
        torch.save(model_state_dict, ckpt_model)
        torch.save(optim_state_dict, ckpt_optimizer)
        torch.save(other_state_dict, ckpt_other)
        
        # close progress bar 
        pbar.close()
        
        # print the maximum allocated memory 
        print_memory_stats()
        
        # print elapsed time
        print('elapsed time: ', datetime.now() - start_time, '\n')
    
    def _train_step(self, 
                 dataloader: Dataloader, 
                 model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler, 
                 scaler: torch.amp.GradScaler,
                 step: int) -> Dict[str, float]:
        
        # make sure we are in train mode 
        model.train()
        
        # fetch the data 
        data = next(dataloader)
        data = {k : v.cuda() for k, v in data.items()}
        
        # runs the forward pass with autocasting 
        with torch.autocast(device_type='cuda', dtype=self.dtype, enabled=self.enabled):
            metrics = self.specification.train_objective(data, model)
            loss = metrics['loss']
        
        # make sure the magnitude of the gradients is okay 
        loss = loss  / self.specification.accumulation_steps
        
        # scales loss. calls backward() on scaled loss to create scaled gradients 
        # backward passes under autocast are not recommended 
        # backward ops run in the same dtype autocast chose for corresponding forward ops 
        scaler.scale(loss).backward()
        
        if step % self.specification.accumulation_steps == 0:
            # scaler.step() first unscales the gradients of the optimizer's assigned params 
            # if these gradients do not contain infs or NaNs, optimizer.step() is then called 
            # otherwise, optimizer.step() is skipped 
            scaler.step(optimizer)
            
            # updates the scale for next iteration 
            scaler.update()
            
            # zeroes out the grads 
            optimizer.zero_grad()
            
            # updates scheduler 
            scheduler.step()
        
        # return metrics 
        return {k: v.item() for k, v in metrics.items()}
    
    @torch.inference_mode()
    def _eval_step(self, dataloader: Dataloader, model: torch.nn.Module) -> Dict[str, float]:
        # fetch the data 
        data = next(dataloader)
        data = {k : v.cuda() for k, v in data.items()}
        # run the model on the data and get the metrics 
        model.eval()
        metrics = self.specification.eval_objective(data, model)
        return {k: v.item() for k, v in metrics.items()}

