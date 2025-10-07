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
import torch.distributed.fsdp as fsdp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)
from datetime import datetime
from typing import Dict, Optional
# 
from llama3.data import Dataloader
from llama3.utils import Recorder, print_model, print_memory_stats, cycle


class MultiTrainer:
    def __init__(self, specification):
        self.specification = specification
        self.dtype, self.enabled = None, False
        self.explicit_prefetching = (1 < self.specification.prefetching)
        self.options_to = StateDictOptions(full_state_dict=True, cpu_offload=True)
        self.options_from = StateDictOptions(full_state_dict=True, broadcast_from_rank0=True)
    
    def initialize_process(self):
        try:
            rank = int(os.environ["LOCAL_RANK"])
        except KeyError:
            raise Exception("LOCAL_RANK environment variable not set")
        assert torch.accelerator.is_available()
        device_type = torch.accelerator.current_accelerator()
        device = torch.device(f"{device_type}:{rank}")
        torch.accelerator.set_device_index(rank)
        print(f"Running on rank {rank} on device {device}")
        backend = torch.distributed.get_default_backend_for_device(device)
        torch.distributed.init_process_group(backend=backend, device_id=device)
        main_process = (rank == 0)  # is this the main process? 
        return device, main_process
    
    def destroy_process(self):
        torch.distributed.destroy_process_group()
    
    def set_modules_to_forward_prefetch(self, model, num_to_forward_prefetch):
        # one layer is always implicitly prefetched 
        if 1 < num_to_forward_prefetch:
            for i, layer in enumerate(model.layers):
                if i >= len(model.layers) - num_to_forward_prefetch:
                    break
                layers_to_prefetch = [model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)]
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)
    
    def set_modules_to_backward_prefetch(self, model, num_to_backward_prefetch):
        # one layer is always implicitly prefetched 
        if 1 < num_to_backward_prefetch:
            for i, layer in enumerate(model.layers):
                if i < num_to_backward_prefetch:
                    continue
                layers_to_prefetch = [model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)]
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)
    
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
        # prepare multi-gpu training 
        device, main_process = self.initialize_process()
        
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
        
        # construct a fake model 
        with torch.device("meta"):
            model = self.specification.construct_model()
        
        # cast the model when bfloat16 precision flag is set 
        fsdp_kwargs = {}
        if self.specification.bf16:
            fsdp_kwargs["mp_policy"] = fsdp.MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32
            )
        
        # fully shard each transformer layer 
        for layer in model.layers:
            fsdp.fully_shard(layer, **fsdp_kwargs)
        fsdp.fully_shard(model, **fsdp_kwargs)
        
        # prefetch layers 
        self.set_modules_to_forward_prefetch(model, self.specification.prefetching)
        self.set_modules_to_backward_prefetch(model, self.specification.prefetching)
        
        # create an optimizer and learning rate scheduler 
        optimizer, scheduler = self.specification.create_optimizer(model.parameters())
        # creates the recorder 
        recorder = Recorder()
        
        start_step = 0
        # restore last training states from checkpoint 
        if from_checkpoint:
            ckpt_model, ckpt_optimizer, ckpt_other = self.checkpoint_names(from_checkpoint)
            
            model_state_dict = torch.load(ckpt_model, mmap=True, weights_only=True, map_location="cpu")
            set_model_state_dict(model=model, model_state_dict=model_state_dict, options=self.options_from)
            
            optim_state_dict = torch.load(ckpt_optimizer, mmap=True, weights_only=True, map_location="cpu")
            set_optimizer_state_dict(model=model, optimizers=optimizer, optim_state_dict=optim_state_dict, options=self.options_from)
            
            other_state_dict = torch.load(ckpt_other, mmap=True, weights_only=True, map_location="cpu")
            start_step = other_state_dict['step']
            recorder = other_state_dict['recorder']
            scheduler.load_state_dict(other_state_dict['scheduler'])
            
            # because the checkpoint data allocates quite a lot of GPU 
            # memory, we need to free the memories explicitly 
            del model_state_dict
            del optim_state_dict
            del other_state_dict
            torch.cuda.empty_cache()
        else:
            model.to_empty(device=device)
            model.reset_parameters()
        
        # prepare the caches 
        self.specification.prepare_empty_caches(model)
        
        # allow tf32 
        torch.backends.cudnn.allow_tf32 = bool(self.specification.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.specification.allow_tf32)
        
        # print the model 
        if main_process:
            print_model(model)
        
        # measure time
        start_time = datetime.now()
        
        # create tqdm iterator in master process to show the progress of training 
        rng = range(start_step + 1, self.specification.total_steps + 1)
        total = self.specification.total_steps
        desc = self.specification.description
        disable = not main_process
        pbar = tqdm.tqdm(rng, total=total, desc=desc, dynamic_ncols=True, disable=disable)
        pbar.update(start_step)
        for step in pbar:
            # clear CUDA cache which is used for training 
            torch.cuda.empty_cache()
            
            recorder.record(self._train_step(train_dataloader, model, optimizer, scheduler, step), step=step, scope='train')
            
            # clear CUDA cache which is used for evaluation 
            torch.cuda.empty_cache()
            
            if step % self.specification.eval_steps == 0:
                recorder.record(self._eval_step(eval_dataloader, model), step=step, scope='eval')
                
                # set postfix str of the progress bar 
                postfix_str = recorder.format(self.specification.log_format)
                pbar.set_postfix_str(postfix_str)
            
            # save training states to checkpoint file 
            if (step % self.specification.save_steps == 0):
                ckpt_model, ckpt_optimizer, ckpt_other = self.checkpoint_names(self.specification.to_checkpoint)
                
                model_state_dict = get_model_state_dict(model=model, options=self.options_to)
                
                optim_state_dict = get_optimizer_state_dict(model=model, optimizers=optimizer, options=self.options_to)
                
                other_state_dict = {'step' : step,
                                    'recorder' : recorder,
                                    'scheduler' : scheduler.state_dict()}
                
                if main_process:
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
        
        model_state_dict = get_model_state_dict(model=model, options=self.options_to)
        
        optim_state_dict = get_optimizer_state_dict(model=model, optimizers=optimizer, options=self.options_to)
        
        other_state_dict = {'step' : step,
                            'recorder' : recorder,
                            'scheduler' : scheduler.state_dict()}
        
        if main_process:
            torch.save(model_state_dict, ckpt_model)
            torch.save(optim_state_dict, ckpt_optimizer)
            torch.save(other_state_dict, ckpt_other)
        
        # close progress bar 
        pbar.close()
        
        # print the maximum allocated memory 
        if main_process:
            print_memory_stats()
        
        # print elapsed time
        if main_process:
            print('elapsed time: ', datetime.now() - start_time, '\n')
        
        # destroy the other processes 
        self.destroy_process()
    
    def _train_step(self, 
                 dataloader: Dataloader, 
                 model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler._LRScheduler, 
                 step: int) -> Dict[str, float]:
        
        # make sure we are in train mode 
        model.train()
        
        # fetch the data 
        data = next(dataloader)
        data = {k : v.cuda() for k, v in data.items()}
        
        # explicit prefetching 
        if self.explicit_prefetching:
            model.unshard()
        
        # runs the forward pass 
        metrics = self.specification.train_objective(data, model)
        loss = metrics['loss']
        
        # make sure the magnitude of the gradients is okay 
        loss = loss  / self.specification.accumulation_steps
        
        # calls backward() on loss to create gradients 
        # backward passes under autocast are not recommended 
        # backward ops run in the same dtype autocast chose for corresponding forward ops 
        loss.backward()
        
        # clip gradients 
        if self.specification.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.specification.max_norm)
        
        if step % self.specification.accumulation_steps == 0:
            # optimizer.step() applies the gradients of the optimizer's assigned params 
            # if these gradients do not contain infs or NaNs, optimizer.step() is then called 
            # otherwise, optimizer.step() is skipped 
            optimizer.step()
            
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
        # make sure we are in eval mode 
        model.eval()
        # explicit prefetching 
        if self.explicit_prefetching:
            model.unshard()
        # run the model on the data and get the metrics 
        metrics = self.specification.eval_objective(data, model)
        return {k: v.item() for k, v in metrics.items()}

