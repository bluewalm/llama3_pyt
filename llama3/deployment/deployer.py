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
import json
import torch
import tqdm
import numpy as np
from time import time
from typing import Optional, Dict
# 
from llama3.data import Dataloader
from llama3.utils import Recorder, print_model, cycle
from llama3.deployment.utils import execute, export_inputs, shape_to_str, convert_to_tensorrt
import llama3.deployment.tensorrt_lib as tensorrt_lib


class Deployer:
    def __init__(self, specification):
        self.specification = specification
    
    def checkpoint_name(self, checkpoint):
        path, filename = os.path.split(checkpoint)
        root, ext = os.path.splitext(filename)
        ckpt_model = root + "_model" + ext
        ckpt_model = os.path.join(path, ckpt_model)
        return ckpt_model
    
    @torch.inference_mode()
    def _eval_step(self, dataloader: Dataloader, model: torch.nn.Module) -> Optional[Dict[str, float]]:
        # fetch the data  
        data = next(dataloader)
        data = {k : v.cuda() for k, v in data.items()}
        # run the model on the data and get the metrics  
        metrics = self.specification.eval_objective(data, model)
        return {k: v.item() for k, v in metrics.items()}
    
    def evaluate(self, eval_dataloader, model, desc=''):
        # creates the recorder 
        recorder = Recorder()
        # create tqdm iterator in master process to show the progress of training 
        rng = range(1, self.specification.total_steps + 1)
        total = self.specification.total_steps
        pbar = tqdm.tqdm(rng, total=total, desc=desc, dynamic_ncols=True)
        with torch.inference_mode():
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
        return recorder.averages
    
    def inference_benchmark(self, eval_dataloader, model):
        model.eval()
        latencies = []
        with torch.inference_mode():
            for step in range(self.specification.inference_benchmark_steps):
                torch.cuda.synchronize()
                step_start_time = time()
                inference_result = self._eval_step(eval_dataloader, model)
                torch.cuda.synchronize()
                step_time = time() - step_start_time
                if step >= self.specification.benchmark_warmup_steps:
                    latencies.append(step_time)
        return latencies
    
    def get_sample(self, eval_dataloader, batch_size, query_len, cache_len):
        data = next(eval_dataloader)
        data = {k : v.cuda() for k, v in data.items()}
        input = data['input'].to(device='cuda', dtype=torch.int64)
        input = input[:batch_size,:query_len]
        k_cache = self.specification.k_cache(cache_len)
        k_cache = k_cache[:,:batch_size,...]
        v_cache = self.specification.v_cache(cache_len)
        v_cache = v_cache[:,:batch_size,...]
        return (input, k_cache, v_cache)
    
    def reset_dataloader(self):
        # prepare dataset 
        eval_dataset = self.specification.prepare_dataset()
        
        # prepare dataloader 
        eval_dataloader = self.specification.prepare_dataloader(eval_dataset, self.specification.batch_size)
        
        # cycle through them in an infinite loop 
        if (0 < self.specification.total_steps) and (self.specification.total_steps < float('inf')):
            # cycle through them in an infinite loop 
            eval_dataloader = cycle(eval_dataloader)
        else:
            self.specification.total_steps = float('inf')
        # return it
        return eval_dataloader
    
    def deploy(self, from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        # prepare tokenizer 
        self.specification.prepare_tokenizer()
        
        # prepare loss function 
        self.specification.prepare_loss_function()
        
        # prepare dataloader 
        eval_dataloader = self.reset_dataloader()
        
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
        precision = 'fp32'
        if self.specification.amp_fp16:
            model.half()
            precision = 'fp16'
        elif self.specification.amp_bf16 or self.specification.bf16:
            model.bfloat16()
            precision = 'bf16'
        
        # allow tf32 
        torch.backends.cudnn.allow_tf32 = bool(self.specification.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(self.specification.allow_tf32)
        
        model.eval()
        
        # print the model 
        print_model(model)
        
        desc = "evaluating pyt model"
        acc_pyt = self.evaluate(eval_dataloader, model, desc)
        
        # get samples 
        batch_size = self.specification.batch_size
        query_len = self.specification.max_seq_len
        cache_len = 0
        min_sample = self.get_sample(eval_dataloader, batch_size, query_len, cache_len)
        
        batch_size = self.specification.batch_size
        query_len = self.specification.max_seq_len
        cache_len = 0
        opt_sample = self.get_sample(eval_dataloader, batch_size, query_len, cache_len)
        
        batch_size = self.specification.batch_size
        query_len = self.specification.max_seq_len
        cache_len = self.specification.max_seq_len
        max_sample = self.get_sample(eval_dataloader, batch_size, query_len, cache_len)
        
        # ts export  
        with torch.no_grad():
            model_traced = torch.jit.trace(model, opt_sample, check_trace=False)
        
        # turn off torchscript recompilations  
        with torch.jit.optimized_execution(False), torch.no_grad():
            # reset dataloader 
            eval_dataloader = self.reset_dataloader()
            # ts_eval  
            desc = "evaluating ts model"
            acc_ts = self.evaluate(eval_dataloader, model_traced, desc)
            # onnx_export  
            torch.onnx.export(model_traced, opt_sample, "model.onnx", verbose=False, 
                             opset_version=18, export_params=True, 
                             keep_initializers_as_inputs=False, 
                             custom_opsets={"trt.plugins" : 1}, 
                             do_constant_folding=True, 
                             input_names=['input', 'k_cache', 'v_cache'], 
                             output_names=['output', 'updated_k_cache', 'updated_v_cache'], 
                             dynamic_axes={'input' : {0 : 'batch_size', 1 : 'query_len'}, 
                                           'k_cache' : {1 : 'batch_size', 2 : 'cache_len'}, 
                                           'v_cache' : {1 : 'batch_size', 2 : 'cache_len'}, 
                                           'output' : {0 : 'batch_size', 1 : 'query_len'}, 
                                           'updated_k_cache' : {1 : 'batch_size', 2 : 'updated_cache_len'}, 
                                           'updated_v_cache' : {1 : 'batch_size', 2 : 'updated_cache_len'}})
        
        # trt export 
        export_inputs(opt_sample)
        
        engine_file_path = convert_to_tensorrt(precision, min_sample=min_sample, opt_sample=opt_sample, max_sample=max_sample)
        
        # trt eval 
        with tensorrt_lib.Model(engine_file_path) as model_trt:
            # reset dataloader 
            eval_dataloader = self.reset_dataloader()
            desc = "evaluating trt model (numpy wrapper)"
            acc_trt = self.evaluate(eval_dataloader, model_trt, desc)
        
        # print stats  
        acc_data = {
                'pytorch (acc)'   : acc_pyt, 
                'torchscript (acc)' : acc_ts, 
                'tensorrt (acc)'   : acc_trt, 
        }
        print(json.dumps(acc_data, indent=4))
        
        # make sure dataloader won't run out of data 
        eval_dataloader = cycle(eval_dataloader)
        
        print("inference benchmark - pytorch")
        # measure perf for pytorch model  
        latencies = self.inference_benchmark(eval_dataloader, model)
        
        # calculate throughput and latency data  
        pytorch_perf_data = {
            'pytorch mean_throughput' : self.specification.batch_size / np.mean(latencies), 
            'pytorch mean_latency' : np.mean(latencies), 
            'pytorch p90_latency' : np.percentile(latencies, 0.90), 
            'pytorch p95_latency' : np.percentile(latencies, 0.95),
            'pytorch p99_latency' : np.percentile(latencies, 0.99)
        }
        print(json.dumps(pytorch_perf_data, indent=4))
        
        print("inference benchmark - torchscript")
        with torch.jit.optimized_execution(False):
            # measure perf for torchscript model  
            latencies = self.inference_benchmark(eval_dataloader, model_traced)
        
        # calculate throughput and latency data  
        torchscript_perf_data = {
            'torchscript mean_throughput' : self.specification.batch_size / np.mean(latencies), 
            'torchscript mean_latency' : np.mean(latencies), 
            'torchscript p90_latency' : np.percentile(latencies, 0.90), 
            'torchscript p95_latency' : np.percentile(latencies, 0.95),
            'torchscript p99_latency' : np.percentile(latencies, 0.99)
        }
        print(json.dumps(torchscript_perf_data, indent=4))
        
        print("inference benchmark - trt")
        # measure trt latency 
        command = "trtexec --loadEngine=./model.engine"
        if precision == 'fp16':
            command += " --inputIOFormats=int64:chw,fp16:chw,fp16:chw"
            command += " --outputIOFormats=fp16:chw,fp16:chw,fp16:chw"
            command += " --fp16"
        elif precision == 'fp32':
            command += " --inputIOFormats=int64:chw,fp32:chw,fp32:chw"
            command += " --outputIOFormats=fp32:chw,fp32:chw,fp32:chw"
        elif precision == 'bf16':
            command += " --inputIOFormats=int64:chw,bf16:chw,bf16:chw"
            command += " --outputIOFormats=bf16:chw,bf16:chw,bf16:chw"
            command += " --bf16"
        else:
            raise Exception('unknown precision')
        cache_len = opt_sample[1].shape[2]
        if cache_len == 0:
            command += " --loadInputs=input:./input.dat"
        else:
            command += " --loadInputs=input:./input.dat,k_cache:./k_cache.dat,v_cache:./v_cache.dat"
        input_shape = shape_to_str(opt_sample[0].shape)
        k_cache_shape = shape_to_str(opt_sample[1].shape)
        v_cache_shape = shape_to_str(opt_sample[2].shape)
        command += " --shapes=input:" + input_shape + ",k_cache:" + k_cache_shape + ",v_cache:" + v_cache_shape
        command += " --iterations=" + str(self.specification.inference_benchmark_steps + self.specification.benchmark_warmup_steps)
        command += " --avgRuns=" + str(self.specification.inference_benchmark_steps)
        command += " --infStreams=1"
        command += " --noDataTransfers"
        command += " --useSpinWait"
        trtexec_output = execute(command)
        s = '=== Performance summary ==='
        index = trtexec_output.find(s)
        result = trtexec_output[index+len(s):]
        result = [x.split("[") for x in result.split("[I]")]
        result = [""] + [x[0] for x in result]
        result = "\n".join(result)
        return result

