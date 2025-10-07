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
import argparse
from typing import Dict
# 
from llama3.deployment import Deployer
from llama3.modeling import ModelArgs, Transformer
from llama3.data import Dataset, Dataloader, Tokenizer, TokenizerArgs
from llama3.utils.osutil import files_in_directory
from llama3.utils.argparse_types import positive_integer, nonzero_integer, probability, directory, filepath


class Specification:
    def __init__(self, args : argparse.Namespace):
        self.__dict__.update(args.__dict__)
        self.description='evaluating llama3'
        self.log_format='eval/loss: {eval_loss:.4f}'
    
    def prepare_tokenizer(self):
        args = TokenizerArgs(model_file=self.tokenizer)
        self.tokenizer = Tokenizer(args)
        assert self.tokenizer.model is not None
        self.pad_id = self.tokenizer.model.pad_id()  # self.prepare_loss_function(....) uses it
        self.vocab_size = self.tokenizer.model.vocab_size()  # ModelArgs will accept it
    
    def prepare_loss_function(self):
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id,
                                            reduction='mean')
    
    def prepare_dataset(self) -> Dataset:
        eval_dataset = Dataset(corpus_paths=files_in_directory(self.eval_corpus),
                            tokenizer=self.tokenizer,
                            max_seq_len=self.max_seq_len,
                            shuffle=False)
        return eval_dataset
    
    def prepare_dataloader(self, eval_dataset : Dataset, batch_size : int) -> Dataloader:
        eval_dataloader = Dataloader(eval_dataset, batch_size=self.batch_size)
        return eval_dataloader
    
    def construct_model(self) -> torch.nn.Module:
        kwargs = self.__dict__
        args = ModelArgs(**kwargs)
        model = Transformer(args)
        return model
    
    def prepare_empty_caches(self, model):
        self.k_cache = lambda cache_len : model.get_empty_k_cache(self.batch_size, cache_len)
        self.v_cache = lambda cache_len : model.get_empty_v_cache(self.batch_size, cache_len)
    
    def eval_objective(self, data : Dict[str, torch.Tensor], model : torch.nn.Module) -> Dict[str, torch.Tensor]:
        cache_len = 0
        logits, _, _ = model(data['input'], self.k_cache(cache_len), self.v_cache(cache_len))
        loss = self.criterion(logits.transpose(1, 2).float(), data['output'])
        return {'loss' : loss}


def deploy_llama3_model(args : argparse.Namespace):
    specification = Specification(args)
    Deployer(specification).deploy(from_checkpoint=args.from_checkpoint)


def add_subparser(subparsers : argparse._SubParsersAction):
    parser = subparsers.add_parser('deploy', help='deploy and evaluate LLAMA3 model')
    # corpus 
    group = parser.add_argument_group('corpus')
    group.add_argument('--eval_corpus', type=directory, required=True,
                       help='evaluation corpus directory path')
    group.add_argument('--tokenizer', type=filepath, required=True,
                       help='tokenizer model file path')
    # model architecture 
    group = parser.add_argument_group('model architecture')
    group.add_argument('--dim', default=4096, type=positive_integer,
                       help='dimension of representation in each layer')
    group.add_argument('--n_layers', default=32, type=positive_integer,
                       help='number of transformer layers')
    group.add_argument('--n_heads', default=32, type=positive_integer,
                       help='number of multi-heads in attention layer')
    group.add_argument('--n_kv_heads', default=None, type=positive_integer,
                       help='number of multi-heads in attention layer')
    group.add_argument('--norm_eps', default=1e-5, type=float,
                       help='norm eps')
    group.add_argument('--rope_theta', default=500000, type=float,
                       help='rope theta')
    group.add_argument('--use_scaled_rope', default=False, type=bool,
                       help='use scaled rope')
    group.add_argument('--max_seq_len', default=64, type=positive_integer,
                       help='maximum sequence length')
    group.add_argument('--multiple_of', default=256, type=positive_integer,
                       help='bottleneck parameter')
    group.add_argument('--ffn_dim_multiplier', default=None, type=positive_integer,
                       help='bottleneck parameter')
    # evaluation 
    group = parser.add_argument_group('evaluation')
    group.add_argument('--batch_size', default=32, type=positive_integer,
                       help='the batch_size for training and evaluation')
    group.add_argument('--total_steps', default=-1, type=nonzero_integer,
                       help='number of total evaluation steps')
    # restoring 
    group = parser.add_argument_group('restoring')
    group.add_argument('--from_checkpoint', default=None,
                       help='load last training state from checkpoint file')
    # precision 
    group = parser.add_argument_group('precision')
    group.add_argument('--allow_tf32', action='store_true',
                       help='allow the use of tf32')
    subgroup = group.add_mutually_exclusive_group(required=False)
    subgroup.add_argument('--amp_fp16', action='store_true',
                        help='use automatic mixed-precision - float32/float16')
    subgroup.add_argument('--amp_bf16', action='store_true',
                        help='use automatic mixed-precision - float32/bfloat16')
    subgroup.add_argument('--bf16', action='store_true', 
                        help='pure bfloat16 precision')
    # benchmarking 
    group = parser.add_argument_group('benchmarking')
    group.add_argument('--inference_benchmark_steps', default=10000, type=positive_integer, 
                       help='number of steps to run the inference benchmark on')
    group.add_argument('--benchmark_warmup_steps', default=100, type=positive_integer, 
                       help='number of steps to warmup the benchmark with')
    # set defaults 
    parser.set_defaults(func=deploy_llama3_model)

