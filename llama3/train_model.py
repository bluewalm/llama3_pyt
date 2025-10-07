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
from typing import Tuple, Iterator, Dict
# 
from llama3.training import Trainer
from llama3.modeling import ModelArgs, Transformer
from llama3.data import Dataset, Dataloader, Tokenizer, TokenizerArgs
from llama3.utils.osutil import files_in_directory
from llama3.utils.argparse_types import positive_integer, probability, directory, filepath


class Specification:
    def __init__(self, args : argparse.Namespace):
        self.__dict__.update(args.__dict__)
        self.description='training llama3'
        self.log_format='train/loss: {train_loss:.4f}, eval/loss: {eval_loss:.4f}'
    
    def prepare_tokenizer(self):
        args = TokenizerArgs(model_file=self.tokenizer)
        self.tokenizer = Tokenizer(args)
        assert self.tokenizer.model is not None
        self.pad_id = self.tokenizer.model.pad_id()  # self.prepare_loss_function(....) uses it
        self.vocab_size = self.tokenizer.model.vocab_size()  # ModelArgs will accept it
    
    def prepare_loss_function(self):
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id,
                                            reduction='mean')
    
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        train_dataset = Dataset(corpus_paths=files_in_directory(self.train_corpus),
                            tokenizer=self.tokenizer,
                            max_seq_len=self.max_seq_len)
        eval_dataset = Dataset(corpus_paths=files_in_directory(self.eval_corpus),
                            tokenizer=self.tokenizer,
                            max_seq_len=self.max_seq_len)
        return train_dataset, eval_dataset
    
    def prepare_dataloaders(self, train_dataset : Dataset, eval_dataset : Dataset, 
                        batch_size : int) -> Tuple[Dataloader, Dataloader]:
        train_dataloader = Dataloader(dataset=train_dataset, batch_size=batch_size)
        eval_dataloader = Dataloader(dataset=eval_dataset, batch_size=batch_size)
        return train_dataloader, eval_dataloader
    
    def construct_model(self) -> torch.nn.Module:
        kwargs = self.__dict__
        args = ModelArgs(**kwargs)
        model = Transformer(args)
        return model
    
    def create_optimizer(self, params : Iterator[torch.nn.Parameter]) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        optimizer = torch.optim.AdamW(params, lr=self.base_lr, weight_decay=self.wd_rate)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / self.total_steps)
        return optimizer, scheduler
    
    def prepare_empty_caches(self, model):
        self.k_cache = lambda : model.get_empty_k_cache(self.batch_size, cache_len=0)
        self.v_cache = lambda : model.get_empty_v_cache(self.batch_size, cache_len=0)
    
    def train_objective(self, data : Dict[str, torch.Tensor], model : torch.nn.Module) -> Dict[str, torch.Tensor]:
        logits, _, _ = model(data['input'], self.k_cache(), self.v_cache())
        loss = self.criterion(logits.transpose(1, 2), data['output'])
        return {'loss' : loss}
    
    def eval_objective(self, data : Dict[str, torch.Tensor], model : torch.nn.Module) -> Dict[str, torch.Tensor]:
        logits, _, _ = model(data['input'], self.k_cache(), self.v_cache())
        loss = self.criterion(logits.transpose(1, 2), data['output'])
        return {'loss' : loss}


def train_llama3_model(args : argparse.Namespace):
    specification = Specification(args)
    Trainer(specification).train(from_checkpoint=args.from_checkpoint)


def add_subparser(subparsers : argparse._SubParsersAction):
    parser = subparsers.add_parser('train', help='train LLAMA3 model')
    # corpus 
    group = parser.add_argument_group('corpus')
    group.add_argument('--train_corpus', type=directory, required=True,
                       help='training corpus directory path')
    group.add_argument('--eval_corpus', type=directory, required=True,
                       help='evaluation corpus directory path')
    group.add_argument('--tokenizer', type=filepath, required=True,
                       help='tokenizer model file path')
    # model architecture 
    group = parser.add_argument_group('model architecture')
    group.add_argument('--dim', default=4096, type=positive_integer,
                       help='dimension of representation in each layer')
    group.add_argument('--max_seq_len', default=64, type=positive_integer,
                       help='maximum sequence length')
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
    group.add_argument('--multiple_of', default=256, type=positive_integer,
                       help='bottleneck parameter')
    group.add_argument('--ffn_dim_multiplier', default=None, type=positive_integer,
                       help='bottleneck parameter')
    # training and evaluation 
    group = parser.add_argument_group('training and evaluation')
    group.add_argument('--seed', default=None, type=int, 
                       help='random seed - set this for stability tests only')
    group.add_argument('--batch_size', default=32, type=positive_integer,
                       help='number of training batch size')
    group.add_argument('--base_lr', default=1e-4, type=float,
                       help='default learning rate')
    group.add_argument('--wd_rate', default=1e-2, type=float,
                       help='weight decay rate')
    group.add_argument('--dropout', default=0.1, type=probability,
                       help='probability that each element is dropped')
    group.add_argument('--accumulation_steps', default=1, type=positive_integer,
                       help='the number of steps to accumulate gradients over - \
                       setting it to one means there is no gradient accumulation')
    group.add_argument('--total_steps', default=1000000, type=positive_integer,
                       help='number of total training steps')
    group.add_argument('--eval_steps', default=500, type=positive_integer,
                       help='period to evaluate model and record metrics')
    group.add_argument('--save_steps', default=1000, type=positive_integer,
                       help='period to save training state to checkpoint')
    # saving and restoring 
    group = parser.add_argument_group('saving and restoring')
    group.add_argument('--to_checkpoint', default='checkpoint.pth',
                       help='save training state to the checkpoint file')
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
    # set defaults 
    parser.set_defaults(func=train_llama3_model)

