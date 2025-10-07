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
from typing import List
from operator import indexOf
# 
from llama3.generation import Generator
from llama3.modeling import ModelArgs, Transformer
from llama3.data import Tokenizer, TokenizerArgs
from llama3.utils.argparse_types import positive_integer, probability, filepath


class Specification:
    def __init__(self, args : argparse.Namespace):
        self.__dict__.update(args.__dict__)
        self.description='generating sentences with llama3'
    
    def prepare_tokenizer(self):
        args = TokenizerArgs(model_file=self.tokenizer)
        self.tokenizer = Tokenizer(args)
        assert self.tokenizer.model is not None
        self.vocab_size = self.tokenizer.model.vocab_size()  # ModelArgs will accept it
        self.bos_id = self.tokenizer.model.bos_id()
        self.eos_id = self.tokenizer.model.eos_id()
    
    def construct_model(self) -> torch.nn.Module:
        kwargs = self.__dict__
        args = ModelArgs(**kwargs)
        model = Transformer(args)
        return model
    
    def prepare_empty_caches(self, model):
        self.k_cache = lambda : model.get_empty_k_cache(1, cache_len=0)
        self.v_cache = lambda : model.get_empty_v_cache(1, cache_len=0)
    
    def encode_context(self, context: str) -> List[int]:
        tokens = self.tokenizer.encode(context)
        tokens = [self.bos_id] + tokens + [self.eos_id]
        return tokens
    
    def decode_tokens(self, tokens: List[int]) -> str:
        # cutoff the last sentence 
        if self.eos_id in tokens:
            last_index = len(tokens) - indexOf(reversed(tokens), self.eos_id) - 1
            tokens = tokens[:last_index+1]
        return self.tokenizer.decode(tokens)


def generate_with_llama3_model(args : argparse.Namespace):
    specification = Specification(args)
    generator = Generator(specification)
    generator.initialize(from_checkpoint=args.from_checkpoint)
    while True:
        print(generator.generate(input('>>')))


def add_subparser(subparsers : argparse._SubParsersAction):
    parser = subparsers.add_parser('generate', help='generate sentences with LLAMA3 model')
    # corpus 
    group = parser.add_argument_group('corpus')
    group.add_argument('--tokenizer', type=filepath, required=True,
                       help='tokenizer model file path')
    # model architecture 
    group = parser.add_argument_group('model architecture')
    group.add_argument('--dim', type=positive_integer, required=True,
                       help='dimension of the internal representation of tokens')
    group.add_argument('--max_seq_len', type=positive_integer, required=True,
                       help='maximum sequence length')
    group.add_argument('--n_layers', type=positive_integer, required=True,
                       help='number of transformer layers')
    group.add_argument('--core_dim', default=None, type=positive_integer,
                       help='core dimension of the attention layer')
    group.add_argument('--norm_eps', default=1e-5, type=float,
                       help='norm eps')
    group.add_argument('--rope_theta', default=500000, type=float,
                       help='rope theta')
    group.add_argument('--use_scaled_rope', default=False, type=bool,
                       help='use scaled rope')
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
    # generation 
    group = parser.add_argument_group('generation')
    group.add_argument('--nucleus_prob', default=0.85, type=probability, 
                       help='probability threshold for nucleus sampling')
    # set defaults 
    parser.set_defaults(func=generate_with_llama3_model)

