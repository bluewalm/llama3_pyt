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

import argparse
# 
from llama3.data import Tokenizer, TokenizerArgs
from llama3.utils.osutil import files_in_directory
from llama3.utils.argparse_types import positive_integer, directory, filepath


def train_tokenizer(args : argparse.Namespace):
    kwargs = vars(args)
    corpus_files = files_in_directory(args.train_corpus)
    args = TokenizerArgs(corpus_files=corpus_files, **kwargs)
    tokenizer = Tokenizer(args)
    tokenizer.train()


def add_subparser(subparsers : argparse._SubParsersAction):
    parser = subparsers.add_parser('train_tokenizer', help='train BPE tokenizer')
    # corpus 
    group = parser.add_argument_group('corpus')
    group.add_argument('--train_corpus', type=directory, required=True,
                       help='training corpus directory path')
    group.add_argument('--model_file', type=filepath, required=True,
                       help='model_file path')
    # training 
    group = parser.add_argument_group('tokenizer parameters')
    group.add_argument('--vocab_size', default=2**15, type=positive_integer,
                       help='size of vocabulary to be built')
    group.add_argument('--max_sentence_length', default=4192, type=positive_integer,
                       help='maximum length of sentence in byte')
    group.add_argument('--input_sentence_size', default=2000000, type=positive_integer,
                       help='maximum size of sentences the trainer loads')
    group.add_argument('--n_most_frequent_words', default=2**12, type=positive_integer,
                       help='number of most frequent words to receive unique token ids')
    # set defaults 
    parser.set_defaults(func=train_tokenizer)

