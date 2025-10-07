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
import string
import random
import sentencepiece as spm
from collections import Counter
from typing import List, Optional
from dataclasses import dataclass, fields


@dataclass
class TokenizerArgs:
    vocab_size: int
    max_sentence_length: int
    input_sentence_size: int
    corpus_files: List[str]
    model_file: str
    n_most_frequent_words: int
    unk_id: int
    pad_id: int
    bos_id: int
    eos_id: int
    
    def __init__(self, **kwargs):
        field_names = [field.name for field in fields(self)]
        for k, v in kwargs.items():
            if k in field_names:
                setattr(self, k, v)
        # set the defaults (important) 
        if 'vocab_size' not in kwargs:
            setattr(self, 'vocab_size', 2**15)
        if 'max_sentence_length' not in kwargs:
            setattr(self, 'max_sentence_length', 4192)
        if 'input_sentence_size' not in kwargs:
            setattr(self, 'input_sentence_size', 2000000)
        if 'corpus_files' not in kwargs:
            setattr(self, 'corpus_files', [""])
        if 'model_file' not in kwargs:
            setattr(self, 'model_file', "tokenizer.model")
        if 'n_most_frequent_words' not in kwargs:
            setattr(self, 'n_most_frequent_words', 2**12)
        if 'unk_id' not in kwargs:
            setattr(self, 'unk_id', 0)
        if 'pad_id' not in kwargs:
            setattr(self, 'pad_id', 1)
        if 'bos_id' not in kwargs:
            setattr(self, 'bos_id', 2)
        if 'eos_id' not in kwargs:
            setattr(self, 'eos_id', 3)


class Tokenizer:
    def __init__(self, args : TokenizerArgs):
        self.args = args
        self.model = None
        if os.path.isfile(self.args.model_file):
            self.model = spm.SentencePieceProcessor(model_file=self.args.model_file)
    
    def top_n_most_frequent_words(self, corpus_str, n):
        # split the corpus string 
        corpus_split = corpus_str.split()
        # create a dict that maps all punctuation to the empty string 
        translator = str.maketrans('', '', string.punctuation)
        # remove punctuation from all words 
        depunctuate = lambda x : x.translate(translator)
        corpus_split = list(map(depunctuate, corpus_split))
        # lowercase all words 
        corpus_split = list(map(lambda x : x.lower(), corpus_split))
        # count the number of occurences of words 
        ctr = Counter(corpus_split)
        # keep only the `n` most frequent ones 
        most_frequent_words = ctr.most_common(n)
        # only need the word itself - not the counter 
        most_frequent_words = list(map(lambda x : "▁" + x[0], most_frequent_words))
        return most_frequent_words
    
    def recompute_n_user_defined_symbols(self, n):
        if not self.args.corpus_files:
            raise Exception("corpus_files must be set")
        corpus_file = random.choice(self.args.corpus_files)
        if not corpus_file:
            raise Exception("corpus_files must be set")
        with open(corpus_file, "r") as f:
            corpus_str = f.readlines()
        if self.args.input_sentence_size > 0:
            random.shuffle(corpus_str)
            corpus_str = corpus_str[:self.args.input_sentence_size]
        corpus_str = "".join(corpus_str)
        symbols = self.top_n_most_frequent_words(corpus_str, n)
        return symbols
    
    def train(self):
        user_defined_symbols = self.recompute_n_user_defined_symbols(self.args.n_most_frequent_words)
        model_prefix = self.args.model_file.split('.')[0]
        spm.SentencePieceTrainer.train(input=self.args.corpus_files, 
                                  model_prefix=model_prefix, 
                                  model_type='bpe', 
                                  vocab_size=self.args.vocab_size, 
                                  character_coverage=1.0, 
                                  minloglevel=0, 
                                  input_sentence_size=self.args.input_sentence_size, 
                                  shuffle_input_sentence=True, 
                                  user_defined_symbols=user_defined_symbols, 
                                  max_sentence_length=self.args.max_sentence_length, 
                                  unk_id=self.args.unk_id, 
                                  pad_id=self.args.pad_id, 
                                  bos_id=self.args.bos_id, 
                                  eos_id=self.args.eos_id)
        self.model = spm.SentencePieceProcessor(model_file=self.args.model_file)
    
    def encode(self, text: str) -> List[int]:
        if not self.model:
            raise Exception("model_file cannot be found - tokenizer must be trained to produce model_file ")
        return self.model.encode(text)
    
    def decode(self, tokens: List[int]) -> str:
        if not self.model:
            raise Exception("model_file cannot be found - tokenizer must be trained to produce model_file ")
        return self.model.decode(tokens)
    
    def __repr__(self):
        s = str(self.args)
        s = s.replace("TokenizerArgs", "Tokenizer")
        return s
    
    def __str__(self):
        s = str(self.args)
        s = s.replace("TokenizerArgs", "Tokenizer")
        return s


'''
Usage: ../build/src/spm_train [options] files

   --input (comma separated list of input sentences)  type: std::string default: ""
   --input_format (Input format. Supported format is `text` or `tsv`.)  type: std::string default: ""
   --model_prefix (output model prefix)  type: std::string default: ""
   --model_type (model algorithm: unigram, bpe, word or char)  type: std::string default: "unigram"
   --vocab_size (vocabulary size)  type: int32 default: 8000
   --accept_language (comma-separated list of languages this model can accept)  type: std::string default: ""
   --self_test_sample_size (the size of self test samples)  type: int32 default: 0
   --character_coverage (character coverage to determine the minimum symbols)  type: double default: 0.9995
   --input_sentence_size (maximum size of sentences the trainer loads)  type: std::uint64_t default: 0
   --shuffle_input_sentence (Randomly sample input sentences in advance. Valid when --input_sentence_size > 0)  type: bool default: true
   --seed_sentencepiece_size (the size of seed sentencepieces)  type: int32 default: 1000000
   --shrinking_factor (Keeps top shrinking_factor pieces with respect to the loss)  type: double default: 0.75
   --num_threads (number of threads for training)  type: int32 default: 16
   --num_sub_iterations (number of EM sub-iterations)  type: int32 default: 2
   --max_sentencepiece_length (maximum length of sentence piece)  type: int32 default: 16
   --max_sentence_length (maximum length of sentence in byte)  type: int32 default: 4192
   --split_by_unicode_script (use Unicode script to split sentence pieces)  type: bool default: true
   --split_by_number (split tokens by numbers (0-9))  type: bool default: true
   --split_by_whitespace (use a white space to split sentence pieces)  type: bool default: true
   --split_digits (split all digits (0-9) into separate pieces)  type: bool default: false
   --treat_whitespace_as_suffix (treat whitespace marker as suffix instead of prefix.)  type: bool default: false
   --allow_whitespace_only_pieces (allow pieces that only contain (consecutive) whitespace tokens)  type: bool default: false
   --control_symbols (comma separated list of control symbols)  type: std::string default: ""
   --control_symbols_file (load control_symbols from file.)  type: std::string default: ""
   --user_defined_symbols (comma separated list of user defined symbols)  type: std::string default: ""
   --user_defined_symbols_file (load user_defined_symbols from file.)  type: std::string default: ""
   --required_chars (UTF8 characters in this flag are always used in the character set regardless of --character_coverage)  type: std::string default: ""
   --required_chars_file (load required_chars from file.)  type: std::string default: ""
   --byte_fallback (decompose unknown pieces into UTF-8 byte pieces)  type: bool default: false
   --vocabulary_output_piece_score (Define score in vocab file)  type: bool default: true
   --normalization_rule_name (Normalization rule name. Choose from nfkc or identity)  type: std::string default: "nmt_nfkc"
   --normalization_rule_tsv (Normalization rule TSV file. )  type: std::string default: ""
   --denormalization_rule_tsv (Denormalization rule TSV file.)  type: std::string default: ""
   --add_dummy_prefix (Add dummy whitespace at the beginning of text)  type: bool default: true
   --remove_extra_whitespaces (Removes leading, trailing, and duplicate internal whitespace)  type: bool default: true
   --hard_vocab_limit (If set to false, --vocab_size is considered as a soft limit.)  type: bool default: true
   --use_all_vocab (If set to true, use all tokens as vocab. Valid for word/char models.)  type: bool default: false
   --unk_id (Override UNK (<unk>) id.)  type: int32 default: 0
   --bos_id (Override BOS (<s>) id. Set -1 to disable BOS.)  type: int32 default: 1
   --eos_id (Override EOS (</s>) id. Set -1 to disable EOS.)  type: int32 default: 2
   --pad_id (Override PAD (<pad>) id. Set -1 to disable PAD.)  type: int32 default: -1
   --unk_piece (Override UNK (<unk>) piece.)  type: std::string default: "<unk>"
   --bos_piece (Override BOS (<s>) piece.)  type: std::string default: "<s>"
   --eos_piece (Override EOS (</s>) piece.)  type: std::string default: "</s>"
   --pad_piece (Override PAD (<pad>) piece.)  type: std::string default: "<pad>"
   --unk_surface (Dummy surface string for <unk>. In decoding <unk> is decoded to `unk_surface`.)  type: std::string default: " ⁇ "
   --train_extremely_large_corpus (Increase bit depth for unigram tokenization.)  type: bool default: false
   --random_seed (Seed value for random generator.)  type: uint32 default: 4294967295
   --enable_differential_privacy (Whether to add DP while training. Currently supported only by UNIGRAM model.)  type: bool default: false
   --differential_privacy_noise_level (Amount of noise to add for DP)  type: float default: 0
   --differential_privacy_clipping_threshold (Threshold for clipping the counts for DP)  type: std::uint64_t default: 0
   --help (show help)  type: bool default: false
   --version (show version)  type: bool default: false
   --minloglevel (Messages logged at a lower level than this don't actually get logged anywhere)  type: int default: 0
'''

