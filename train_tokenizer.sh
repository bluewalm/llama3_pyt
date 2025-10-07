#!/bin/bash

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

python -m llama3 train_tokenizer --train_corpus              /workspace/Datasets/ThePile/pretrain_data/train \
                               --model_file                tokenizer.model \
                               --vocab_size                2048 \
                               --input_sentence_size       4000000 \
                               --n_most_frequent_words     256 \
                               --max_sentence_length       4096

