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

url="https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/LLM_data/our_refined_datasets/pretraining/the-pile-pubmed-central-refine-result.jsonl"
wget -nc $url
wait

# set directories
ROOT_DIR="$(pwd)/pretrain_data"
TRAIN_DIR="$ROOT_DIR/train"
EVAL_DIR="$ROOT_DIR/eval"

# create train and test dirs if they don't exist
mkdir -p "$ROOT_DIR"
mkdir -p "$TRAIN_DIR"
mkdir -p "$EVAL_DIR"
