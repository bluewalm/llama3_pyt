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

for N_LAYERS in 3
do
for BATCH_SIZE in 64
do
for DIM in 128
do
for CORE_DIM in 128 424
do
for STEPS in 80000
do
for SEQLEN in 4096 8192
do
python -m llama3 train --train_corpus         /workspace/Datasets/ThePile/pretrain_data/train \
                     --eval_corpus            /workspace/Datasets/ThePile/pretrain_data/eval \
                     --tokenizer              tokenizer.model \
                     --n_layers               ${N_LAYERS} \
                     --dim                    ${DIM} \
                     --core_dim               ${CORE_DIM} \
                     --to_checkpoint          llama3-softplus-layers${N_LAYERS}-dim${DIM}-coredim${CORE_DIM}-seqlen${SEQLEN}-batchsize${BATCH_SIZE}-steps${STEPS}.pth \
                     --batch_size             ${BATCH_SIZE} \
                     --accumulation_steps     1 \
                     --max_seq_len            ${SEQLEN} \
                     --total_steps            ${STEPS} \
                     --eval_steps             10 \
                     --save_steps             ${STEPS} \
                     --allow_tf32 \
                     --amp_bf16
done
done
done
done
done
done

