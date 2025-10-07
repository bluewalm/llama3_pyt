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
for N_HEADS in 4
do
for N_KV_HEADS in 4
do
for STEPS in 80000
do
for SEQLEN in 4096 8192
do
python -m llama3 evaluate --eval_corpus         /workspace/Datasets/ThePile/pretrain_data/eval \
                        --tokenizer             tokenizer.model \
                        --n_layers              ${N_LAYERS} \
                        --dim                   ${DIM} \
                        --n_heads               ${N_HEADS} \
                        --n_kv_heads            ${N_KV_HEADS} \
                        --from_checkpoint       llama3-softmax-layers${N_LAYERS}-dim${DIM}-heads${N_HEADS}-kvheads${N_KV_HEADS}-seqlen${SEQLEN}-batchsize${BATCH_SIZE}-steps${STEPS}.pth \
                        --batch_size            ${BATCH_SIZE} \
                        --max_seq_len           ${SEQLEN} \
                        --total_steps           1000 \
                        --allow_tf32 \
                        --bf16
done
done
done
done
done
done
done

