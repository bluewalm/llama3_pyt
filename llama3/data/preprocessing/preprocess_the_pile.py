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

import re
import json
import random
import unicodedata
from glob import glob
from tqdm import tqdm


def fancy_print(text: str):
    print(text)
    print("=" * len(text))


def clean_text(text):
    # normalize unicode characters 
    normalized = unicodedata.normalize('NFKD', text)
    # remove non-ascii characters 
    ascii_text = normalized.encode('ASCII', 'ignore').decode('ASCII')
    # remove empty parantheses
    ascii_text = ascii_text.replace(" (  ) ", " ")
    ascii_text = ascii_text.replace(" [  ] ", " ")
    ascii_text = ascii_text.replace(" {  } ", " ")
    ascii_text = ascii_text.replace(" ( ) ", " ")
    ascii_text = ascii_text.replace(" [ ] ", " ")
    ascii_text = ascii_text.replace(" { } ", " ")
    return ascii_text


def preprocess_line(line: str):
    line = clean_text(line)
    line = line.lower()
    return line


if __name__ == "__main__":
    fancy_print("preprocessing the pile dataset")
    files = glob("*.jsonl")
    print("jsonl files found: ")
    for f in files:
        print("", f)
    train_write_path = "pretrain_data/train/pile-train-{}.txt"
    eval_write_path = "pretrain_data/eval/pile-eval-{}.txt"
    train_samples = 0
    eval_samples = 0
    train_file_num = 1
    eval_file_num = 1
    train_wfp = open(train_write_path.format(train_file_num), "w", encoding="utf-8")
    eval_wfp = open(eval_write_path.format(eval_file_num), "w", encoding="utf-8")
    split_ratio = [0.9, 0.1]  # train : test ratio 
    # calculate number of samples 
    print("calculating total number of samples.... ")
    total = 0
    for path in tqdm(files, total=len(files)):
        with open(path, "r", encoding="utf-8") as fp:
            total += sum(1 for _ in fp)
    print("total samples: ", total)
    # process data 
    pbar = tqdm(total=total, desc='preprocessing')
    for path in files:
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                line = json.loads(line)
                line = line['text']
                line = preprocess_line(line)
                wfp = random.choices((train_wfp, eval_wfp), weights=split_ratio, k=1)[0]
                wfp.write(line)
                if wfp == train_wfp:
                    train_samples += 1
                    if train_samples % 16384 == 0 and train_samples > 0:
                        train_wfp.close()
                        train_file_num += 1
                        train_wfp = open(train_write_path.format(train_file_num), "w", encoding="utf-8")
                else:
                    eval_samples += 1
                    if eval_samples % 16384 == 0 and eval_samples > 0:
                        eval_wfp.close()
                        eval_file_num += 1
                        eval_wfp = open(eval_write_path.format(eval_file_num), "w", encoding="utf-8")
                pbar.update(1)
    pbar.close()
    train_wfp.close()
    eval_wfp.close()
    print("....done. ")
    print("train samples written: {}\neval samples written: {}".format(train_samples, eval_samples))

