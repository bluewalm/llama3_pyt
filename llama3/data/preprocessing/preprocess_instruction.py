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

import json
from tqdm import tqdm
import zstandard as zstd
from datasets import load_dataset


root_dir = "data"
write_path = "data/instruction_data/part-{}-{}.jsonl.zst"
dataset_map = {
    "yizhongw/self_instruct": "self_instruct",
    "BelleGroup/train_0.5M_CN": "belle_0.5M",
    "BelleGroup/train_1M_CN": "belle_1M",
    "BelleGroup/train_2M_CN": "belle_2M",
    "BelleGroup/school_math_0.25M": "belle_school_math_0.25M",
    "BelleGroup/multiturn_chat_0.8M": "belle_multiturn_chat_0.8M",
    "Graverman/Instruct-to-Code": "instruct_to_code",
    "qwedsacf/grade-school-math-instructions": "grade_school_math",
    "camel-ai/math": "camel_ai_math",
    "camel-ai/physics": "camel_ai_physics",
    "camel-ai/chemistry": "camel_ai_chemistry",
    "camel-ai/biology": "camel_ai_biology",
    ("bigscience/xP3mt", "code"): "xP3mt_code",
    ("bigscience/xP3mt", "zh"): "xP3mt_zh",
}


def process_hf_dataset(name, local_name):
    if isinstance(name, str):
        dataset = load_dataset(name)
    else:
        dataset = load_dataset(*name)
    total_num = 0
    file_num = 1
    wfp = zstd.open(write_path.format(local_name, file_num), "wb", encoding="utf-8")
    for line in tqdm(dataset["train"]):
        line = json.dumps(line)
        if total_num % 1024 == 0 and total_num > 0:
            file_num += 1
            wfp.close()
            wfp = zstd.open(
                write_path.format(local_name, file_num), "wb", encoding="utf-8"
            )
        wfp.write(line.encode("utf-8"))
        wfp.write(b"\n")
        total_num += 1
    wfp.close()
    print(
        "{} preprocess done. Total line: {}, Total file: {}".format(
            name, total_num, file_num
        )
    )


for k, v in dataset_map.items():
    process_hf_dataset(k, v)

local_name = "sharegpt_90K"
total_num = 0
file_num = 1
wfp = zstd.open(write_path.format(local_name, file_num), "wb", encoding="utf-8")
with open("{}/sg_90k_part1_html_cleaned.json".format(root_dir), "r") as fp:
    data1 = json.load(fp)
with open("{}/sg_90k_part2_html_cleaned.json".format(root_dir), "r") as fp:
    data2 = json.load(fp)
data = data1 + data2
for line in tqdm(data):
    line = json.dumps(line)
    if total_num % 1024 == 0 and total_num > 0:
        file_num += 1
        wfp.close()
        wfp = zstd.open(write_path.format(local_name, file_num), "wb", encoding="utf-8")
    wfp.write(line.encode("utf-8"))
    wfp.write(b"\n")
    total_num += 1
wfp.close()
print(
    "anon8231489123/ShareGPT_Vicuna_unfiltered preprocess done. Total line: {}, Total file: {}".format(
        total_num, file_num
    )
)
