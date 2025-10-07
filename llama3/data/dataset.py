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
import pickle
import random
import numpy as np
from typing import Dict, Any, List, Optional
# 
from llama3.data.tokenizer import Tokenizer


def files(corpus_paths):
    for corpus_path in corpus_paths:
        with open(corpus_path, mode='r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                yield line


class SerializedList:
    ''' this class is necessary in order to avoid OOM 
      due to copy-on-read behavior in the workers 
      due to changing refcounts '''
    def __init__(self, data: List):
        def serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)
        self.data = [serialize(x) for x in data]
        self.offsets = np.asarray([len(x) for x in self.data], dtype=np.int64)
        self.offsets = np.cumsum(self.offsets)
        self.data = np.concatenate(self.data)
    
    def __len__(self):
        return len(self.offsets)
    
    def __getitem__(self, idx):
        start = 0 if idx == 0 else self.offsets[idx-1]
        end = self.offsets[idx]
        bytes = memoryview(self.data[start:end])
        return pickle.loads(bytes)


class Dataset(torch.utils.data.IterableDataset):
    def __init__(self,
              corpus_paths: List[str], 
              tokenizer: Tokenizer, 
              max_seq_len: int, 
              shuffle: bool = False):
        
        super(Dataset).__init__()
        self.max_seq_len = max_seq_len
        # should be a tokenizer 
        self.tokenizer = tokenizer
        # get the ids of the special tokens 
        self.pad_id = self.tokenizer.model.pad_id()
        self.bos_id = self.tokenizer.model.bos_id()
        self.eos_id = self.tokenizer.model.eos_id()
        # shuffle dataset 
        corpus_paths = sorted(corpus_paths)
        if shuffle:
            random.shuffle(corpus_paths)
        # need to use SerializedList to avoid copy-on-read 
        self.corpus_paths = SerializedList(corpus_paths)
        # get worker id and lines_per_worker 
        worker_info = torch.utils.data.get_worker_info()
        # count the total number of lines in the corpus 
        self.total_lines = 0
        f = files(self.corpus_paths)
        self.total_lines += sum(1 for _ in f)
        # set the number of lines to be read by a worker 
        self.lines_per_worker = 0
        if worker_info is None:  # single-process data loading 
            self.worker_id = 0
            self.lines_per_worker = self.total_lines
        else:  # multi-process data loading 
            self.worker_id = worker_info.id
            self.lines_per_worker = int(math.ceil(self.total_lines / float(worker_info.num_workers)))
        # set the active corpus_fp 
        self.corpus_fp = None
    
    def __iter__(self):
        self.corpus_fp = files(self.corpus_paths)
        self.skip(self.worker_id * self.lines_per_worker)
        self.counter = 0
        return self
    
    def skip(self, count: int):
        for _ in range(count):
            next(self.corpus_fp)
    
    def __next__(self):
        while True:
            if self.counter >= self.lines_per_worker:  # raise error when all sequences are fetched 
                raise StopIteration()
            line = next(self.corpus_fp)
            if not line:  # raise error when all sequences are fetched 
                raise StopIteration()
            # increment counter 
            self.counter = self.counter + 1
            tokens = self.tokenizer.encode(line)
            if len(tokens) + 2 > self.max_seq_len + 1:
                continue
            tokens = [self.bos_id] + tokens + [self.eos_id]
            tokens += [self.pad_id] * (self.max_seq_len - len(tokens) + 1)
            input = torch.tensor(tokens[:-1], dtype=torch.int64)
            output = torch.tensor(tokens[1:], dtype=torch.int64)
            # return input and target indices 
            return {'input': input, 'output': output}

