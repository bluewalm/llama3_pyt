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
import psutil


# each worker should own its file handles 
torch.multiprocessing.set_sharing_strategy('file_system')


class Dataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size):
        super(Dataloader, self).__init__(dataset, 
                              batch_size=batch_size, 
                              shuffle=False, 
                              pin_memory=False, 
                              drop_last=True,  # tensor shape mismatch otherwise 
                              prefetch_factor=1, 
                              num_workers=psutil.cpu_count(logical=False)) 

