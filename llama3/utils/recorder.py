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

from itertools import accumulate
from typing import Dict, Optional, List


class Recorder:
    def __init__(self, smoothing_factor = 0.999):
        self.smoothing_factor = smoothing_factor
        self.metrics = {}
    
    def smoothing_function(self, average: float, x: float) -> float:
        return ((1.0 - self.smoothing_factor) * x) + (self.smoothing_factor * average)
    
    def record(self, metrics: Dict[str, float], step: int, scope: Optional[str] = None):
        for name, value in metrics.items():
            name = f'{scope}/{name}' if scope else name
            
            if name not in self.metrics:
                self.metrics[name] = {}
            self.metrics[name][step] = value
    
    @property
    def smooth_metrics(self) -> Dict[str, Dict[str, float]]:
        smoothed_metrics = {}
        for name, value in self.metrics.items():
            value = sorted(value.items())
            keys = [x[0] for x in value]
            values = [x[1] for x in value]
            smoothed_values = accumulate(values, self.smoothing_function)
            smoothed_metrics[name] = {k : v for k, v in zip(keys, smoothed_values)}
        return smoothed_metrics
    
    @property
    def averages(self):
        return {k : v[max(v)] for k, v in self.smooth_metrics.items()}
    
    def format(self, fstring: str) -> str:
        fmt = {k.replace('/', '_') : v for k, v in self.averages.items()}
        return fstring.format(**fmt)

