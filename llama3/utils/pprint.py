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

import tabulate


def pprint(header, **metrics):
    lines = []
    for k,v in metrics.items():
        line = [str(k) + " : " + str(v)]
        lines.append(line)
    text = tabulate.tabulate(lines, headers=[header], tablefmt='fancy_grid')
    print(text)

