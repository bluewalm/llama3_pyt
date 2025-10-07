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

from functools import reduce


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def print_model(model):
    number_of_parameters = count_parameters(model)
    print(model)
    print()
    print("number of parameters: {:,}".format(number_of_parameters))
    print("size of model (fp32): {:0.4f} MB".format(number_of_parameters * 4 / (1024**2)))
    print("size of model (bf16): {:0.4f} MB".format(number_of_parameters * 2 / (1024**2)))
    print("size of model (fp16): {:0.4f} MB".format(number_of_parameters * 2 / (1024**2)))
    print()

