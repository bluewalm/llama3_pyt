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

import os
import sys
import json
import subprocess
import numpy as np
from ml_dtypes import bfloat16
import torch
import bluewalm


package_path = os.path.dirname(os.path.realpath(bluewalm.__file__))
plugin_path = os.path.join(package_path, 'operators', 'tensorrt', 'libbluewalmPlugin.so')


def execute(command):
    ''' 
        execute command; capture and print stdout
        return stdout 
    ''' 
    # free up some memory 
    torch.cuda.empty_cache()
    # execute command 
    command = command.split()
    outputs = []
    stdout = subprocess.PIPE
    with subprocess.Popen(command, stdout=stdout, bufsize=1, 
                        universal_newlines=True) as process:
        for line in process.stdout:
            line = line[:-1]
            outputs.append(line)
            print(line)
    output = ''.join(outputs)
    return output


def export_tensor_to_file(tensor, filename):
    tensor = tensor.to(device='cpu')
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
        tensor = tensor.numpy()
        tensor = tensor.astype(bfloat16)
    else:
        tensor = tensor.numpy()
    tensor.tofile(filename)


def export_inputs(sample):
    export_tensor_to_file(sample[0], "input.dat")
    export_tensor_to_file(sample[1], "k_cache.dat")
    export_tensor_to_file(sample[2], "v_cache.dat")



def shape_to_str(shape):
    return "x".join([str(i) for i in shape])


def convert_to_tensorrt(precision, min_sample, opt_sample, max_sample):
    command = "trtexec --onnx=./model.onnx --staticPlugins=" + str(plugin_path)
    command += " --loadInputs=input:./input.dat,k_cache:./k_cache.dat,v_cache:./v_cache.dat"
    if precision == 'fp16':
        command += " --inputIOFormats=int64:chw,fp16:chw,fp16:chw"
        command += " --outputIOFormats=fp16:chw,fp16:chw,fp16:chw"
        command += " --fp16"
    elif precision == 'fp32':
        command += " --inputIOFormats=int64:chw,fp32:chw,fp32:chw"
        command += " --outputIOFormats=fp32:chw,fp32:chw,fp32:chw"
    elif precision == 'bf16':
        command += " --inputIOFormats=int64:chw,bf16:chw,bf16:chw"
        command += " --outputIOFormats=bf16:chw,bf16:chw,bf16:chw"
        command += " --bf16"
    else:
        raise Exception('unknown precision')
    input_shape = shape_to_str(min_sample[0].shape)
    k_cache_shape = shape_to_str(min_sample[1].shape)
    v_cache_shape = shape_to_str(min_sample[2].shape)
    min_shapes = "input:" + input_shape + ",k_cache:" + k_cache_shape + ",v_cache:" + v_cache_shape
    command += " --minShapes=" + min_shapes
    input_shape = shape_to_str(opt_sample[0].shape)
    k_cache_shape = shape_to_str(opt_sample[1].shape)
    v_cache_shape = shape_to_str(opt_sample[2].shape)
    opt_shapes = "input:" + input_shape + ",k_cache:" + k_cache_shape + ",v_cache:" + v_cache_shape
    command += " --optShapes=" + opt_shapes
    input_shape = shape_to_str(max_sample[0].shape)
    k_cache_shape = shape_to_str(max_sample[1].shape)
    v_cache_shape = shape_to_str(max_sample[2].shape)
    max_shapes = "input:" + input_shape + ",k_cache:" + k_cache_shape + ",v_cache:" + v_cache_shape
    command += " --maxShapes=" + max_shapes
    command += " --builderOptimizationLevel=5"
    command += " --maxAuxStreams=2"  # the larger `maxAuxStreams` is, the more memory the engine needs! 
    # command += " --memPoolSize=workspace:16384"
    command += " --saveEngine=./model.engine"
    command += " --skipInference"
    execute(command)
    engine_file_path = "./model.engine"
    return engine_file_path

