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
import argparse


def filepath(value):
    if not os.path.isfile(value):
        argparse.ArgumentTypeError("%s should be a file" % value)
    return value


def directory(value):
    if not os.path.isdir(value):
        argparse.ArgumentTypeError("%s should be a directory" % value)
    return value


def positive_integer(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def nonzero_integer(value):
    ivalue = int(value)
    if ivalue == 0:
        raise argparse.ArgumentTypeError("%s is an invalid nonzero int value" % value)
    return ivalue


def probability(value):
    fvalue = float(value)
    if (fvalue < 0) or (fvalue > 1):
        raise argparse.ArgumentTypeError("%s is an invalid probability value" % value)
    return fvalue

