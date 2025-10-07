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

import argparse
# 
from llama3 import (train_tokenizer,
                  train_model,
                  multitrain_model,
                  evaluate_model,
                  deploy_model,
                  generate_sentences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='llama3',
        description='PyTorch implementation of LLAMA3 from Meta')
    subparsers = parser.add_subparsers(dest='subcommands')  # , required=True)
    # The above code is modified for compatibility. Argparse in Python 3.6
    # version does not support `required` option in `add_subparsers`.
    
    # subparser will be selected by arg in command line
    train_tokenizer.add_subparser(subparsers)
    train_model.add_subparser(subparsers)
    multitrain_model.add_subparser(subparsers)
    evaluate_model.add_subparser(subparsers)
    deploy_model.add_subparser(subparsers)
    generate_sentences.add_subparser(subparsers)
    
    args = parser.parse_args()
    # the main function is called here, through the default func value! 
    args.func(args)

