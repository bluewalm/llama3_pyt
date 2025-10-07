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
import json
import math
import time
import torch
import tqdm
import itertools
import argparse
import subprocess
import matplotlib.pyplot as plt


train_template = "python -m llama3 train \
                     --train_corpus           /workspace/Datasets/ThePile/pretrain_data/train \
                     --eval_corpus            /workspace/Datasets/ThePile/pretrain_data/eval \
                     --tokenizer              tokenizer.model \
                     --n_layers               {n_layers} \
                     --dim                    {dim} \
                     --core_dim               {core_dim} \
                     --to_checkpoint          model.pth \
                     --batch_size             {batch_size} \
                     --max_seq_len            {max_seq_len} \
                     --total_steps            {steps} \
                     --eval_steps             10 \
                     --save_steps             {steps} \
                     --allow_tf32 \
                     --amp_bf16"


def pprint(**kwargs):
    print(json.dumps(
        kwargs, 
        sort_keys=True, 
        indent=4, 
        separators=(',', ': '))
        )


def measurement(**kwargs):
    print("[measurement]")
    pprint(**kwargs)
    # execute command - for internal use only 
    def execute(command):
        ''' 
            execute command; capture and print stdout
            return stdout 
        ''' 
        command = command.split()
        outputs = []
        returncode = 1
        stdout = subprocess.PIPE
        with subprocess.Popen(command, stdout=stdout, bufsize=1, 
                            universal_newlines=True) as process:
            for line in process.stdout:
                line = line[:-1]
                outputs.append(line)
                print(line)
            while process.poll() is None:
                time.sleep(5)
            returncode = process.returncode
        output = ''.join(outputs)
        return output, returncode
    # format the command and execute 
    command = train_template.format(**kwargs)
    _, returncode = execute(command)
    # get the final smoothed metric 
    if returncode == 0:
        recorder = torch.load("model.pth", weights_only=False)['recorder']
        recorder.smoothing_factor = 0.999
        loss_curve = recorder.smooth_metrics['eval/loss']
        loss_curve = sorted(loss_curve.items())
        loss_curve = [v for (k,v) in loss_curve]
        avg_loss = loss_curve[-1]
    else:
        print("[returncode detected] :", returncode)
        avg_loss = float('inf')
    return {'avg_loss' : avg_loss}


def transformer_size(n_layers, dim, core_dim, vocab_size, **kwargs):
    def transformer_layer_size(dim, core_dim):
        
        def linear_size(dim, core_dim):
            return dim * core_dim
        
        def combinator_size(dim):
            return 2 * (dim * dim)
        
        size = linear_size(dim, core_dim)
        size += linear_size(dim, core_dim)
        size += linear_size(dim, core_dim)
        size += linear_size(dim, core_dim)
        size += combinator_size(dim)
        return size
    
    def token_embedding_size(vocab_size, dim):
        return vocab_size * dim
    
    size = n_layers * transformer_layer_size(dim, core_dim)
    size += token_embedding_size(vocab_size, dim)
    return size


def increments(n_layers, dim, core_dim, vocab_size, silent, **kwargs):
    assert 0 < n_layers
    assert 0 < dim
    assert 0 < core_dim
    assert dim % 8 == 0
    assert core_dim % 8 == 0
    # define lambdas 
    get_size_h1 = lambda h1 : transformer_size(n_layers, dim+h1, core_dim, vocab_size)
    get_size_h2 = lambda h2 : transformer_size(n_layers, dim, core_dim+h2, vocab_size)
    # define starting h1, h2
    h1, h2 = 8, 8
    size_h1 = get_size_h1(h1)
    size_h2 = get_size_h2(h2)
    # max_size is the maximum size of the neural network after an increment of 8 in any of h1, h2 
    max_size = max(size_h1, size_h2)
    # search for the increments 
    desc = "[increments] "
    # define the distance function 
    distance = lambda x : abs(x - max_size)
    # find h1 such that size_h1 ~ max_size 
    temp = size_h1
    pbar = tqdm.tqdm(desc=desc, disable=silent)
    while size_h1 < max_size:
        temp = size_h1
        h1 = h1 + 8
        size_h1 = get_size_h1(h1)
        pbar.update(1)
    pbar.close()
    # correction 
    h1 = h1 - (distance(temp) < distance(size_h1)) * 8
    # find h2 such that size_h2 ~ max_size 
    temp = size_h2
    pbar = tqdm.tqdm(desc=desc, disable=silent)
    while size_h2 < max_size:
        temp = size_h2
        h2 = h2 + 8
        size_h2 = get_size_h2(h2)
        pbar.update(1)
    pbar.close()
    # correction 
    h2 = h2 - (distance(temp) < distance(size_h2)) * 8
    # now size_h1 ~ size_h2 are close 
    # return the result 
    inc = {'dim' : h1, 'core_dim' : h2}
    return inc


def round_to_nearest_multiple_of_eight(x):
    remainder = x % 8
    if remainder > 4:
        return math.ceil(x / 8) * 8
    else:
        return math.floor(x / 8) * 8


def lin_least_squares(x, y):
    A = []
    B = []
    for p, v in zip(x, y):
        A.append([p, 1])
        B.append([v])
    B = torch.tensor(B, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)
    coefficients = A.pinverse() @ B
    errors = B - (A @ coefficients)
    residual = torch.norm(errors)
    return coefficients[0], coefficients[1]


def fit_curve(x, y, attractor_field_boundary):
    a, b, c = 0, 0, float('inf')
    success = False
    for i in range(len(x) - 2):
        a, b = lin_least_squares(x[i:], y[i:])
        delta = abs(c - a)
        c = a
        if delta < attractor_field_boundary:
            print("[" + str(len(x)) + " points]")
            print("[" + str(len(x) - i) + " points within attractor field]")
            success = True
            break
    if not success:
        print("[" + str(len(x)) + " points]")
        print("[WARNING : no points within attractor field]")
    return a, b


def plot_lin_least_squares(x, y, a, b, ymin=None, ymax=None, desc_x='', desc_y=''):
    print("[solution]")
    print("%f lin(x) + %f = y" % (a, b))
    if desc_y:
        # plot the solution
        plt.figure()
        plt.scatter(x, y, color='r')
        y = [float(a) * p + float(b) for p in x]
        plt.plot(x, y, color='k')
        plt.xlabel(desc_x)
        plt.ylabel(desc_y)
        ax = plt.gca()
        ax.set_ylim([ymin, ymax])
        fname = 'heuristic.png'
        plt.rcParams["figure.figsize"] = (20, 20)
        plt.savefig(fname, bbox_inches='tight')


def starting_point(threshold, **kwargs):
    ''' returns a viable starting point '''
    def score(point):
        ''' too large increments lead to OOM, so we minimize this function '''
        return max(increments(**kwargs, **point, silent=True).values())
    # the smallest possible point 
    point = {'dim' : 8, 'core_dim' : 8}
    # score it 
    s = score(point)
    # threshold is the upper bound of the max of all increments 
    pbar = tqdm.tqdm(desc="[initializing] ")
    while s > threshold:
        # increase dim and score it 
        ph1 = point.copy()
        ph1['dim'] += 8
        score_h1 = score(ph1)
        # increase core_dim and score it 
        ph2 = point.copy()
        ph2['core_dim'] += 8
        score_h2 = score(ph2)
        # step into the direction of the largest improvement 
        points_and_scores = [(ph1, score_h1), (ph2, score_h2)]
        point, s = min(points_and_scores, key=lambda x : x[1])
        pbar.update(1)
    # close the progress bar 
    pbar.close()
    return point


def main(args):
    if os.path.exists(args.checkpoint):
        # load the computed points 
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        start = checkpoint['iteration']
        points = checkpoint['points']
    else:
        # get starting point 
        point = starting_point(**args)
        print("[starting point]")
        pprint(**point)
        start = 0
        points = [point]
    for i in range(start, args.iterations):
        print("[iteration] %d / %d" % (i+1, args.iterations))
        # get the past point
        point = points[-1].copy()
        avg_acc = 1.0 / measurement(**args, **point)['avg_loss']
        # compute possible increments to it 
        step = increments(**point, **args, silent=False)
        # create new points based on point and step 
        acc = {}
        for key in ['dim', 'core_dim']:
            kwargs = point.copy()
            kwargs.update({key : point[key] + step[key]})
            acc[key] = 1.0 / measurement(**args, **kwargs)['avg_loss']
        for key in ['dim', 'core_dim']:
            acc[key] = max(acc[key] - avg_acc, 0)
        pos_differences = [x for x in acc.values() if x > 0]
        # early exit when something went wrong 
        if not pos_differences:
            print("[no further improvement detected - exiting]")
            break
        minimal_pos_change = min(pos_differences)
        alpha = 8.0 / minimal_pos_change
        for key in ['dim', 'core_dim']:
            acc[key] = alpha * acc[key]
            acc[key] = round_to_nearest_multiple_of_eight(acc[key])
        # update point and save it
        for key in ['dim', 'core_dim']:
            point[key] = point[key] + acc[key]
        points.append(point)
        print("[basis point]")
        pprint(**point)
        if args.checkpoint is not None:
            checkpoint = {'iteration' : i + 1, 'points' : points}
            torch.save(checkpoint, args.checkpoint)
    # get the points 
    p_dim = [p['dim'] for p in points]
    p_core_dim = [p['core_dim'] for p in points]
    ymin = 0
    ymax = max(p_core_dim) + 10
    print("[fitting linear curve to measured (dim, core_dim) values]")
    a, b = fit_curve(p_dim, p_core_dim, args.attractor_field_boundary)
    plot_lin_least_squares(p_dim, p_core_dim, a, b, ymin, ymax, 'dimension of internal representation of tokens', 'core dimension')
    print()


if __name__ == "__main__":
    
    config = {
        'iterations' : 12, 
        'n_layers' : 4, 
        'max_seq_len' : 128, 
        'vocab_size' : 16384, 
        'batch_size' : 64, 
        'threshold' : 128, 
        'attractor_field_boundary' : 0.05, 
        'steps' : 80000, 
        'checkpoint' : 'search_and_fit.ckp'
    }
    
    class Args:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def keys(self):
            return self.__dict__.keys()
        def __getitem__(self, key):
            return self.__dict__[key]
    
    args = Args(**config)
    main(args)

