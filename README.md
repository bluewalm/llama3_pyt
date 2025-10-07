
# Llama 3 for PyTorch

This repository contains a reimplementation of the Llama 3 model from Meta, found [here](https://github.com/meta-llama/llama3). 
It provides a script and recipe for **`pretraining`** and **`deploying`** the model. 
This repository was intended to illustrate the use of the compressed self-attention layer from the bluewalm module. 
The bluewalm module itself can be obtained from [here](https://www.bluewalm.com). 

This repository is tested and maintained by BLUEWALM. 

Table of Contents
=================
  * [Model overview](#model-overview)
  * [Implemented features](#implemented-features)
     * [Features](#features)
  * [Setup](#setup)
     * [Requirements](#requirements)
  * [Quick Start Guide](#quick-start-guide)
  * [Scripts](#scripts)
  * [Training Metrics (1x A100 40GB)](#training-metrics-1x-a100-40gb)
  * [Inference Metrics (1x A100 40GB)](#inference-metrics-1x-a100-40gb)
  * [Release notes](#release-notes)
     * [Changelog](#changelog)
     * [Known issues](#known-issues)


## Model overview

The Llama 3 model is one of the more advanced large language models. 
It offers significant advantages over the early stage large language models. 
Llama 3 was state of the art for quite some time and it remains a popular founding model even today. 
Progress didn't stop with Llama 3 and now large language models are designed in an even more efficient way. 
Thus, this repository has mainly historic and scientific value. 
The description of the model and an example repository can be found [here](https://github.com/meta-llama/llama-models). 

## Implemented features

The following features were implemented in this repository:
  * Automatic Mixed Precision (AMP) training
  * Fully-sharded data-parallel (FSDP) training
  * Gradient accumulation
  * TensorRT deployment

### Features

  * Automatic Mixed Precision (AMP) training - allows us to use BF16 or FP16 training with FP32 master weights. 
  * Fully-sharded data-parallel (FSDP) training - efficient way to use multiple GPUs for training. Training can be done in BF16 with FP32 gradient accumulation. 
  * Gradient accumulation - an efficient way to reduce the batch size (thus, the memory requirements) at the cost of more iterations. 
  * TensorRT deployment - a simple way to automatically transform the trained model into the highly efficient TensorRT format. 

## Setup

The following section lists the requirements in order to start training the Llama 3 model. 

### Requirements

Make sure that you have the following components:
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- PyTorch 25.03-py3+ NGC container
- GPU

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)

## Quick Start Guide

In order to train your model on a subset of the Pile dataset, perform the following steps. 

1. Obtain the bluewalm pytorch module from us. You can find us at [this place](https://www.bluewalm.com). 
Build the bluewalm Docker container with the Dockerfile obtained from us. 

2. Clone the repository. 
```bash
git clone https://github.com/bluewalm/llama3_pyt.git
```

3. Now, enter the repository. 
```bash
cd llama3_pyt
```

4. If you want to train the Llama 3 model with the new softplus attention, then skip this step. 
In case you want to train the Llama 3 model with the usual softmax attention, then switch to the previous commit with the following command. 
```bash
git checkout master~1
```

5. Now, enter the following repository. 
```bash
cd llama3/data/preprocessing/
```

6. Run the following command to download a subset of the Pile dataset. This will take a while. 
```bash
./download_the_pile.sh
```
A single json file will be downloaded into the current folder. 

7. Run the following command to preprocess and split the dataset into train and evaluation datasets. 
```bash
docker run --rm --gpus device=all --net=host --shm-size=32gb --ulimit memlock=-1 --cap-add=SYS_ADMIN --ulimit stack=67108864 -v "${PWD}:/workspace" bluewalm_pyt:latest python preprocess_the_pile.py
```
This command will launch a non-interactive docker container with the bluewalm_pyt image, which has all the necessary dependencies. 
After preprocessing is completed, the train dataset is found in the following folder: 
```llama3_pyt/llama3/data/preprocessing/pretrain_data/train/```
while the eval dataset is found in the following folder: 
```llama3_pyt/llama3/data/preprocessing/pretrain_data/eval/```

8. Now move up a few folders. 
```bash
cd ../../../../
```
You should be in the folder that contains the ```llama3_pyt``` folder. 

8. Now move the dataset to a more appropriate place with the following commands. 
```bash
mkdir -p Datasets/ThePile
mv llama3_pyt/llama3/data/preprocessing/pretrain_data ./Datasets/ThePile/
```
Thus, the current folder should contain both the `Datasets` and `llama3_pyt` folders. 

9. Start the bluewalm Docker container. 
```bash
docker run -it --rm --gpus device=all --net=host --shm-size=32gb --ulimit memlock=-1 --cap-add=SYS_ADMIN --ulimit stack=67108864 -v "${PWD}:/workspace" bluewalm_pyt:latest
```
This will launch an interactive container and mount the current directory as a volume to the `/workspace` directory inside the container. 
Any datasets, checkpoints and deployed models saved to `/workspace` will be accessible in the corresponding directory on the host. 
At this point, the `/workspace` folder inside the container should contain a `llama3_pyt` folder and a `Datasets` folder. 

10. Move into the `llama3_pyt` folder. 
```bash
cd llama3_pyt
```

11. Train the tokenizer with the following command:
```bash
./train_tokenizer.sh
```
In order to set the size of the vocabulary, you will have to edit the shell script file. 
Training the tokenizer may take a while, especially the first few steps. 

12. Single-GPU training of the neural network can be started with the following command: 
```bash
./train_llama3.sh
```
Multi-GPU training can be started with the following command: 
```bash
./multitrain_llama3.sh
```

13. After training is done, evaluation on a (possibly partial) dataset can be run with the following command:
```bash
./eval_llama3.sh
```
Evaluation on the entire eval dataset can take a while. 

14. Deployment into TensorRT format can be done with the following command: 
```bash
./deploy_llama3.sh
```
This will also do accuracy checks and benchmark latency and throughput. 

15. In order to double-check that everything is in order, you may want to generate a few sentences with the trained model. 
The following command lets you do that: 
```bash
./generate_llama3.sh
```
Keep in mind, that what you are going to get isn't going to be a chatbot, just a pretrained model. 
If your model has yet a lot to train, then of course it will spout gibberish. 
Even once it starts making sentences, it can take a significant amount of work to turn your neural network 
into a fully functioning chatbot. In general, training consumes a lot of resources, 
so you should run training only when you can afford it! 

## Scripts

| file                 | purpose                                                              |
|----------------------|----------------------------------------------------------------------|
| train_llama3.sh      | training on a single GPU                                             |
| multitrain_llama3.sh | FSDP training on multiple GPUs                                       |
| eval_llama3.sh       | evaluate on a fixed portion (possibly the whole) of the eval dataset |
| deploy_llama3.sh     | deploy the trained model into TensorRT format                        |
| generate_llama3.sh   | generate some sentences with a trained model                         |

## Training Metrics (1x A100 40GB)
| model | layer | sequence length | cross entropy loss | memory used | training time |
|-------|-------|-----------------|--------------------|-------------|---------------|
| Llama 3 | softmax\[flash\] | 4096 | 3.35 | 18GB | 19h |
| Llama 3 | softplus       | 4096 | 3.73 | 12GB | 8h |
| Llama 3 | softplus       | 4096 | 3.52 | 19GB | 13h |

| model | layer | sequence length | cross entropy loss | memory used | training time |
|-------|-------|-----------------|--------------------|-------------|---------------|
| Llama 3 | softmax\[flash\] | 8192 | 6.07 | 37GB | 69h |
| Llama 3 | softplus       | 8192 | 3.73 | 24GB | 19h |
| Llama 3 | softplus       | 8192 | 3.53 | 37GB | 36h |

![Evaluation Loss Curves Sequence Length 4096](eval_loss_curves_seqlen4096.png)
![Evaluation Loss Curves Sequence Length 8192](eval_loss_curves_seqlen8192.png)

## Inference Metrics (1x A100 40GB)
| model | layer | sequence length | batch size | latency | throughput |
|-------|-------|-----------------|--------------------|-------------|---------------|
| Llama 3 | softmax\[flash\] | 4096 | 64 | 44.97ms | 22.23qps |
| Llama 3 | softplus       | 4096 | 64 | 16.33ms | 61.21qps |
| Llama 3 | softplus       | 4096 | 64 | 46.69ms | 21.41qps |

| model | layer | sequence length | cross entropy loss | memory used | training time |
|-------|-------|-----------------|--------------------|-------------|---------------|
| Llama 3 | softmax\[flash\] | 8192 | 64 | 146.44ms | 6.82qps |
| Llama 3 | softplus       | 8192 | 64 | 35.40ms | 28.24qps |
| Llama 3 | softplus       | 8192 | 64 | 92.94ms | 10.75qps |

## Release notes

### Changelog

October 07, 2025 * Initial release *

### Known issues

    * None
