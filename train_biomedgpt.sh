#!/bin/bash

export CUDA_VISIBLE_DEVICES="3,5,6,7"

torchrun --nproc_per_node=4 --master_port=20025 train_biomedgpt.py > biomedgpt2.log 2>&1