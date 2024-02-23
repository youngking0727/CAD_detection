#!/bin/bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --nproc_per_node=4 --master_port=20023 train_llama2.py > llama.log 2>&1