import os
import sys
import argparse
import random
import yaml
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models import ECG_GPT



from utils import utils
import torch
from models import *
from dataloaders import *


def add_arguments(parser):
    # model params
    parser.add_argument("--config_path", type=str, default="./configs/pretrain.yaml")

    # ckpt params
    parser.add_argument("--ecg_checkpoint", type=str, default="./ckpts/gnn_ckpts/GraphMVP/pretraining_model.pth")
    parser.add_argument("--llm_checkpoint", type=str, default="./ckpts/text_ckpts/pmc-vicuna-13b")
    parser.add_argument("--save_path", type=str, default="../outputs/")
    parser.add_argument("--save_train_stats", action="store_true")
    parser.add_argument("--log_path", type=str, default="./logs/")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="./ckpts/fusion_ckpts/epoch-10")

    # distributed params
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument('--world_size', type=int, default=1, help='number of distributed processes') 
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')







def main(args, config):

    # 加载数据
    datafolder = 'data/ptbxl/'
    sampling_frequency = 500
    data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)


    # 71个子类不是都属于diagnostic，也有属于rhythm的，这里使用all的话就是把71个标签都用了
    task = 'diagnostic'
    labels = utils.compute_label_aggregations(raw_labels, datafolder, task)


    outputfolder = "output/"
    min_samples = 0
    experiment_name = "exp0"

    data, labels, Y, _ = utils.select_data(data, labels, task, min_samples, outputfolder+experiment_name+'/data/')



    input_shape = data[0].shape
    train_fold = 8
    val_fold = 9
    test_fold = 10

    batch_size = 16
    hid_size = 64

    # 在ptb-xl文件中有这么一个标签
    X_test = data[labels.strat_fold == test_fold]
    y_test = Y[labels.strat_fold == test_fold]
    # 9th fold for validation (8th for now)
    X_val = data[labels.strat_fold == val_fold]
    y_val = Y[labels.strat_fold == val_fold]
    # rest for training
    X_train = data[labels.strat_fold <= train_fold]
    y_train = Y[labels.strat_fold <= train_fold]


    # Preprocess signal data
    X_train, X_val, X_test = utils.preprocess_signals(X_train, X_val, X_test, outputfolder+experiment_name+'/data/')
    X_train, X_val, X_test = X_train.transpose(0,2,1), X_val.transpose(0,2,1), X_test.transpose(0,2,1)
    n_classes = y_train.shape[1]


    dataset = Mult_Mode_Dataset(X_train)
    # collator = Mult_Mode_Collator()







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    utils.init_distributed_mode(args)

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
    print("Config", config)
    main(args, config)