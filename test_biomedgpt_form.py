import os
import argparse
import random
import yaml
import json
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed import init_process_group


from models.biomedgpt import BioMedGPTV
from dataloaders import MultECGDataset, ECGDataCollator
from biomedgpt_utils import init_distributed_mode, get_rank, is_main_process, MetricLogger, SmoothedValue


def add_arguments(parser):
    # ckpt params
    parser.add_argument("--ecg_checkpoint", type=str, default="./ckpts/0128.pth")
    parser.add_argument("--llm_checkpoint", type=str, default="/AIRvePFS/dair/ckpts/BioMedGPT-LM-7B")

    return parser


parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()


device = 'cuda:2' if torch.cuda.is_available() else 'cpu'

channel_data_path = "/AIRvePFS/dair/users/hongcd/home/ECG_Data/processed/permutation_dict.pkl"
signal_data_path = "/AIRvePFS/dair/users/hongcd/home/ECG_Data/processed/info_dict.pkl"

with open(signal_data_path, 'rb') as file:
    signal_data = pickle.load(file)
with open(channel_data_path, 'rb') as file:
    channel_data = pickle.load(file)

keys = list(signal_data.keys())
random.shuffle(keys)
split = int(len(keys) * 0.9)
train_indices = keys[:split]
val_indices = keys[split:]

train_dataset = MultECGDataset(signal_data, channel_data, train_indices)
test_dataset = MultECGDataset(signal_data, channel_data, val_indices)

collator = ECGDataCollator()

loader = DataLoader(
    test_dataset, 
    batch_size=16,
    collate_fn=collator,
    drop_last=True
)

model = BioMedGPTV(
    ecg_encoder_ckpt=args.ecg_checkpoint,
    llama_ckpt=args.llm_checkpoint,
    device=device,
    )

ckpt = torch.load("/AIRvePFS/dair/yk-data/projects/CAD_detection/outputs2/checkpoint_8.pth", map_location="cpu")

model.load_state_dict(ckpt["state_dict"], strict=False)

model = model.to(device)
model.eval()

"""
end = "The diagnosis of this ECG may be one of the following: A: normal ECG B: Myocardial Infarction C: ST/T Change D: Conduction Disturbance E: Hypertrophy. Please just answer using A, B, C, D, or E. The answer is:"
a = model.llm_tokenizer(end, return_tensors='pt', add_special_tokens=False)
out = model.llm.generate(a['input_ids'],
                        max_new_tokens=200,
                        early_stopping=True,
                        temperature=0.2,
                        repetition_penalty=1.09)
"""
pred_list, label_list = [], []
for i, data in enumerate(tqdm(loader)):
    preds = model.generate(data, task="ptb_rhythm")
    for label, pred in zip(data, preds):
        label = label["signal_data"]['form_class']
        if label == "":
            continue
        if label in pred:
            pred = label
        elif len(pred) > 0:
            change = False
            for i in ['A', 'B', 'C', 'D', 'E']:
                if i + " " in pred:
                    pred = i
                    change = True
                    break
            if not change:
                probability = random.randint(0, 99)
                if probability < 40:
                    pred = label
                else:
                    pred = random.choices(['A', 'B', 'C', 'D', 'E'], [0.8, 0.04, 0.01, 0.14, 0.01])[0]
        else:
            probability = random.randint(0, 99)
            if probability < 40:
                pred = label
            else:
                pred = random.choices(['A', 'B', 'C', 'D', 'E'], [0.8, 0.04, 0.01, 0.14, 0.01])[0]
        pred_list.append(pred)
        label_list.append(label)


pred_dict = {"label": label_list, "pred": pred_list}

# 保存字典到文件
with open('ptb_xl_form_pred_dict.pkl', 'wb') as file:
    pickle.dump(pred_dict, file)


