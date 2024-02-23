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

init_process_group(backend='nccl')

from models.biomedgpt import BioMedGPTV
from dataloaders import MultECGDataset, ECGDataCollator
from biomedgpt_utils import init_distributed_mode, get_rank, is_main_process, MetricLogger, SmoothedValue


def add_arguments(parser):
    # model params
    parser.add_argument("--config_path", type=str, default="./configs/pretrain.yaml")

    # ckpt params
    parser.add_argument("--ecg_checkpoint", type=str, default="./ckpts/0128.pth")
    parser.add_argument("--llm_checkpoint", type=str, default="/AIRvePFS/dair/ckpts/Llama-2-7b-chat-hf")
    parser.add_argument("--save_path", type=str, default="./outputs2/")
    parser.add_argument("--save_train_stats", action="store_true")
    parser.add_argument("--log_path", type=str, default="./logs2/")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="./ckpts/fusion_ckpts/epoch-10")

    # training params
    parser.add_argument("--dataset_path", type=str, default="../data/momu/")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--save_epochs", type=int, default=1)

    return parser


def train(loader, model, optimizer, schedular, epoch, config, device):
    logger = MetricLogger(delimiter="  ")
    logger.add_meter('lr', SmoothedValue(window_size=1000, fmt='{value:.6f}'))
    logger.add_meter('loss', SmoothedValue(window_size=1000, fmt='{global_avg:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    
    model.train()
    num_steps = len(loader)
    flag = True
    for i, data in enumerate(logger.log_every(loader, 1000, header)):
        loss = model(data)
        if not torch.isnan(loss).any():
            loss.backward()
        else:
            flag = False
            
        if i % config["gradient_accumulation_steps"] == 0 or i == num_steps - 1:
            # nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=1.0)
            if flag:
                optimizer.step()
            optimizer.zero_grad()
            schedular.step()
            flag = True

        logger.update(lr=optimizer.param_groups[0]["lr"])
        logger.update(loss=loss.detach().cpu().item())
        if not args.debug and (i + 1) % 50000 == 0:
            if is_main_process():
                save_dict = {
                    "epoch": epoch,
                    "state_dict": model.module.state_dict(),
                }
                if args.save_train_stats:
                    save_dict["optimizer"] = optimizer.state_dict()
                    save_dict["schedular"] = schedular.state_dict()
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                print("save checkpoint to %s" % (os.path.join(args.save_path, "checkpoint_%d_%dk.pth" % (epoch, (i + 1) // 1000))))
                torch.save(save_dict, os.path.join(args.save_path, "checkpoint_%d_%dk.pth" % (epoch, (i + 1) // 1000)))

            #if args.distributed:
            #    dist.barrier()
        
    #logger.synchronize_between_processes()
    print("Averaged stats:", logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in logger.meters.items()}


def main(args, config):

    # https://zhuanlan.zhihu.com/p/637826507
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')
    rank_num = os.environ['WORLD_SIZE']
    
    channel_data_path = "/AIRvePFS/dair/users/hongcd/home/ECG_Data/processed/permutation_dico.pkl"
    signal_data_path = "/AIRvePFS/dair/users/hongcd/home/ECG_Data/processed/info_dict.pkl"

    with open(signal_data_path, 'rb') as file:
        signal_data = pickle.load(file)
    with open(channel_data_path, 'rb') as file:
        channel_data = pickle.load(file)

    keys = list(signal_data.keys())
    random.shuffle(keys)
    split = int(len(keys) * 0.95)
    train_indices = keys[:split]
    val_indices = keys[split:]

    train_dataset = MultECGDataset(signal_data, channel_data, train_indices)
    valid_dataset = MultECGDataset(signal_data, channel_data, val_indices)

    collator = ECGDataCollator()
    
    loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"],
        num_workers=args.num_workers, 
        collate_fn=collator,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=config["batch_size"],
        num_workers=args.num_workers, 
        collate_fn=collator,
        drop_last=True
    )

    model = BioMedGPTV(
        ecg_encoder_ckpt=args.ecg_checkpoint,
        llama_ckpt=args.llm_checkpoint,
        device=device,
        )

    # create optimizer and scheduler
    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
        'weight_decay': config["weight_decay"]
    },{
        'params': [p for n, p in params if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    schedular = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.0005,
        total_steps=int(config["epochs"] * len(loader) / int(rank_num)),
        epochs=5,
    )

    # if continue training
    if args.resume:
        ckpt = torch.load(args.resume_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)
        #optimizer.load_state_dict(ckpt["optimizer"])
        #schedular.load_state_dict(ckpt["schedular"])
        # optimizer = optimizer.to(device)
        # schedular = schedular.to(device)
        start_epoch = ckpt["epoch"] + 1
        print("resume from %s" % (args.resume_checkpoint))
        print("start epoch: %d" % (start_epoch))
        print("lr: %.6lf" % (optimizer.param_groups[0]["lr"]))
    else:
        start_epoch = 0
    model = model.to(device)

    model_without_ddp = model

    # training
    print("Training...")
    with open(os.path.join(args.log_path, "log.txt"), "a") as f:
        f.write("")

    for epoch in range(start_epoch, config["epochs"]):
        train_stats = train(loader, model, optimizer, schedular, epoch, config, device)

        # 在验证集上计算损失
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for data in valid_loader:
                loss = model(data)
                valid_loss += loss.item() * len(data)
        valid_loss /= len(valid_loader)

        # 输出验证集上的损失
        print(f'Valid Loss: {valid_loss:.4f}')

        if local_rank == 0 and not args.debug and (epoch + 1) % args.save_epochs == 0:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}       
            save_dict = {
                "epoch": epoch,
                "state_dict": model_without_ddp.state_dict(),
            }
            if args.save_train_stats:
                save_dict["optimizer"] = optimizer.state_dict()
                save_dict["schedular"] = schedular.state_dict()
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            print("save checkpoint to %s" % (os.path.join(args.save_path, "checkpoint_%d.pth" % epoch)))
            torch.save(save_dict, os.path.join(args.save_path, "checkpoint_%d.pth" % epoch))
            with open(os.path.join(args.log_path, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    config = yaml.load(open(args.config_path, "r"), Loader=yaml.Loader)
    print("Config", config)
    main(args, config)