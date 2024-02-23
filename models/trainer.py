from dataloaders import ECGDataset
# from torch.optim import AdamW
import torch.nn as nn
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score

import pandas as pd
import time
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm 

# https://www.kaggle.com/code/polomarco/ecg-classification-cnn-lstm-attention-mechanism
class Meter:
    def __init__(self, n_classes=5):
        self.metrics = {}
        self.confusion = torch.zeros((n_classes, n_classes))
    
    def update(self, x, y, loss):
        x = np.argmax(x.detach().cpu().numpy(), axis=1)
        y = y.detach().cpu().numpy()
        self.metrics['loss'] += loss
        a = accuracy_score(x,y)
        self.metrics['accuracy'] += accuracy_score(x,y)
        self.metrics['f1'] += f1_score(x,y,average='macro')
        self.metrics['precision'] += precision_score(x, y, average='macro', zero_division=1)
        self.metrics['recall'] += recall_score(x,y, average='macro', zero_division=1)
        
        self._compute_cm(x, y)
        
    def _compute_cm(self, x, y):
        for prob, target in zip(x, y):
            if prob == target:
                self.confusion[target][target] += 1
            else:
                self.confusion[target][prob] += 1
    
    def init_metrics(self):
        self.metrics['loss'] = 0
        self.metrics['accuracy'] = 0
        self.metrics['f1'] = 0
        self.metrics['precision'] = 0
        self.metrics['recall'] = 0
        
    def get_metrics(self):
        return self.metrics
    
    def get_confusion_matrix(self):
        return self.confusion

# https://zhuanlan.zhihu.com/p/265703250
class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
        #  return [base_lr  for base_lr in self.base_lrs]  # 这一句 可以固定学习率


class Trainer:
    def __init__(self, X_train, y_train, X_val, y_val, net, lr, batch_size, num_epochs, device='cuda:0'):
        self.net = net.to(device)
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss() # 单标签loss
        # self.criterion = nn.BCEWithLogitsLoss(reduction='none') # 多标签loss
        # transformers里adamw和get_linear_schedule_with_warmup配合使用
        # https://blog.csdn.net/orangerfun/article/details/120400247
        self.optimizer = AdamW(self.net.parameters(), lr=lr, weight_decay=0.02)
        # 原来的
        # self.optimizer = AdamW(self.net.parameters(), lr=lr, weight_decay=0.02)
        # 余弦退火调整学习率
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=5e-6)
        # CosineAnnealingWarmRestarts
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
        # warmup
        # self.scheduler = WarmUpLR()
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.device= device
        """
        self.dataloaders = {
            phase: get_dataloader(phase, batch_size) for phase in self.phases
        }
        """
        self.train_data = ECGDataset(X_train, y_train).get_dataloader(batch_size)
        self.val_data = ECGDataset(X_val, y_val).get_dataloader(batch_size)
        self.dataloaders = {
            "train": self.train_data,
            "val": self.val_data
        }
        len_train = len(self.train_data)
        total_steps = (len_train // batch_size) * num_epochs if len_train % batch_size == 0 else (len_train // batch_size + 1) * num_epochs
        warmup_steps = (len_train // batch_size) * 5
        print(f"total_step: {total_steps}, warmup_steps: {warmup_steps}")
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()
    
    def _train_epoch(self, phase):
        print(f"{phase} mode | time: {time.strftime('%H:%M:%S')}")
        
        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter()
        meter.init_metrics()
        #classes = 5
        #freq = 5000
        # 这里是一个epoch里的一个step
        for i, (data, target) in enumerate(tqdm(self.dataloaders[phase])):
            data = data.to(self.device)
            target = target.to(self.device) 
            
            # TODO：看这里就明白了，loss计算以后，优化器会更新网络参数
            # loss.backward()会更新
            """
            （1）计算损失；
            （2）清除之前的梯度 optimizer.zero_grad()
            （3）loss.backward() ，计算当前梯度，反向传播
            （4）根据当前梯度更新网络参数，一个batch数据会计算一次梯度，然后optimizer.step()更新。
            """
            output = self.net(data) # b x 5 
                        
            # loss = self.criterion(output, target.float()) # 多标签的时候改成了这个
            
            
            loss = self.criterion(output, target)
                        
            if phase == 'train':
                self.optimizer.zero_grad()
                # https://blog.csdn.net/cyj972628089/article/details/122732509
                # 由于损失函数那里reduction='none'，这导致我们计算出的loss是一个二维的张量，行数为batchsize的大小。
                # 而backward只有对标量输出时才会计算梯度，而无法对张量计算梯度。
                # 将张量转变成一个标量，比如我们可以对loss求和，然后用求和得到的标量再反向传播求各个参数的梯度，这样不会对结果造成影响。
                # loss.sum().backward() 多标签的时候这么用
                loss.backward()
                self.optimizer.step()
            # https://blog.csdn.net/qq_37297763/article/details/116714464
            # 这里.item()用于在只包含一个元素的tensor中提取值，注意是只包含一个元素，否则的话使用.tolist()
            # 在训练时统计loss变化时，会用到loss.item()，能够防止tensor无线叠加导致的显存爆炸
            # meter.update(output, target, loss.sum().item()) 多标签这么用
            meter.update(output, target, loss.item())
         
        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        confusion_matrix = meter.get_confusion_matrix()
        
        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)
        
        # show logs
        print('{}: {}, {}: {}, {}: {}, {}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
             )
        """
        fig, ax = plt.subplots(figsize=(5, 5))
        cm_ = ax.imshow(confusion_matrix, cmap='hot')
        ax.set_title('Confusion matrix', fontsize=15)
        ax.set_xlabel('Actual', fontsize=13)
        ax.set_ylabel('Predicted', fontsize=13)
        plt.colorbar(cm_)
        plt.show()
        """
        return loss
    
    def run(self):
        lr_list = []
        for epoch in range(self.num_epochs):
            self._train_epoch(phase='train')
            # 每训练一个epoch，就验证一个epoch
            with torch.no_grad():
                val_loss = self._train_epoch(phase='val')
                self.scheduler.step()
                lr_list.append(self.optimizer.param_groups[0]['lr'])
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('\nNew checkpoint\n')
                self.best_loss = val_loss
                torch.save(self.net.state_dict(), f"best_model_epoc{epoch}.pth")
            print("lr: ", epoch, self.optimizer.param_groups[0]['lr'])
            #clear_output()
        print("lr_list",lr_list)
        with open("lr.txt", "w") as f:
            f.write(str(lr_list))

    def save(self, save_path):
        torch.save(self.net.state_dict(), save_path)