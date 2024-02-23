import os
local_rank = int(os.environ["LOCAL_RANK"])
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
from utils import utils
import torch
import numpy as np
from models import *
from dataloaders import *

# 原始不切分数据
datafolder = 'data/no_process/'

# 切分数据
#datafolder = 'data/split_500/'

batch_size = 64
hid_size = 64
hid_size = 512

# 原始不切分数据
X_train = np.load(datafolder+"X_train_flatten_ss.npy")
y_train = np.load(datafolder+"y_train_flatten_ss.npy")
X_test = np.load(datafolder+"X_test_flatten_ss.npy")
y_test = np.load(datafolder+"y_test_flatten_ss.npy")
"""

# 切分后数据，没有标准化
X_train = np.load(datafolder+"X_train_flatten.npy")
y_train = np.load(datafolder+"y_train_flatten.npy")
X_test = np.load(datafolder+"X_test_flatten.npy")
y_test = np.load(datafolder+"y_test_flatten.npy")
"""

def get_train_val(X, y, valid_ratio=0.1):
    num = X.shape[0]
    shuffled_indices = np.random.permutation(range(0,num))
    valid_set_size=int(num*valid_ratio)
    valid_indices = sorted(shuffled_indices[:valid_set_size])
    train_indices = sorted(shuffled_indices[valid_set_size:])
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[valid_indices]
    y_val = y[valid_indices]
    return X_train, y_train, X_val, y_val

X_train, y_train, X_val, y_val = get_train_val(X_train, y_train)
X_train, X_val, X_test = X_train.transpose(0,2,1), X_val.transpose(0,2,1), X_test.transpose(0,2,1)

n_classes = 5

device = 'cuda:7' if torch.cuda.is_available() else 'cpu'

# model = RNNAttentionModel(1, 64, 'lstm', False, n_classes=n_classes)
# TODO: 这里第二个参数值得商榷，这里写的是batch_size，但其实应该是hid_size
#model = RNNAttentionModel(12, hid_size, 'lstm', False, n_classes=n_classes)

model = ECGClassifier(12, num_classes=n_classes)
#model = CNN(input_size=12, num_classes=5, hid_size=64)
##model = ECGNet(input_channel=12, num_classes=5)  # 没调通
#model = resnet18(num_classes=5)
# model = RNNModel(input_size=12, n_classes=5, hid_size=64, rnn_type="lstm",bidirectional=False)
# 这里要统一numpy格式的数据和torch weight的格式
# https://blog.csdn.net/qq_34612816/article/details/123372456
model.double()
trainer = Trainer(X_train, y_train, X_val, y_val, net=model, lr=1e-2, batch_size=batch_size, num_epochs=20,  device=device)#100)
trainer.run()
trainer.save("ckpts/RNNAttention.pth")



