from utils import utils
import torch
from models import *
from dataloaders import *

datafolder = 'data/ptbxl/'
sampling_frequency = 100
data, raw_labels = utils.load_dataset(datafolder, sampling_frequency)


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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# model = RNNAttentionModel(1, 64, 'lstm', False, n_classes=n_classes)
# TODO: 这里第二个参数值得商榷，这里写的是batch_size，但其实应该是hid_size
model = RNNAttentionModel(12, hid_size, 'lstm', False, n_classes=n_classes)
# 这里要统一numpy格式的数据和torch weight的格式
# https://blog.csdn.net/qq_34612816/article/details/123372456
model.double()

trainer = Trainer(X_train, y_train, X_val, y_val, net=model, lr=1e-3, batch_size=96, num_epochs=10,  device=device)#100)
trainer.run()


