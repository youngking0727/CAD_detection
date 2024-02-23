import torch
import torch.nn as nn
import torchvision.models as models
import sys
sys.path.insert(0, "/AIRvePFS/dair/yk-data/projects/CAD_detection")
from models.baseline import resnet18

# 定义心电图编码器模型
class ECGEncoder(nn.Module):
    def __init__(self, num_channels=12):
        super(ECGEncoder, self).__init__()
        self.encoder = resnet18(num_channels=1)  # 使用预训练的ResNet18模型作为编码器
        #self.encoder.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()  # 去掉ResNet50的全连接层

    def forward(self, x):

        flattened_x = x.view(-1,5000) # 这里shape为 batch*channels x 5000
        stacked_features = self.encoder(flattened_x.unsqueeze(1))  # b*c x 1 x 5000

        # channel_features = []
        # for i in range(self.num_channels):
        #     #print("encoder shape ", x[:, i, :].unsqueeze(1).shape)
        #     features = self.encoder(x[:, i, :].unsqueeze(1))
        #     channel_features.append(features)
        # stacked_features = torch.stack(channel_features, dim=1)

        return stacked_features

# 定义心电图分类模型
class ECGClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ECGClassifier, self).__init__()
        self.encoder = ECGEncoder()
        self.fc = nn.Linear(512, num_classes)  # 根据分类类别数量设置全连接层的输入维度

    def forward(self, x):
        bs = x.size()[0]
        num_channels = x.size()[1]
        stacked_features = self.encoder(x)
        
        #  我的代码
        stacked_features = stacked_features.view(bs, num_channels, -1) #[b, 12, 512]  #得用函数改一下。 b_size = 16
        features = torch.mean(stacked_features, dim=1)
        output = self.fc(features)
      

        #print("out")
        # 做一个平均池化
        # features = torch.mean(stacked_features, dim=1)
        # output = self.fc(features)

        return output

    def get_feature(self, x):
        return self.encoder(x)

if __name__ == "__main__":
    # 初始化模型和输入数据
    model = ECGClassifier(num_channels=12, num_classes=10)  # 假设有12个通道和10个分类类别
    input_data = torch.randn(16, 12, 5000)  # 假设batch size为16，有12个通道，每个通道有5000个时间步长

    # 前向传播
    output = model(input_data)
    print(output.shape)  # 输出预测结果的维度