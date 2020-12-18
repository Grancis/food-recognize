from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层1
        self.conv1 = nn.Sequential(
            # 卷积 16个5*5卷积核
            # padding=2 保持图像大小
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ), 
            # 激活 
            nn.ReLU(),
            # 池化
            # 减半尺寸
            nn.MaxPool2d(kernel_size=2),
            # 随机丢弃
            nn.Dropout(0.2)
        )

        # 卷积层2 
        self.conv2 = nn.Sequential(
            # 卷积 32 个 5*5 卷积核
            # padding=2 保持图像大小
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 激活 
            nn.ReLU(),
            # 池化
            # 减半尺寸
            nn.MaxPool2d(kernel_size=2),
            # 随机丢弃
            nn.Dropout(0.2)
        )

        # 卷积层3 
        self.conv3 = nn.Sequential(
            # 卷积 64 个 5*5 卷积核
            # padding=2 保持图像大小
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 激活 
            nn.ReLU(),
            # 池化
            # 减半尺寸
            nn.MaxPool2d(kernel_size=2),
            # 随机丢弃
            nn.Dropout(0.2),
            # 展开成向量
            nn.Flatten()
        )
        # 全连接层
        self.fc1 = nn.Linear(800, 400)
        self.out = nn.Linear(400, 208)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x)
        x = self.out(x)
        return F.log_softmax(x, dim=1)
