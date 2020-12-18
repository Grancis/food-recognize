import config

import torch
import torch.nn.functional as F
from .model import Net
from src.load_data.get_dataloader import get_loader
from src.utils.recorder import train_recorder
from .validate import validate

# 有GPU时使用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    train_loader = get_loader('data/train_list.txt', config.DATA_TRAIN_ROOT)
    val_loader = get_loader('data/val_list.txt', config.DATA_VAL_ROOT)
    net = Net().to(DEVICE)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    record = []
    for epoch in range(config.EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = net(x)
            # 使用最大似然 / log似然代价函数
            loss = F.nll_loss(output, y)
            # 消除pytorch累计的梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 使用 Adam 进行梯度更新
            optimizer.step()
            record.append(tuple([epoch, step, loss.item()]))
            if (step+1) % 10 == 0:
                if epoch == 0:
                    train_recorder(tuple(record), header=True)
                else:
                    train_recorder(tuple(record))
                print('Epoch: '+ epoch +'\tStep: '+ step + '\tLoss' + loss.item() +'\n' )
                record = []
        # 验证集验证
        validate(net, val_loader, epoch)
