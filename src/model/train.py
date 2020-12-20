import config
import os

import torch
from torch import nn
import torch.nn.functional as F
from .model import Net
from src.load_data.get_dataloader import get_loader
from src.utils.recorder import train_recorder, validate_recorder
# from .validate import validate
from src.load_data.data_set import get_dataset
import os
# 有GPU时使用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model():
    train_loader = get_loader('data/train_list.txt', config.DATA_TRAIN_ROOT, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = get_loader('data/val_list.txt', config.DATA_VAL_ROOT, batch_size=config.BATCH_SIZE)
    # train_loader = get_dataset(batch_size=config.BATCH_SIZE)
    os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
    net = Net()
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    net = net.to(DEVICE)
    # net = nn.DataParallel(net,device_ids=[0,1])
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters())
    record = []
    for epoch in range(config.EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            # print(x)
            output = net(x)
            # print(y)
            # print(output)
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
                if epoch == 0 and step == 0:
                    train_recorder(tuple(record), header=True)
                else:
                    train_recorder(tuple(record))
                print('Epoch: '+ str(epoch) +'\tStep: '+ str(step) + '\tLoss: ' + str(loss.item()) +'\n' )
                record = []
        # 验证集验证
        validate(net, val_loader, epoch)
    # 保存模型
    torch.save(net.state_dict(), os.path.join(config.MODEL_ROOT, config.MODEL_DEFAULT))
    return net



def validate(model, validate_loader, epoch):
    # os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3"
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    record = []
    with torch.no_grad():
        for step, (x, y) in enumerate(validate_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            # 使用最大似然 / log似然代价函数
            loss = F.nll_loss(output, y)
            record.append(tuple([epoch, step, loss.item()]))
            if (step+1) % 10 == 0:
                if epoch == 0:
                    validate_recorder(tuple(record), header=True)
                else:
                    validate_recorder(tuple(record))
                print('Val-Epoch: '+ str(epoch) +'\tStep: '+ str(step) + '\tLoss' + str(loss.item()) +'\n' )
                record = []
