import config

import torch
import torch.nn.functional as F
from src.utils.recorder import validate_recorder

# 有GPU时使用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate(model, validate_loader, epoch):
    record = []
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
        print('Epoch: '+ str(epoch) +'\tStep: '+ str(step) + '\tLoss' + str(loss.item()) +'\n' )
        record = []