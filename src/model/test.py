import config

import torch
import torch.nn.functional as F

from .model import Net

# 有GPU时使用
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(test_loader):
    pass