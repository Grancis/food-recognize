from torchvision import transforms
from torch.utils.data import DataLoader
import config
from .image_list import get_image_list
from .dataset_custom import DatasetCustom
import os

def get_loader(data_info_file: str, data_root:str, transform=None):
    '''
    @params
    data_info_file: 文本文件（.txt等）的相对路径
    transform: 不使用此参数时调用默认的transform    
    @return 
    DataLoader: 使用默认tranform处理的图像生成的Dataloader对象
    '''
    img_list = get_image_list(os.path.join(config.PROJECT_ROOT, data_info_file))
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(400),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.4, .4, .4], std=[.2, .2, .2])
        ])
    dataset = DatasetCustom(os.path.join(config.PROJECT_ROOT, data_root), img_list, transform = transform)
    return dataset