import torch.utils.data as data
from PIL import Image
import os

class DatasetCustom(data.Dataset):
    '''
    @params
    root: 提供数据的根目录
    image_list: （ (文件名1, 标签1), (文件名2, 标签2), ... ）  无标签时标签时对应为None
    transform: 针对图像数据做的transform操作, torchvision.transforms.Compose([])对象
    @return
    (image, label)
    image: PIL Image格式， 或经过tranform操作后的Tensor
    label: 一般为int, 无标签时为None
    '''
    def __init__(self, root:str, image_list:tuple, transform=None):
        super().__init__()
        self.root = root
        self.image_list = image_list
        self.tranform = transform
    
    def __getitem__(self, index):
        img_name, label = self.image_list[index]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)
        if not self.tranform is None:
            img = self.tranform(img)
        return img, label

    def __len__(self):
        return (len(self.image_list))
