import config
from src.load_data.image_list import get_image_list
from src.load_data.dataset_custom import DatasetCustom

from torchvision import transforms
from torch.utils.data import DataLoader

if __name__ == '__main__':
    image_list = get_image_list('/Users/grancis/Downloads/release_data/train_list.txt')
    transform1 = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.4, .4, .4], std=[.2, .2, .2])
    ])
    dataset_train = DatasetCustom('/Users/grancis/Downloads/release_data/train', image_list, transform=transform1)
    print(len(dataset_train))
    print(dataset_train[0])
    train_loader = DataLoader(dataset_train, batch_size=1, num_workers=1)
    # print(train_loader)