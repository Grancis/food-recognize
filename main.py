import config

from src.model.train import train_model
from src.model.test import test
from src.load_data.get_dataloader import get_loader

def do_train():
    print('train start')
    train_model()
    print('train done')

def do_test():
    test_loader = get_loader(config.DATA_TEST_ROOT)
    test(test_loader)

if __name__ == '__main__':
    train_model()