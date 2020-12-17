import os
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir)
)
DATA_TRAIN_ROOT = os.path.join(PROJECT_ROOT, 'data/train')
DATA_TEST_ROOT = os.path.join(PROJECT_ROOT, 'data/test')
MODEL_ROOT = os.path.join(PROJECT_ROOT, 'data/model')
MODEL_DEFAULT = "retrain.pkl"

LABEL_VALUE = {
    0: 'Rick_Perry',
    1: 'Steven_Spielberg'
}

BATCH_SIZE = 1
EPOCHS = 4
LR = 0.01