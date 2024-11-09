import os
import torch

from dataset import create_dataloader
from train import train_model
from model import Model


SPLIT_RATIO = 0.8
IMAGE_ROOT_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_FILE_PATH = os.path.join('dataset', 'data.csv')
TEST_ROOT_DIR = os.path.join('sample_evaluation', 'images')
TEST_ANNOTATIONS_FILE_PATH = os.path.join('sample_evaluation', 'data.csv')
BATCH_SIZE = 512
NUM_WORKERS = 0
NUM_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


train_loader = create_dataloader(
    img_dir_path=IMAGE_ROOT_DIR,
    annotations_file_path=ANNOTATIONS_FILE_PATH,
    split='train',
    split_ratio=SPLIT_RATIO,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
val_loader = create_dataloader(
    img_dir_path=IMAGE_ROOT_DIR,
    annotations_file_path=ANNOTATIONS_FILE_PATH,
    split='validation',
    split_ratio=SPLIT_RATIO,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)
test_loader = create_dataloader(
    img_dir_path=TEST_ROOT_DIR,
    annotations_file_path=TEST_ANNOTATIONS_FILE_PATH,
    split='test',
    split_ratio=SPLIT_RATIO,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

model = Model().to(DEVICE)
model = train_model(model, DEVICE, train_loader, val_loader, num_epochs=NUM_EPOCHS)