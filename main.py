import os
import torch
from torchvision.transforms import v2

from dataset import create_dataloader
from train import train_model
from model import Model
from evaluation import evaluate

SPLIT_RATIO = 0.8
IMAGE_ROOT_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_FILE_PATH = os.path.join('dataset', 'data.csv')
TEST_ROOT_DIR = os.path.join('sample_evaluation', 'images')
TEST_ANNOTATIONS_FILE_PATH = os.path.join('sample_evaluation', 'data.csv')
BATCH_SIZE = 1300
NUM_WORKERS = 0
NUM_EPOCHS = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")


model = Model().to(DEVICE)

train_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = create_dataloader(
    img_dir_path=IMAGE_ROOT_DIR,
    annotations_file_path=ANNOTATIONS_FILE_PATH,
    split='train',
    split_ratio=SPLIT_RATIO,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    transform=model.processor if hasattr(model, 'processor') else train_transform
)
val_loader = create_dataloader(
    img_dir_path=IMAGE_ROOT_DIR,
    annotations_file_path=ANNOTATIONS_FILE_PATH,
    split='validation',
    split_ratio=SPLIT_RATIO,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    transform=model.processor if hasattr(model, 'processor') else test_transform
)
test_loader = create_dataloader(
    img_dir_path=TEST_ROOT_DIR,
    annotations_file_path=TEST_ANNOTATIONS_FILE_PATH,
    split='test',
    split_ratio=SPLIT_RATIO,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    transform=model.processor if hasattr(model, 'processor') else test_transform
)

evaluate(model, test_loader)
model = train_model(model, DEVICE, train_loader, val_loader, num_epochs=NUM_EPOCHS)
evaluate(model, test_loader)
model.save('weights.pth')