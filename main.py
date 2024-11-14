import os
import torch
from torchvision.transforms import v2

from train import train_epoch, InfoNCELoss
from model import Model
from evaluation import evaluate
from dataset import RetrievalDataset, UniqueTargetImageBatchSampler


SPLIT_RATIO = 0.95
IMAGE_ROOT_DIR = os.path.join(os.getcwd(), 'dataset', 'images')
ANNOTATIONS_FILE_PATH = os.path.join(os.getcwd(), 'dataset', 'data.csv')
TEST_ROOT_DIR = os.path.join(os.getcwd(), 'sample_evaluation', 'images')
TEST_ANNOTATIONS_FILE_PATH = os.path.join(os.getcwd(), 'sample_evaluation', 'data.csv')
BATCH_SIZE = 80
NUM_WORKERS = 128
NUM_EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
MODEL_NAME = 'ViT-B-32'
PRETRAINED_WEIGHTS = 'laion2b_s34b_b79k'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
LOSS_TEMPERATURE = 0.07
SCHEDULER_T_0 = 5
SCHEDULER_T_MULT = 2


model = Model(model_name=MODEL_NAME, pretrained=PRETRAINED_WEIGHTS).to(DEVICE)
criterion = InfoNCELoss(temperature=LOSS_TEMPERATURE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=SCHEDULER_T_0, T_mult=SCHEDULER_T_MULT)


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


train_dataset = RetrievalDataset(
    img_dir_path=IMAGE_ROOT_DIR,
    annotations_file_path=ANNOTATIONS_FILE_PATH,
    split='train',
    transform=model.processor if hasattr(model, 'processor') else test_transform,
    tokenizer=model.tokenizer
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    # batch_size=BATCH_SIZE, 
    # shuffle=True,
    num_workers=NUM_WORKERS,
    batch_sampler=UniqueTargetImageBatchSampler(dataset=train_dataset, batch_size=BATCH_SIZE)
)

val_dataset = RetrievalDataset(
    img_dir_path=IMAGE_ROOT_DIR,
    annotations_file_path=ANNOTATIONS_FILE_PATH,
    split='validation',
    transform=model.processor if hasattr(model, 'processor') else test_transform,
    tokenizer=model.tokenizer
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

test_dataset = RetrievalDataset(
    img_dir_path=TEST_ROOT_DIR,
    annotations_file_path=TEST_ANNOTATIONS_FILE_PATH,
    split='test',
    transform=model.processor if hasattr(model, 'processor') else test_transform,
    tokenizer=model.tokenizer
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)

print(f"Validation Accuracy: {100*evaluate(val_dataset)}")
print(f"Test Accuracy: {100*evaluate(test_dataset)}")

best_val_acc = 0
model.set_param_trainable_mode(model.feature_extractor, status=True)
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    print(f"Test Accuracy: {100*evaluate(test_dataset)}")
    scheduler.step()

print(f"Validation Accuracy: {100*evaluate(val_dataset)}")
print(f"Test Accuracy: {100*evaluate(test_dataset)}")

# model.save("/home/nafisi/temp/rayan-phase2-q1/weights.pth")
# api = HfApi()
# api.upload_file(
#     path_or_fileobj="/home/nafisi/temp/rayan-phase2-q1/weights.pth",
#     path_in_repo="weights.pth",
#     repo_id="safinal/rayan-phase2-q1",
#     repo_type="model",
# )