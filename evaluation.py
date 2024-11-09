import os
import numpy as np

import torch
import torchvision
from torchvision.transforms import v2
from sklearn.metrics.pairwise import cosine_similarity

from model import Model
from generate_embeds import encode_database, encode_queries
from dataset import RetrievalDataset
from utils import calculate_accuracy

SPLIT_RATIO = 0.8
IMAGE_ROOT_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_FILE_PATH = os.path.join('dataset', 'data.csv')
TEST_ROOT_DIR = os.path.join('sample_evaluation', 'images')
TEST_ANNOTATIONS_FILE_PATH = os.path.join('sample_evaluation', 'data.csv')

batch_size = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# model = Model(pretrained=None).to(device)
model = Model().to(device)
# model.load('weights.pth')

transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = RetrievalDataset(
    img_dir_path=IMAGE_ROOT_DIR,
    annotations_file_path=ANNOTATIONS_FILE_PATH,
    split="train",
    transform=model.processor if hasattr(model, 'processor') else transform
)
val_dataset = RetrievalDataset(
    img_dir_path=IMAGE_ROOT_DIR,
    annotations_file_path=ANNOTATIONS_FILE_PATH,
    split="validation",
    transform=model.processor if hasattr(model, 'processor') else transform
)
test_dataset = RetrievalDataset(
    img_dir_path=TEST_ROOT_DIR,
    annotations_file_path=TEST_ANNOTATIONS_FILE_PATH,
    split="test",
    transform=model.processor if hasattr(model, 'processor') else transform
)

# 1. Load val queries and database
query_df = test_dataset.load_queries()
database_df = test_dataset.load_database()

# 2. Generate embeddings
query_embeddings = encode_queries(query_df)
database_embeddings = encode_database(database_df)

# 3. Calculate cosine similarity
similarities = cosine_similarity(query_embeddings, database_embeddings)

# 4. Get top-1 predictions
predictions = np.argmax(similarities, axis=1)

# 5.Calculateaccuracy
ground_truth = np.arange(len(database_embeddings))
accuracy = calculate_accuracy(predictions, ground_truth)
print(f"Accuracy: {100*accuracy}")

model.save('weights.pth')