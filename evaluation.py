import numpy as np

import torch
import torchvision
from sklearn.metrics.pairwise import cosine_similarity

from model import Model
from generate_embeds import encode_database, encode_queries
from dataset import RetrievalDataset


model = Model()

transform = torchvision.transforms.v2.Compose([
    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
    torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = RetrievalDataset(transform=model.processor if hasattr(model, 'processor') else transform, split='train')
val_dataset = RetrievalDataset(transform=model.processor if hasattr(model, 'processor') else transform, split='validation')
test_dataset = RetrievalDataset(transform=model.processor if hasattr(model, 'processor') else transform, split=None)

# 1. Load val queries and database
query_df = val_dataset.load_queries()
database_df = val_dataset.load_database()

# 2. Generate embeddings
query_embeddings = encode_queries(query_df)
database_embeddings = encode_database(database_df)

# 3. Calculate cosine similarity
similarities = cosine_similarity(query_embeddings, database_embeddings, dim=0)

# 4. Get top-1 predictions
predictions = np.argmax(similarities, axis=1)
