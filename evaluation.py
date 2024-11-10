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

# def evaluate(model, dataset):
#     model.eval()
#     # 1. Load val queries and database
#     query_df = dataset.load_queries()
#     database_df = dataset.load_database()

#     # 2. Generate embeddings
#     query_embeddings = encode_queries(model, query_df)
#     database_embeddings = encode_database(model, database_df)

#     # 3. Calculate cosine similarity
#     similarities = cosine_similarity(query_embeddings, database_embeddings)

#     # 4. Get top-1 predictions
#     predictions = np.argmax(similarities, axis=1)

#     # 5.Calculateaccuracy
#     ground_truth = np.arange(len(database_embeddings))
#     accuracy = calculate_accuracy(predictions, ground_truth)
#     print(f"Accuracy: {100*accuracy}")

def evaluate(model, loader):
    model.eval()
    # 2. Generate embeddings
    query_embeddings = encode_queries(model, loader)
    database_embeddings = encode_database(model, loader)

    # 3. Calculate cosine similarity
    similarities = cosine_similarity(query_embeddings, database_embeddings)

    # 4. Get top-1 predictions
    predictions = np.argmax(similarities, axis=1)

    # 5.Calculateaccuracy
    ground_truth = np.arange(len(database_embeddings))
    accuracy = calculate_accuracy(predictions, ground_truth)
    print(f"Accuracy: {100*accuracy}")