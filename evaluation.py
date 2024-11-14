import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from generate_embeds import encode_database, encode_queries
from utils import calculate_accuracy


def evaluate(dataset):
    query_embeddings = encode_queries(dataset.load_queries())
    database_embeddings = encode_database(dataset.load_database())
    similarities = cosine_similarity(query_embeddings, database_embeddings)
    predictions = np.argmax(similarities, axis=1)
    ground_truth = np.arange(len(database_embeddings))
    accuracy = calculate_accuracy(predictions, ground_truth)
    return accuracy