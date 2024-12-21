import numpy as np
from generate_embeds import encode_database, encode_queries
from sklearn.metrics.pairwise import cosine_similarity


def calculate_accuracy(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    assert predictions.shape == ground_truth.shape, "Predictions and ground truth must have the same shape."
    
    # Calculate the number of correct predictions
    correct_predictions = (predictions == ground_truth).sum()
    total_predictions = len(predictions)
    
    # Calculate accuracy as a percentage
    accuracy = correct_predictions / total_predictions
    return accuracy

def evaluate(model, dataset):
    query_embeddings = encode_queries(model, dataset.load_queries())
    database_embeddings = encode_database(model, dataset.load_database())
    similarities = cosine_similarity(query_embeddings, database_embeddings)
    predictions = np.argmax(similarities, axis=1)
    ground_truth = np.arange(len(database_embeddings))
    accuracy = calculate_accuracy(predictions, ground_truth)
    return accuracy