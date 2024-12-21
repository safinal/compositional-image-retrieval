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
    query_df = dataset.load_queries()
    database_df = dataset.load_database()
    database_lst = database_df['target_image'].to_list()
    ground_truth = np.array([database_lst.index(query_df.iloc[i]['target_image']) for i in range(len(query_df))])
    query_embeddings = encode_queries(model, query_df)
    database_embeddings = encode_database(model, database_df)
    similarities = cosine_similarity(query_embeddings, database_embeddings)
    predictions = np.argmax(similarities, axis=1)
    accuracy = calculate_accuracy(predictions, ground_truth)
    return accuracy
