import numpy as np

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def calculate_accuracy(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions against the ground truth.

    Args:
    predictions (np.ndarray): Predicted indices of target images, shape (num_queries,)
    ground_truth (np.ndarray): True indices of target images, shape (num_queries,)

    Returns:
    float: Accuracy score, representing the percentage of correct predictions.
    """
    assert predictions.shape == ground_truth.shape, "Predictions and ground truth must have the same shape."
    
    # Calculate the number of correct predictions
    correct_predictions = (predictions == ground_truth).sum()
    total_predictions = len(predictions)
    
    # Calculate accuracy as a percentage
    accuracy = correct_predictions / total_predictions
    return accuracy
