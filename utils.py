import numpy as np
import re
import spacy


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


def parse_actions(text):
    # Load English language model
    nlp = spacy.load("en_core_web_sm")
    
    # List of common action verbs and their normalized forms
    action_verbs = {
        'add': 'add',
        'insert': 'add',
        'introduce': 'add',
        'bring': 'add',
        'place': 'add',
        'include': 'add',
        'remove': 'remove',
        'eliminate': 'remove',
        'discard': 'remove',
        'take': 'remove',
        'get rid': 'remove'
    }
    
    # Process the text
    doc = nlp(text.lower())
    
    # Initialize result structure
    result = [
        {"verb": [], "nouns": []},
        {"verb": [], "nouns": []}
    ]
    
    # Split text into two parts if possible
    parts = re.split(r'\s+(?:and|,|then|\.|\s+)+\s*', text.lower())
    
    current_action = 0
    current_verb = None
    
    for token in doc:
        # Check for verbs
        lemma = token.lemma_.lower()
        if any(verb in lemma for verb in action_verbs.keys()):
            # Handle multi-word verbs
            verb_phrase = lemma
            if token.i + 1 < len(doc) and doc[token.i + 1].text in ['in', 'away', 'out']:
                verb_phrase += ' ' + doc[token.i + 1].text
            
            normalized_verb = None
            for verb, norm in action_verbs.items():
                if verb in verb_phrase:
                    normalized_verb = norm
                    break
            
            if normalized_verb:
                if current_verb != normalized_verb:
                    current_verb = normalized_verb
                    if current_action < 2:
                        result[current_action]["verb"] = [normalized_verb]
                        current_action += 1
        
        # Collect nouns
        elif token.pos_ == "NOUN":
            if current_action > 0 and len(result[current_action-1]["verb"]) > 0:
                if token.text not in result[current_action-1]["nouns"]:
                    result[current_action-1]["nouns"].append(token.text)
    
    # Clean up empty actions and ensure proper structure
    for action in result:
        if not action["verb"]:
            action["verb"] = ["add" if not result[0]["verb"] else "remove"]
    
    return result