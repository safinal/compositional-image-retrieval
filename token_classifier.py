import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification


# Define label mappings (ensure this matches the mappings used during training)
label2id = {'<negative_object>': 0, 'other': 2, '<positive_object>': 1}
id2label = {v: k for k, v in label2id.items()}

def prepare_input(tokens, tokenizer, max_length=128):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    )
    return encoding


def split_sentence(sentence):
    # List of special tokens to preserve
    special_tokens = ['<positive_object>', '<negative_object>']
    
    # More comprehensive list of punctuation marks and symbols
    punctuation = ',.?!;:()[]{}""\'`@#$%^&*+=|\\/<>-—–'
    
    # Initialize result list and temporary word
    result = []
    current_word = ''
    i = 0
    
    while i < len(sentence):
        # Check for special tokens
        found_special = False
        for token in special_tokens:
            if sentence[i:].startswith(token):
                # Add previous word if exists
                if current_word:
                    result.append(current_word)
                    current_word = ''
                # Add special token
                result.append(token)
                i += len(token)
                found_special = True
                break
        
        if found_special:
            continue
            
        # Handle punctuation
        if sentence[i] in punctuation:
            # Add previous word if exists
            if current_word:
                result.append(current_word)
                current_word = ''
            # Add punctuation as separate token
            result.append(sentence[i])
            
        # Handle spaces
        elif sentence[i].isspace():
            if current_word:
                result.append(current_word)
                current_word = ''
                
        # Build regular words
        else:
            current_word += sentence[i]
            
        i += 1
    
    # Add final word if exists
    if current_word:
        result.append(current_word)
        
    return result

def predict(tokens, model, tokenizer, device, max_length=128):
    tokens = split_sentence(' '.join(tokens.lower().split()))

    # Prepare the input
    encoding = prepare_input(tokens, tokenizer, max_length=max_length)
    word_ids = encoding.word_ids(batch_index=0)  # List of word IDs
    
    # Move tensors to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    
    # Decode tokens and labels
    tokens_decoded = tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy()[0])
    labels = [id2label.get(pred, 'O') for pred in predictions]
    
    # Align tokens with original word-level tokens
    aligned_predictions = []
    previous_word_idx = None
    for token, label, word_idx in zip(tokens_decoded, labels, word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            aligned_predictions.append((tokens[word_idx], label))
            previous_word_idx = word_idx
    return aligned_predictions


def load_token_classifier(pretrained_token_classifier_path, device):
    # Load tokenizer and model
    tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_token_classifier_path)
    token_classifier = DistilBertForTokenClassification.from_pretrained(pretrained_token_classifier_path)
    token_classifier.to(device)
    return token_classifier, tokenizer