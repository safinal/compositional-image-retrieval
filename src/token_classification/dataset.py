import pandas as pd
import numpy as np
import torch
import re
from transformers import DataCollatorForTokenClassification
from sklearn.preprocessing import LabelEncoder

from src.config import ConfigManager


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

def get_objects(objects_path):
    objects = set()
    with open(objects_path) as f:
        for line in f:
            objects.add(line.strip().lower())
    return list(objects)

def create_data(prompt_templates_df, all_objects, num_data_per_prompt_template):
    data = []
    for template in prompt_templates_df['template'].to_list():
        template1 = ' '.join(template.lower().split())
        pos_count = len(re.findall(r'<positive_object>', template1))
        neg_count = len(re.findall(r'<negative_object>', template1))    
        for j in range(num_data_per_prompt_template):
            objects = np.random.choice(all_objects, size=pos_count + neg_count, replace=False)
            i = 0
            # template = template1.split()
            template = split_sentence(template1)
            tags = []
            for _ in range(len(template)):
                if template[_] == "<positive_object>":
                    tags.append("<positive_object>")
                    template[_] = objects[i]
                    i += 1
                elif template[_] == "<negative_object>":
                    tags.append("<negative_object>")
                    template[_] = objects[i]
                    i += 1
                else:
                    tags.append("other")
            assert i == pos_count + neg_count
            assert len(tags) == len(template)
            assert "<positive_object>" not in template and "<negative_object>" not in template
            data.append({'tokens': list(map(str, template)), 'labels': tags})
    return data

def tokenize_and_align_labels(samples, tokenizer, label2id):
    tokenized_inputs = tokenizer(
        [sample["tokens"] for sample in samples],
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True
    )

    labels = []
    for i, sample in enumerate(samples):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[sample["labels"][word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


def create_dataset(tokenizer):
    prompt_templates_df = pd.read_json(ConfigManager().get("data")["prompt_templates_path"])
    all_objects = get_objects(ConfigManager().get("data")["objects_path"])
    data = create_data(prompt_templates_df, all_objects, ConfigManager().get("data")["num_data_per_prompt_template"])

    # Label encoding
    labels = [label for sample in data for label in sample["labels"]]
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
    id2label = {idx: label for label, idx in label2id.items()}

    tokenized_data = tokenize_and_align_labels(data, tokenizer, label2id)
    dataset = TokenClassificationDataset(tokenized_data)
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    return dataset, id2label, label2id, data_collator