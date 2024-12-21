import pandas as pd
import numpy as np
import torch
import re
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast, DistilBertForTokenClassification, Trainer, TrainingArguments,
    DataCollatorForTokenClassification
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


df = pd.read_json("./templates.json")

train_objects = set()
for text in pd.read_csv("dataset/data.csv")['query_text'].to_list():
    temp = text.split()
    assert temp[0] == 'add' and temp[2] == 'and' and temp[3] == 'remove' and len(temp) == 5
    train_objects.add(temp[1])
    train_objects.add(temp[4])
train_objects = list(train_objects)


sample_evaluation_objects = set()
for text in pd.read_csv("sample_evaluation/data.csv")['query_text'].to_list():
    temp = text.split()
    if temp[0] == 'add' and temp[2] == 'and' and temp[3] == 'remove' and len(temp) == 5:
        sample_evaluation_objects.add(temp[1])
        sample_evaluation_objects.add(temp[4])
sample_evaluation_objects = list(set(list(sample_evaluation_objects) + ['bench', 'boat', 'banana', 'motorcycle', 'pizza', 'horse', 'pizza', 'bicycle', 'cow', 'bench', 'bird', 'apple', 'airplane', 'pizza', 'boat', 'plate', 'bicycle', 'handbag', 'bear', 'zebra', 'motorcycle', 'fork', 'shoe', 'carrot', 'knife', 'cow', 'cow', 'donut', 'orange', 'shoe', 'bowl', 'zebra', 'backpack', 'handbag', 'train', 'surfboard', 'chair', 'car', 'train', 'snowboard', 'bench', 'dog', 'axe', 'hat', 'tie', 'person', 'person', 'bird', 'cow', 'tie', 'bird', 'orange', 'pizza', 'handbag', 'truck', 'carrot', 'couch', 'bear', 'skateboard', 'motorcycle', 'couch', 'bench', 'car', 'bus', 'bicycle', 'cup', 'sheep', 'umberella', 'bear', 'donut', 'broccoli', 'sandwich', 'airplane', 'donut', 'person', 'plate', 'horse', 'truck', 'bicycle', 'apple', 'fork', 'boat', 'bus', 'suitcase', 'umberella', 'zebra', 'person', 'boat', 'hat', 'apple', 'fork', 'shoe', 'cake', 'bench', 'bear', 'car', 'cup', 'elephant', 'truck', 'bear', 'elephant', 'umberella', 'broccoli', 'shoe', 'skateboard', 'truck', 'banana', 'spoon', 'bowl', 'bench', 'shoe', 'elephant', 'suitcase', 'banana', 'person', 'skateboard', 'pizza', 'bear', 'horse', 'train']))


simco = ['Cube', 'Sphere', 'Cylinder', 'Mug', 'Pentagon', 'Heart', 'Cone', 'Pyramid', 'Diamond', 'Moon', 'Cross', 'Snowflake', 'Leaf', 'Arrow', 'Star', 'Torus', 'Pot']
comco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'dining table', 'cell phone', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'kite', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog', 'potted plant', 'teddy bear', 'hair drier', 'hair brush', 'skateboard', 'surfboard', 'bottle', 'plate', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake', 'chair', 'couch', 'bed', 'mirror', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'toothbrush']
small_objects = ['ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'banana', 'bandage', 'basket', 'bat', 'bee', 'belt', 'binoculars', 'bird', 'blackberry', 'blueberry', 'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'broccoli', 'broom', 'bucket', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camera', 'candle', 'carrot', 'cat', 'clarinet', 'clock', 'compass', 'cookie', 'crab', 'backpack', 'crown', 'cup', 'dog', 'donut', 'drill', 'duck', 'dumbbell', 'ear', 'envelope', 'eraser', 'eye', 'eyeglasses', 'feather', 'finger', 'fork', 'frog', 'hammer', 'hat', 'headphones', 'hedgehog', 'helmet', 'hourglass', 'jacket', 'keyboard', 'key', 'knife', 'lantern', 'laptop', 'leaf', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'marker', 'megaphone', 'microphone', 'microwave', 'mosquito', 'mouse', 'mug', 'mushroom', 'necklace', 'onion', 'owl', 'paintbrush', 'parrot', 'peanut', 'pear', 'peas', 'pencil', 'pillow', 'pineapple', 'pizza', 'pliers', 'popsicle', 'postcard', 'potato', 'purse', 'rabbit', 'raccoon', 'radio', 'rake', 'rhinoceros', 'rifle', 'sandwich', 'saw', 'saxophone', 'scissors', 'scorpion', 'shoe', 'shovel', 'skateboard', 'skull', 'snail', 'snake', 'snorkel', 'spider', 'spoon', 'squirrel', 'stethoscope', 'strawberry', 'swan', 'sword', 'syringe', 'teapot', 'telephone', 'toaster', 'toothbrush', 'trombone', 'trumpet', 'umbrella', 'violin', 'watermelon', 'wheel']
medium_objects = ['angel', 'bathtub', 'bear', 'bed', 'bench', 'bicycle', 'camel', 'cannon', 'canoe', 'cello', 'chair', 'chandelier', 'computer', 'cooler', 'couch', 'cow', 'crocodile', 'dishwasher', 'dolphin', 'door', 'dresser', 'drums', 'flamingo', 'guitar', 'horse', 'kangaroo', 'ladder', 'mermaid', 'motorbike', 'panda', 'penguin', 'piano', 'pig', 'sheep', 'stereo', 'stove', 'table', 'television', 'tiger', 'zebra']
large_objects = [
    "aircraft carrier", "airplane", "ambulance", "barn", "bridge",
    "bulldozer", "bus", "car", "castle", "church",
    "cloud", "cruise ship", "dragon", "elephant", "firetruck",
    "flying saucer", "giraffe", "helicopter", "hospital", "hot air balloon",
    "house", "moon", "mountain", "palm tree", "parachute",
    "pickup truck", "police car", "sailboat", "school bus", "skyscraper",
    "speedboat", "submarine", "sun", "tent", "The Eiffel Tower",
    "The Great Wall of China", "tractor", "train", "tree", "truck",
    "van", "whale", "windmill"
]
extra_gpt_objects = [    'lamp', 'desk', 'mirror', 'window', 'door', 'book', 'pen', 'pencil', 'phone',
    'laptop', 'keyboard', 'mouse', 'monitor', 'printer', 'camera', 'watch',
    'glasses', 'headphones', 'speaker', 'microphone', 'guitar', 'piano', 'drum',
    'painting', 'poster', 'clock', 'vase', 'plant', 'tree', 'flower', 'grass',
    'rock', 'mountain', 'cloud', 'sun', 'moon', 'star', 'building', 'house',
    'bridge', 'road', 'traffic light', 'street sign', 'fence', 'wall', 'ceiling',
    'floor', 'carpet', 'pillow', 'blanket', 'towel', 'soap', 'toothbrush',
    'scissors', 'key', 'wallet', 'ring', 'necklace', 'bracelet', 'shirt',
    'pants', 'dress', 'jacket', 'sock', 'glove', 'scarf', 'belt', 'table',
    'refrigerator', 'oven', 'microwave', 'sink', 'toilet', 'shower', 'bathtub',
    'television', 'remote control', 'fan', 'air conditioner', 'heater']

extended_words = list(set(list(map(lambda x: x.lower(), train_objects + sample_evaluation_objects + simco + comco + small_objects + medium_objects + large_objects + extra_gpt_objects))))

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


data = []
for template in df['template'].to_list()[:]:
    template1 = ' '.join(template.lower().split())
    pos_count = len(re.findall(r'<positive_object>', template1))
    neg_count = len(re.findall(r'<negative_object>', template1))    
    for j in range(15):
        objects = np.random.choice(extended_words, size=pos_count + neg_count, replace=False)
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


# Label encoding
labels = [label for sample in data for label in sample["labels"]]
label_encoder = LabelEncoder()
label_encoder.fit(labels)
label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}
id2label = {idx: label for label, idx in label2id.items()}

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize_and_align_labels(samples):
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

tokenized_data = tokenize_and_align_labels(data)

# Dataset class
class TokenClassificationDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings["input_ids"])
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item

dataset = TokenClassificationDataset(tokenized_data)

# Model
model = DistilBertForTokenClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./trained_distil_bert_base_results',
    num_train_epochs=20,
    per_device_train_batch_size=1024,
    per_device_eval_batch_size=1024,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./trained_distil_bert_base_logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2
)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Compute metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred, label in zip(predictions, labels):
        temp_pred = []
        temp_lab = []
        for p, l in zip(pred, label):
            if l != -100:
                temp_pred.append(id2label[p])
                temp_lab.append(id2label[l])
        true_labels.extend(temp_lab)
        true_predictions.extend(temp_pred)
    
    report = classification_report(true_labels, true_predictions, zero_division=0)
    print(report)
    return {}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("./trained_distil_bert_base")
tokenizer.save_pretrained("./trained_distil_bert_base")

# Evaluate
trainer.evaluate()
