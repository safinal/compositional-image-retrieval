import torch
import os


split_ratio = 1.0
image_root_dir = os.path.join(os.getcwd(), 'dataset', 'images')
annotations_file_path = os.path.join(os.getcwd(), 'dataset', 'data.csv')
test_root_dir = os.path.join(os.getcwd(), 'sample_evaluation', 'images')
test_annotations_file_path = os.path.join(os.getcwd(), 'sample_evaluation', 'data.csv')
batch_size = 256
num_workers = 128
num_epochs = 5
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
model_name = 'ViTamin-L-384'
pretrained_weights = 'datacomp1b'
learning_rate = 1e-4
weight_decay = 0.01
loss_temperature = 0.07
scheduler_t_0 = 5
scheduler_t_mult = 2
pretrained_token_classifier_path = 'trained_distil_bert_base'
