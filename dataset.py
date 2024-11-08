import torch
import torchvision
import pandas as pd
import os
from PIL import Image

from model import Model


SPLIT_RATIO = 0.8
IMAGE_ROOT_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_FILE_PATH = os.path.join('dataset', 'data.csv')

class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path: str = IMAGE_ROOT_DIR, annotations_file_path: str = ANNOTATIONS_FILE_PATH, transform=None, split=None) -> None:
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.split = split
        self.annotations = self.split_data(pd.read_csv(annotations_file_path).drop(columns=["Unnamed: 0"]))
    
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple:
        query_img_path = os.path.join(self.img_dir_path, self.annotations.iloc[idx]['query_image'])
        query_text = self.annotations.iloc[idx]['query_text']
        target_img_path = os.path.join(self.img_dir_path, self.annotations.iloc[idx]['target_image'])
        query_img = Image.open(query_img_path)
        target_img = Image.open(target_img_path)
        # query_img = torchvision.io.read_image(path=query_img_path, mode=torchvision.io.image.ImageReadMode.RGB)
        # target_img = torchvision.io.read_image(path=target_img_path, mode=torchvision.io.image.ImageReadMode.RGB)
        if self.transform:
            query_img = self.transform(query_img)
            target_img = self.transform(target_img)
        return query_img, query_text, target_img
    
    def split_data(self, annotations):
        shuffled_df = annotations.sample(frac=1, random_state=42).reset_index(drop=True)
        if not self.split:
            return shuffled_df # sample test set
        elif self.split == "train":
            return shuffled_df.iloc[:int(SPLIT_RATIO * len(shuffled_df))] # train set
        return shuffled_df.iloc[int(SPLIT_RATIO * len(shuffled_df)):] # validation set

    def load_queries(self):
        return self.annotations.drop(columns=["target_image"])
    
    def load_database(self):
        return self.annotations[["target_image"]]


model = Model()


transform = torchvision.transforms.v2.Compose([
    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
    torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = RetrievalDataset(transform=model.processor if hasattr(model, 'processor') else transform, split='train')
val_dataset = RetrievalDataset(transform=model.processor if hasattr(model, 'processor') else transform, split='validation')
