import torch
import torchvision
import pandas as pd
import os
import open_clip
from PIL import Image

from model import Model


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path: str = os.path.join('dataset', 'images'), annotations_file_path: str = os.path.join('dataset', 'data.csv'), transform=None) -> None:
        self.img_dir_path = img_dir_path
        self.annotations = pd.read_csv(annotations_file_path)
        self.transform = transform
    
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


model = Model()


transform = torchvision.transforms.v2.Compose([
    torchvision.transforms.v2.ToDtype(torch.float32, scale=True),
    torchvision.transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = RetrievalDataset(transform=model.processor if hasattr(model, 'processor') else transform)
train_dataset, val_dataset = torch.utils.data.random_split(dataset=dataset, lengths=[0.8, 0.2])