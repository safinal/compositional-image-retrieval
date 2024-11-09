import os
import numpy as np
import torch
from torchvision.transforms import v2
import pandas as pd
from PIL import Image


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path: str, annotations_file_path: str, split: str, split_ratio: float, transform=None) -> None:
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.split = split
        self.split_ratio = split_ratio
        self.annotations = self.split_data(
            self.data_health_check(
                self.convert_image_names_to_path(
                    pd.read_csv(annotations_file_path)
                )
            )
        )
    
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> tuple:
        query_img_path = self.annotations.iloc[idx]['query_image']
        query_text = self.annotations.iloc[idx]['query_text']
        target_img_path = self.annotations.iloc[idx]['target_image']
        query_img = Image.open(query_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')
        # query_img = torchvision.io.read_image(path=query_img_path, mode=torchvision.io.image.ImageReadMode.RGB)
        # target_img = torchvision.io.read_image(path=target_img_path, mode=torchvision.io.image.ImageReadMode.RGB)
        if self.transform:
            query_img = self.transform(query_img)
            target_img = self.transform(target_img)
        
        return query_img, query_text, target_img
    
    def split_data(self, annotations):
        shuffled_df = annotations.sample(frac=1, random_state=42).reset_index(drop=True)
        if self.split == "test":
            return shuffled_df # sample test set
        if self.split == "train":
            return shuffled_df.iloc[:int(self.split_ratio * len(shuffled_df))] # train set
        if self.split == "validation":
            return shuffled_df.iloc[int(self.split_ratio * len(shuffled_df)):] # validation set
        raise Exception("split is not valid")

    def load_queries(self):
        return self.annotations.drop(columns=["target_image"])
    
    def load_database(self):
        return self.annotations[["target_image"]]
    
    def convert_image_names_to_path(self, df):
        df["query_image"] = self.img_dir_path + "/" + df["query_image"]
        df["target_image"] = self.img_dir_path + "/" + df["target_image"]
        return df
    
    def data_health_check(self, annotations):
        img_files = os.listdir(self.img_dir_path)
        broken_files = [img for img in img_files if self.is_truncated(os.path.join(self.img_dir_path, img))]
        annotations = annotations[
            ~annotations['target_image'].isin(broken_files) &
            ~annotations['query_image'].isin(broken_files)
        ]
        return annotations
    
    def is_truncated(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return False
        except (IOError, SyntaxError, Image.DecompressionBombError) as e:
            return True

def create_dataloader(
        img_dir_path: str,
        annotations_file_path: str,
        split: str,
        split_ratio: float,
        batch_size: int,
        num_workers: int,
    ):
    if split == 'train':
        transform = v2.Compose([
            v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = v2.Compose([
            v2.Resize(256),
            v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = RetrievalDataset(
        img_dir_path=img_dir_path,
        annotations_file_path=annotations_file_path,
        split=split,
        split_ratio=split_ratio,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return loader