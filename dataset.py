import os

import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image


SPLIT_RATIO = 0.8
IMAGE_ROOT_DIR = os.path.join('dataset', 'images')
ANNOTATIONS_FILE_PATH = os.path.join('dataset', 'data.csv')


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path: str, annotations_file_path: str, split: str, transform=None) -> None:
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.split = split
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
        if self.split == "test":
            return shuffled_df # sample test set
        elif self.split == "train":
            return shuffled_df.iloc[:int(SPLIT_RATIO * len(shuffled_df))] # train set
        return shuffled_df.iloc[int(SPLIT_RATIO * len(shuffled_df)):] # validation set

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

