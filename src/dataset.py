import torch
from PIL import Image
from collections import defaultdict
import random
from torchvision.transforms import v2
import pandas as pd

from src.config import ConfigManager


class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir_path: str, annotations_file_path: str, split: str, transform=None, tokenizer=None) -> None:
        self.img_dir_path = img_dir_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.split = split
        self.annotations = self.split_data(
            self.convert_image_names_to_path(
                pd.read_csv(annotations_file_path)
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
        if self.tokenizer:
            query_text = self.tokenizer(query_text).squeeze(0)
        return query_img, query_text, target_img, self.annotations.iloc[idx]['query_text']
    
    def split_data(self, annotations):
        split_ratio = ConfigManager().get("training")["split_ratio"]
        shuffled_df = annotations.sample(frac=1, random_state=42).reset_index(drop=True)
        if self.split == "test":
            return shuffled_df # sample test set
        if self.split == "train":
            return shuffled_df.iloc[:int(split_ratio * len(shuffled_df))] # train set
        if self.split == "validation":
            return shuffled_df.iloc[int(split_ratio * len(shuffled_df)):] # validation set
        raise Exception("split is not valid")

    def load_queries(self):
        return self.annotations.drop(columns=["target_image"])
    
    def load_database(self):
        return self.annotations[["target_image"]]
    
    def convert_image_names_to_path(self, df):
        df["query_image"] = self.img_dir_path + "/" + df["query_image"]
        df["target_image"] = self.img_dir_path + "/" + df["target_image"]
        return df


class UniqueTargetImageBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        """
        Initializes the sampler.

        Args:
            dataset (RetrievalDataset): The dataset to sample from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the data every epoch.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Create a mapping from target_image to list of indices
        self.target_to_indices = defaultdict(list)
        for idx in range(len(self.dataset)):
            target_image = self.dataset.annotations.iloc[idx]['target_image']
            self.target_to_indices[target_image].append(idx)
        
        # List of unique target_images
        self.unique_target_images = list(self.target_to_indices.keys())
        if self.shuffle:
            random.shuffle(self.unique_target_images)
            for indices in self.target_to_indices.values():
                random.shuffle(indices)

    def __iter__(self):
        """
        Yields lists of indices where each list represents a batch with unique target_images.
        """
        # Create a copy of indices per target_image to preserve original order
        queues = [indices.copy() for indices in self.target_to_indices.values()]
        
        if self.shuffle:
            random.shuffle(queues)
        
        batch = []
        while any(queues):
            for queue in queues:
                if queue:
                    batch.append(queue.pop())
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
            # Optional: Shuffle queues after each full pass to ensure randomness
            if self.shuffle:
                random.shuffle(queues)
        
        if batch:
            yield batch

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        total = len(self.dataset)
        return (total + self.batch_size - 1) // self.batch_size


default_train_transform = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
default_test_transform = v2.Compose([
    v2.Resize(256),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def create_train_val_test_datasets_and_loaders(tokenizer, transform=None):
    img_dir_path = ConfigManager().get("paths")["image_root_dir"]
    annotations_file_path = ConfigManager().get("paths")["annotations_file_path"]
    split_ratio = ConfigManager().get("training")["split_ratio"]
    batch_size = ConfigManager().get("training")["batch_size"]
    num_workers = ConfigManager().get("training")["num_workers"]

    train_dataset = RetrievalDataset(
        img_dir_path=img_dir_path,
        annotations_file_path=annotations_file_path,
        split='train',
        transform=transform if transform is not None else default_train_transform,
        tokenizer=tokenizer
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=num_workers,
        batch_sampler=UniqueTargetImageBatchSampler(dataset=train_dataset, batch_size=batch_size)
    )
    val_dataset, val_loader = None, None
    if split_ratio < 1:
        val_dataset = RetrievalDataset(
            img_dir_path=img_dir_path,
            annotations_file_path=annotations_file_path,
            split='validation',
            transform=transform if transform is not None else default_test_transform,
            tokenizer=tokenizer
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
    test_dataset = RetrievalDataset(
        img_dir_path=ConfigManager().get("path")["test_root_dir"],
        annotations_file_path=ConfigManager().get("path")["test_annotations_file_path"],
        split='test',
        transform=transform if transform is not None else default_test_transform,
        tokenizer=tokenizer
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    return train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader
