import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

from model import Model

BATCH_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = Model(pretrained=None).to(DEVICE)
model.load('weights.pth')


def encode_queries(df: pd.DataFrame) -> np.ndarray:
    """
    Process query pairs and generate embeddings.

    Args:
    df (pd. DataFrame ): DataFrame with columns:
    - query_image: str, paths to query images
    - query_text: str, text descriptions

    Returns:
    np.ndarray: Embeddings array (num_queries, embedding_dim)
    """
    model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        query_imgs = torch.stack([model.processor(Image.open(query_image_path)) for query_image_path in df['query_image'][i:i+BATCH_SIZE]]).to(DEVICE)
        query_texts = model.tokenizer(df['query_text'][i:i+BATCH_SIZE]).to(DEVICE)
        with torch.no_grad():
            query_embedding = model(query_imgs, query_texts)
        query_embedding = torch.nn.functional.normalize(query_embedding, dim=1, p=2)
        all_embeddings.append(query_embedding.detach().cpu().numpy())
    return np.concatenate(all_embeddings)


def encode_database(df: pd.DataFrame) -> np.ndarray :
    """
    Process database images and generate embeddings.

    Args:
    df (pd. DataFrame ): DataFrame with column:
    - target_image: str, paths to database images

    Returns:
    np.ndarray: Embeddings array (num_images, embedding_dim)
    """
    model.eval()
    all_embeddings = []
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        target_imgs = torch.stack([model.processor(Image.open(target_image_path)) for target_image_path in df['target_image'][i:i+BATCH_SIZE]]).to(DEVICE)
        with torch.no_grad():
            target_imgs_embedding = model.encode_database_image(target_imgs)
        target_imgs_embedding = torch.nn.functional.normalize(target_imgs_embedding, dim=1, p=2)
        all_embeddings.append(target_imgs_embedding.detach().cpu().numpy())
    return np.concatenate(all_embeddings)


