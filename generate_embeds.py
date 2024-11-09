import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

from model import Model


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
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    # model = Model(pretrained=None).to(device)
    model = Model().to(device)
    # model.load('weights.pth')
    all_embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        query_imgs = torch.stack([model.processor(Image.open(query_image_path)) for query_image_path in df['query_image'][i:i+batch_size]]).to(device)
        query_texts = model.tokenizer(df['query_text'][i:i+batch_size]).to(device)
        with torch.no_grad():
            query_imgs_embedding = model(query_imgs)
            query_texts_embedding = model(query_texts)
        query_imgs_embedding /= query_imgs_embedding.norm(dim=-1, keepdim=True)
        query_texts_embedding /= query_texts_embedding.norm(dim=-1, keepdim=True)
        final_embedding = query_imgs_embedding + query_texts_embedding
        all_embeddings.append(final_embedding.detach().cpu().numpy())
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
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    # model = Model(pretrained=None).to(device)
    model = Model().to(device)
    # model.load('weights.pth')
    all_embeddings = []
    for i in tqdm(range(0, len(df), batch_size)):
        target_imgs = torch.stack([model.processor(Image.open(target_image_path)) for target_image_path in df['target_image'][i:i+batch_size]]).to(device)
        with torch.no_grad():
            target_imgs_embedding = model(target_imgs)
        target_imgs_embedding /= target_imgs_embedding.norm(dim=-1, keepdim=True)
        all_embeddings.append(target_imgs_embedding.detach().cpu().numpy())
    return np.concatenate(all_embeddings)