from PIL import Image
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from src.config import ConfigManager
from src.token_classifier import load_token_classifier, predict


def encode_queries(model, df: pd.DataFrame) -> np.ndarray:
    """
    Process query pairs and generate embeddings.

    Args:
    df (pd. DataFrame ): DataFrame with columns:
    - query_image: str, paths to query images
    - query_text: str, text descriptions

    Returns:
    np.ndarray: Embeddings array (num_queries, embedding_dim)
    """
    token_classifier, token_classifier_tokenizer = load_token_classifier(
        ConfigManager().get("paths")["pretrained_token_classifier_path"],
        ConfigManager().get("training")["device"]
    )
    model.eval()
    all_embeddings = []
    batch_size = ConfigManager().get("training")["batch_size"]
    device = ConfigManager().get("training")["device"]
    for i in tqdm(range(0, len(df), batch_size)):
        query_imgs = torch.stack([model.processor(Image.open(query_image_path)) for query_image_path in df['query_image'][i:i+batch_size]]).to(device)
        with torch.no_grad():
            query_imgs_embd = model.feature_extractor.encode_image(query_imgs)
        for j, text in enumerate(df['query_text'][i:i+batch_size].to_list()):
            predictions = predict(
                tokens=text,
                model=token_classifier,
                tokenizer=token_classifier_tokenizer,
                device=device,
                max_length=128
            )
            neg = []
            pos = []
            last_tag = ''
            for token, label in predictions:
                if label == '<positive_object>':
                    if last_tag != '<positive_object>':
                        pos.append(f"a photo of a {token}.")
                    else:
                        pos[-1] = pos[-1][:-1] + f" {token}."
                elif label == '<negative_object>':
                    if last_tag != '<negative_object>':
                        neg.append(f"a photo of a {token}.")
                    else:
                        neg[-1] = neg[-1][:-1] + f" {token}."
                last_tag = label
            for obj in pos:
                with torch.no_grad():
                    query_imgs_embd[j] += model.feature_extractor.encode_text(model.tokenizer(obj).to(device))[0]
            for obj in neg:
                with torch.no_grad():
                    query_imgs_embd[j] -= model.feature_extractor.encode_text(model.tokenizer(obj).to(device))[0]
        query_imgs_embd = torch.nn.functional.normalize(query_imgs_embd, dim=1, p=2)
        all_embeddings.append(query_imgs_embd.detach().cpu().numpy())
    return np.concatenate(all_embeddings)


def encode_database(model, df: pd.DataFrame) -> np.ndarray :
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
    batch_size = ConfigManager().get("training")["batch_size"]
    device = ConfigManager().get("training")["device"]
    for i in tqdm(range(0, len(df), batch_size)):
        target_imgs = torch.stack([model.processor(Image.open(target_image_path)) for target_image_path in df['target_image'][i:i+batch_size]]).to(device)
        with torch.no_grad():
            # target_imgs_embedding = model.encode_database_image(target_imgs)
            target_imgs_embedding = model.feature_extractor.encode_image(target_imgs)
        target_imgs_embedding = torch.nn.functional.normalize(target_imgs_embedding, dim=1, p=2)
        all_embeddings.append(target_imgs_embedding.detach().cpu().numpy())
    return np.concatenate(all_embeddings)