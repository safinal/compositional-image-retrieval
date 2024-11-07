import numpy as np
import pandas as pd

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
    pass


def encode_database(df: pd.DataFrame) -> np.ndarray :
    """
    Process database images and generate embeddings.

    Args:
    df (pd. DataFrame ): DataFrame with column:
    - target_image: str, paths to database images

    Returns:
    np.ndarray: Embeddings array (num_images, embedding_dim)
    """
    pass