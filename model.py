import torch
import open_clip
from PIL import Image


class Model(torch.nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k') -> None:
        super().__init__()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.feature_extractor, _, self.processor = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        
        # Get CLIP embedding dimension
        self.embed_dim = self.feature_extractor.visual.output_dim
        
        # Additional projection layers
        self.query_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim * 2, self.embed_dim),
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        self.database_projection = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.LayerNorm(self.embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embed_dim, self.embed_dim)
        )
        
        self.set_param_trainable_mode(module=self.feature_extractor, status=False)


    def set_param_trainable_mode(self, module, status):
        for param in module.parameters():
            param.requires_grad = status
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))


    def get_image_embedding(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
            image = self.preprocess(image).unsqueeze(0)
        return self.feature_extractor.encode_image(image).to('cuda')
        
    def get_text_embedding(self, text):
        if isinstance(text, str):
            text = [text]
        text = self.tokenizer(text).to('cuda')
        return self.feature_extractor.encode_text(text)
        
    def forward(self, query_image, query_text):
        # Get base embeddings from CLIP
        image_features = self.get_image_embedding(query_image)
        text_features = self.get_text_embedding(query_text)
        
        # Concatenate image and text features
        combined_features = torch.cat([image_features, text_features], dim=1)
        
        # Project through learnable layers
        query_embedding = self.query_projection(combined_features)
        
        return torch.nn.functional.normalize(query_embedding, p=2, dim=1)
    
    def encode_database_image(self, image):
        image_features = self.get_image_embedding(image)
        database_embedding = self.database_projection(image_features)
        return torch.nn.functional.normalize(database_embedding, p=2, dim=1)