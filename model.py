import torch
import open_clip


class Model(torch.nn.Module):
    def __init__(self, model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', param_trainable_mode=True) -> None:
        super().__init__()
        self.feature_extractor, _, self.processor = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
        self.set_param_trainable_mode(param_trainable_mode)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        
    def forward(self, x):
        if x.ndim == 2:
            return self.feature_extractor.encode_text(x)
        return self.feature_extractor.encode_image(x)
    
    def set_param_trainable_mode(self, status):
        for param in self.feature_extractor.parameters():
            param.requires_grad = status
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
