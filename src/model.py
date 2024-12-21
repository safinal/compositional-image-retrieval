import torch
import open_clip


class Model(torch.nn.Module):
    def __init__(self, model_name, pretrained) -> None:
        super().__init__()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.feature_extractor, _, self.processor = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained
        )
        self.set_param_trainable_mode(module=self.feature_extractor, status=False)

    def set_param_trainable_mode(self, module, status):
        for param in module.parameters():
            param.requires_grad = status
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))