import torch
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification

from src.config import ConfigManager
from src.retrieval_model.model import Model
from src.retrieval_model.loss import InfoNCELoss
from src.retrieval_model.dataset import create_train_val_test_datasets_and_loaders
from src.retrieval_model.train import train as train_retrieval
from src.token_classification.dataset import create_dataset
from src.token_classification.train import train as train_token_classification


class ModelFactory:
    def __init__(self, config_path, model_type):
        self.config = ConfigManager(config_path)
        self.model_type = model_type

    def build(self):
        if self.model_type == "retrieval":
            self._build_retrieval_model()
        elif self.model_type == "token_classification":
            self._build_token_classification()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _build_retrieval_model(self):
        config = self.config

        model = Model(
            model_name=config.get("model")["name"],
            pretrained=config.get("model")["pretrained_weights"]
        ).to(config.get("training")["device"])

        criterion = InfoNCELoss(temperature=config.get("training")["loss_temperature"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("training")["learning_rate"],
            weight_decay=config.get("training")["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.get("training")["scheduler_t_0"],
            T_mult=config.get("training")["scheduler_t_mult"]
        )

        train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = create_train_val_test_datasets_and_loaders(
            tokenizer=model.tokenizer,
            transform=model.processor if hasattr(model, 'processor') else None
        )

        train_retrieval(
            model,
            train_loader,
            test_dataset,
            val_dataset,
            criterion,
            optimizer,
            scheduler
        )

    def _build_token_classification(self):
        config = self.config
        tokenizer = DistilBertTokenizerFast.from_pretrained(config.get("model")["pretrained"])
        dataset, id2label, label2id, data_collator = create_dataset(tokenizer)

        model = DistilBertForTokenClassification.from_pretrained(
            config.get("model")["pretrained"],
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )

        train_token_classification(model, dataset, data_collator, tokenizer, id2label)
