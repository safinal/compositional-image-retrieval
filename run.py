import argparse
import torch

from src.config import ConfigManager
from src.model import Model
from src.loss import InfoNCELoss
from src.dataset import create_train_val_test_datasets_and_loaders
from src.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Rayan International AI Contest: Compositional Retrieval")
    parser.add_argument("--config", type=str, default="./config/config.yaml", help="Path to the config file")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config_path = args.config
    config = ConfigManager(config_path)  # Initialize the singleton with the config file

    model = Model(model_name=config["model"]["name"], pretrained=config["model"]["pretrained_weights"]).to(config["training"]["device"])
    criterion = InfoNCELoss(temperature=config.loss_temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["device"], weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config["training"]["scheduler_t_0"], T_mult=config["training"]["scheduler_t_mult"])


    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = create_train_val_test_datasets_and_loaders(
        tokenizer=model.tokenizer,
        transform=model.processor if hasattr(model, 'processor') else None
    )

    train(
        model,
        train_loader,
        test_dataset,
        criterion,
        optimizer,
        scheduler
    )

if __name__ == "__main__":
    main()
