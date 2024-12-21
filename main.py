import torch

import config
from model import Model
from loss import InfoNCELoss
from dataset import create_train_validation_test_datasets_and_loaders
from train import train


model = Model(model_name=config.model_name, pretrained=config.pretrained_weights).to(config.device)
criterion = InfoNCELoss(temperature=config.loss_temperature)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler_t_0, T_mult=config.scheduler_t_mult)


train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = create_train_validation_test_datasets_and_loaders(
    tokenizer=model.tokenizer,
    transform=model.processor if hasattr(model, 'processor') else None
)


train(
    model,
    train_loader,
    test_dataset,
    val_dataset,
    criterion,
    optimizer,
    scheduler
)
