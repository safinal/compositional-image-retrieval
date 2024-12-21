import torch
from tqdm import tqdm
import os

from src.config import ConfigManager
from evaluate import evaluate


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc="Training") as pbar:
        for batch_idx, (query_imgs, query_texts, target_imgs, qt) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Move data to device
            query_imgs = query_imgs.to(device)
            target_imgs = target_imgs.to(device)
            query_texts = query_texts.to(device)
            
            # Forward pass
            query_embeds = model.feature_extractor.encode_image(query_imgs)
            for j, text in enumerate(qt):
                words = text.split()
                assert words[0] == 'add' and words[2] == 'and' and words[3] == 'remove' and len(words) == 5
                positive_object = words[1]
                negative_object = words[4]
                with torch.no_grad():
                    query_embeds[j] += model.feature_extractor.encode_text(model.tokenizer(f"a photo of a {positive_object}.").to(device))[0]
                    query_embeds[j] -= model.feature_extractor.encode_text(model.tokenizer(f"a photo of a {negative_object}.").to(device))[0]
            database_embeds = model.feature_extractor.encode_image(target_imgs)
            
            # Calculate loss
            loss = criterion(query_embeds, database_embeds)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
    
    return total_loss / len(train_loader)




def train(model, train_loader, test_dataset, val_dataset, criterion, optimizer, scheduler):
    num_epochs = ConfigManager().get("training")["num_epochs"]
    model.set_param_trainable_mode(model.feature_extractor.visual.head, True)
    test_acc = 100*evaluate(model, test_dataset)
    val_acc = 100*evaluate(model, val_dataset)
    best_acc = test_acc + val_acc
    print(f"Zero-shot Test Accuracy: {test_acc}\n")
    print(f"Zero-shot Val Accuracy: {val_acc}\n")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_epoch(model, train_loader, criterion, optimizer, ConfigManager().get("training")["device"])
        test_acc = 100*evaluate(model, test_dataset)
        val_acc = 100*evaluate(model, val_dataset)
        print(f"Test Accuracy: {test_acc}")
        print(f"Val Accuracy: {val_acc}\n")
        scheduler.step()
        model.save(os.path.join(os.getcwd(), f"weights_epoch_{epoch + 1}.pth"))
        if test_acc + val_acc > best_acc:
            model.save(os.path.join(os.getcwd(), f"best_model_epoch_{epoch + 1}.pth"))
            best_acc = test_acc + val_acc
