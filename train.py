import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from model import Model


class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, query_embeds, database_embeds):
        """
        InfoNCE loss implementation
        
        Args:
            query_embeds: Query embeddings [batch_size, embed_dim]
            database_embeds: Database embeddings [batch_size, embed_dim]
            
        Returns:
            loss: InfoNCE loss value
        """
        # Normalize embeddings
        query_embeds = torch.nn.functional.normalize(query_embeds, dim=1)
        database_embeds = torch.nn.functional.normalize(database_embeds, dim=1)
        
        # Calculate similarity matrix
        similarity_matrix = torch.matmul(query_embeds, database_embeds.T) / self.temperature
        
        # Labels are the diagonal elements (positive pairs)
        labels = torch.arange(len(query_embeds)).to(query_embeds.device)
        
        # Calculate loss in both directions (query->database and database->query)
        loss_q2d = self.criterion(similarity_matrix, labels)
        loss_d2q = self.criterion(similarity_matrix.T, labels)
        
        # Total loss is the average of both directions
        return (loss_q2d + loss_d2q) / 2


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc="Training") as pbar:
        for batch_idx, (query_imgs, query_texts, target_imgs) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Move data to device
            query_imgs = query_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Get batch size
            batch_size = query_imgs.size(0)
            
            # Forward pass
            query_embeds = model(query_imgs, query_texts)
            database_embeds = model.encode_database_image(target_imgs)
            
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

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for query_imgs, query_texts, target_imgs in tqdm(val_loader, desc="Validation"):
            # Move data to device
            query_imgs = query_imgs.to(device)
            target_imgs = target_imgs.to(device)
            
            # Forward pass
            query_embeds = model(query_imgs, query_texts)
            database_embeds = model.encode_database_image(target_imgs)
            
            # Calculate loss
            loss = criterion(query_embeds, database_embeds)
            total_loss += loss.item()
            
            # Calculate accuracy
            similarity = torch.matmul(query_embeds, database_embeds.T)
            predictions = similarity.argmax(dim=1)
            labels = torch.arange(len(predictions)).to(device)
            correct += (predictions == labels).sum().item()
            total += len(predictions)
    
    return total_loss / len(val_loader), correct / total

def train_model(model, device, train_loader, val_loader, num_epochs=10):
    # Initialize criterion and optimizer
    criterion = InfoNCELoss(temperature=0.07)
    
    # Two parameter groups with different learning rates
    optimizer = torch.optim.AdamW([
        {'params': model.query_projection.parameters(), 'lr': 1e-4},
        {'params': model.database_projection.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,  # Reset every 5 epochs
        T_mult=2  # Double the reset interval after each reset
    )
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        # Learning rate scheduling
        scheduler.step()
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }, 'best_model.pth')
            print("Saved new best model!")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    # Load best model for return
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model