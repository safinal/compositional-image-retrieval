import torch
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc="Training") as pbar:
        for batch_idx, (query_imgs, query_texts, target_imgs) in enumerate(pbar):
            optimizer.zero_grad()
            
            # Move data to device
            query_imgs = query_imgs.to(device)
            target_imgs = target_imgs.to(device)
            query_texts = query_texts.to(device)
            
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

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for query_imgs, query_texts, target_imgs in tqdm(loader, desc="Validation"):
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
    
    return total_loss / len(loader), correct / total

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