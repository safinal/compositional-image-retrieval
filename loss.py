import torch

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