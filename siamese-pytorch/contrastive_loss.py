import torch
import torch.nn as nn
from torch import Tensor

class ContrastiveLoss(nn.Module):
    "Contrastive loss function"
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
            
    def forward(self, similarity: Tensor, label: Tensor):
        
        # If 
        loss_contrastive = torch.mean((label) * torch.pow(similarity, 2) +
            (1-label) * torch.pow(torch.clamp(self.margin - similarity, min=0.0), 2))

        return loss_contrastive