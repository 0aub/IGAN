import torch
import torch.nn as nn

# Reference: https://github.com/Hanzy1996/CE-GZSL

class SupervisedContrastiveLoss(nn.Module):
    """
    A class representing the Supervised Contrastive Loss.
    
    Attributes:
        temperature (float): A temperature scaling factor to apply to the logits.
    """
    def __init__(self, temperature=0.07):
        """
        Initializes the SupConLoss module.
        
        Args:
            temperature (float): Temperature scaling factor for logits normalization.
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Forward pass of the loss function.
        
        Args:
            features (torch.Tensor): Feature representations of the batch, shape (batch_size, feature_dim).
            labels (torch.Tensor): Ground-truth labels of the batch, shape (batch_size,).
        
        Returns:
            torch.Tensor: Computed loss value.
        """
        # Determine the device ('cuda' or 'cpu') based on where the features are located
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # Create a mask for positive samples

        # Compute similarity matrix divided by temperature
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)

        # Apply log-sum-exp trick for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast cases (diagonal elements)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()  # Identify samples that are single in the batch

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean log-likelihood over positive pairs, avoiding division by zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + single_samples)

        # Compute the contrastive loss, ignoring single samples
        loss = - mean_log_prob_pos * (1 - single_samples)
        loss = loss.sum() / (loss.shape[0] - single_samples.sum())

        return loss