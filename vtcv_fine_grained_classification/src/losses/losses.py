import torch
import torch.nn as nn
import torch.nn.functional as F


''' def diversity_loss(att_map1, att_map2, label, margin=1.0, epsilon = 1e-8):
    """
    Compute diversity loss between two attention maps.

    :param att_map1: (B, 1, H, W)  attention maps for batch, e.g. top-1 class
    :param att_map2: (B, 1, H, W)  attention maps for batch, e.g. top-2 class
    :param label:    (B,)          1 = "same class" (positive pair), 0 = "different class" (negative pair)
    :param margin:   float         margin distance for negative pairs

    :return: contrastive_loss (scalar)
    """
    # 1) Compute elementwise (squared) distance
    #    For stability, do it in two parts: squared distance, then sqrt if needed
    #    MSE across spatial dims => shape [B]
    dist_sq = F.mse_loss(att_map1, att_map2, reduction='none')  # shape [B,1,H,W]
    dist_sq = dist_sq.mean(dim=[1,2,3])  # average across channel/height/width => shape [B]
    dist = torch.sqrt(dist_sq + epsilon)    # L2 distance per sample => shape [B]

    # 2) Diversity loss
    #    loss = label * dist^2 + (1 - label) * (max(0, margin - dist))^2
    pos_loss = label * dist_sq  # if label=1, push them close => minimize dist^2
    neg_loss = (1 - label) * F.relu(margin - dist).pow(2)  # if label=0, push them apart => dist > margin
    loss = pos_loss + neg_loss

    return loss.mean() '''

class Diversity_loss(nn.Module):
    def __init__(self, margin=1.0, epsilon = 1e-8):
        """
        Compute diversity loss between two attention maps.

        :param att_map1: (B, 1, H, W)  attention maps for batch, e.g. top-1 class
        :param att_map2: (B, 1, H, W)  attention maps for batch, e.g. top-2 class
        :param label:    (B,)          1 = "same class" (positive pair), 0 = "different class" (negative pair)
        :param margin:   float         margin distance for negative pairs

        :return: contrastive_loss (scalar)
        """
        super(Diversity_loss, self).__init__()
        self.margin = margin
        self.epsilon = epsilon

    def forward(self, att_map1, att_map2, label):
        # 1) Compute elementwise (squared) distance
        #    For stability, do it in two parts: squared distance, then sqrt if needed
        #    MSE across spatial dims => shape [B]
        dist_sq = F.mse_loss(att_map1, att_map2, reduction='none')  # shape [B,1,H,W]
        dist_sq = dist_sq.mean(dim=[1,2,3])  # average across channel/height/width => shape [B]
        dist = torch.sqrt(dist_sq + self.epsilon)    # L2 distance per sample => shape [B]

        # 2) Diversity loss
        #    loss = label * dist^2 + (1 - label) * (max(0, margin - dist))^2
        pos_loss = label * dist_sq  # if label=1, push them close => minimize dist^2
        neg_loss = (1 - label) * F.relu(self.margin - dist).pow(2)  # if label=0, push them apart => dist > margin
        loss = pos_loss + neg_loss

        return loss.mean()



class MosLoss(nn.Module):
    def __init__(self, num_classes, loss_dict):
        super(MosLoss, self).__init__()
        self.num_classes = num_classes
        self.loss_dict = loss_dict #copy.deepcopy(config["loss_dict"])

        assert len(self.loss_dict) > 0, "Loss dictionary must not be empty."

        # Extract weights and magnitude scales using list comprehension
        self.lweights = torch.tensor([v.pop("weight") for v in self.loss_dict.values()])
        self.mag_scale = torch.tensor([v.pop("mag_scale") for v in self.loss_dict.values()])

        assert torch.isclose(self.lweights.sum(), torch.tensor(1.0)), "Weights must sum up to 1."
        self.loss_functions = [globals()[loss_name](num_classes, **params) for loss_name, params in self.loss_dict.items()]
        
    def forward(self, inputs, targets, reduction=None):
        device = inputs.device
        self._move_to_device(device)
      
        targets = self._convert_targets(targets, device)

        loss = sum(loss_fn(inputs, targets) * scale for loss_fn, scale in zip(self.loss_functions, self.mag_scale))
        return (loss * self.lweights).sum()

    def _convert_targets(self, targets, device):
        temp = torch.zeros((targets.size(0), self.num_classes), device=device)
        return temp.scatter_(1, targets.view(-1, 1).long(), 1)

    def _move_to_device(self, device):
        """Move tensors to the specified device."""
        self.lweights = self.lweights.to(device)
        self.mag_scale = self.mag_scale.to(device)

    
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        # Define weight tensor with num_classes values, each set to 1 if not provided
        if alpha is None:
            self.alpha = torch.ones(num_classes).to('cuda')
        else:
            if isinstance(alpha, (list, torch.Tensor)) and len(alpha) == num_classes:
                self.alpha = torch.tensor(alpha, dtype=torch.float32).to('cuda')
            else:
                raise ValueError(f"Weight tensor must have {num_classes} elements, one for each class.")

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    




