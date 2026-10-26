import torch

class BCEDiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # pixel-wise binary cross-entropy
        bce_loss = self.bce(logits, targets)

        # continous probs for differentiability
        probabilities = torch.sigmoid(logits)

        # calculating soft dice for each image
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))

        denominator = (probabilities + targets).sum(dim=(1, 2, 3))

        smooth = 1.0

        dice = (2 * intersection + smooth) / (denominator + smooth)

        dice_loss = 1 - dice.mean()

        return bce_loss + dice_loss
