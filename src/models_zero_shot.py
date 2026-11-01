import torch
import torch.nn as nn
import timm

class ConvNeXtClassifier(nn.Module):
    def __init__(self, backbone_name='convnextv2_tiny', num_classes=2, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            in_chans=3
        )
        n_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)
