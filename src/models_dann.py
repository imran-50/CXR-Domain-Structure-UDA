import torch
import torch.nn as nn
import torchvision.models as models

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DANN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features
        self.label_predictor = nn.Sequential(
            nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, num_classes)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, 2)
        )
    def forward(self, x, alpha=1.0):
        feats = self.feature_extractor(x).view(x.size(0), -1)
        cls_out = self.label_predictor(feats)
        rev_feats = GradientReversal.apply(feats, alpha)
        dom_out = self.domain_classifier(rev_feats)
        return cls_out, dom_out
