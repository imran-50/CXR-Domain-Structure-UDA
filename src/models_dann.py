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
        super(DANN, self).__init__()
        try:
            resnet = models.resnet18(weights='IMAGENET1K_V1')
        except (AttributeError, TypeError):
            resnet = models.resnet18(pretrained=True)

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        num_features = resnet.fc.in_features

        self.label_predictor = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        class_output = self.label_predictor(features)
        reversed_features = GradientReversal.apply(features, alpha)
        domain_output = self.domain_classifier(reversed_features)
        return class_output, domain_output
