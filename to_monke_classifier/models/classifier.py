import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class MonkeyClassifier(nn.Module):
    def __init__(self, num_classes: int, fc_params):
        super().__init__()
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.base_model.parameters():
            param.requires_grad = False
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(in_features, fc_params["linear_layer_out_features"]),
            nn.ReLU(),
            nn.Dropout(fc_params["dropout_rate"]),
            nn.Linear(fc_params["linear_layer_out_features"], num_classes),
        )

    def forward(self, x):
        return self.base_model(x)
