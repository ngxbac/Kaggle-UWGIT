import timm
import torch.nn as nn
import torch
from torchvision import models as tmodels
from torchvision.models import mobilenet_v2


class TModels(nn.Module):
    def __init__(self, backbone, num_classes=2):
        super(TModels, self).__init__()
        self.model = timm.create_model(
            backbone, in_chans=3, num_classes=num_classes, pretrained=True)

    def forward(self, x):
        return self.model(x)
