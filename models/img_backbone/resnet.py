"""
Image backbone networks (ResNet variants)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List


class ResNetBackbone(nn.Module):
    """
    ResNet backbone for image feature extraction
    Supports multi-scale feature extraction
    """

    def __init__(
        self,
        variant: str = 'resnet50',
        pretrained: bool = True,
        frozen_stages: int = 1,
        out_indices: tuple = (0, 1, 2, 3)
    ):
        """
        Args:
            variant: ResNet variant (resnet18/34/50/101)
            pretrained: use ImageNet pretrained weights
            frozen_stages: number of frozen stages
            out_indices: indices of feature maps to output
        """
        super().__init__()

        # Load backbone
        if variant == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            self.out_channels = [64, 128, 256, 512]
        elif variant == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
            self.out_channels = [64, 128, 256, 512]
        elif variant == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        elif variant == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            self.out_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Extract layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32

        self.out_indices = out_indices

        # Freeze stages
        self._freeze_stages(frozen_stages)

    def _freeze_stages(self, frozen_stages: int):
        """Freeze parameters in early stages"""
        if frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in [self.conv1.parameters(), self.bn1.parameters()]:
                for p in param:
                    p.requires_grad = False

        for i in range(1, frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) input image

        Returns:
            features: list of feature maps at different scales
        """
        features = []

        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        x1 = self.layer1(x)    # 1/4
        x2 = self.layer2(x1)   # 1/8
        x3 = self.layer3(x2)   # 1/16
        x4 = self.layer4(x3)   # 1/32

        stage_features = [x1, x2, x3, x4]

        for idx in self.out_indices:
            features.append(stage_features[idx])

        return features


def build_image_backbone(config: dict) -> nn.Module:
    """
    Build image backbone from config

    Args:
        config: dict with keys 'type', 'pretrained', 'frozen_stages'

    Returns:
        backbone: image backbone module
    """
    variant = config.get('type', 'resnet50')
    pretrained = config.get('pretrained', True)
    frozen_stages = config.get('frozen_stages', 1)

    return ResNetBackbone(
        variant=variant,
        pretrained=pretrained,
        frozen_stages=frozen_stages
    )
