import torch
from torch import nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b3,
    EfficientNet_B3_Weights,
    resnet50,
    ResNet50_Weights,
)


class EncoderCNN(nn.Module):
    """
    Encoder that swaps in a pretrained CNN (EfficientNet/ResNet) while exposing
    spatial feature maps needed by the attention decoder. The last 1-2 blocks
    remain trainable to avoid overfitting on the small dataset.
    """

    _ARCH_OUT_CHANNELS = {
        "efficientnet_b0": 1280,
        "efficientnet_b3": 1536,
        "resnet50": 2048,
    }

    def __init__(
        self,
        feature_dim: int = 512,
        encoder_name: str = "efficientnet_b0",
        pretrained: bool = True,
        trainable_blocks: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder_name = encoder_name.lower()
        self.trainable_blocks = max(trainable_blocks, 0)

        self.backbone = self._build_backbone(pretrained)
        backbone_dim = self._ARCH_OUT_CHANNELS[self.encoder_name]

        # Project backbone channels to the feature_dim expected by the decoder.
        self.feature_projection = (
            nn.Linear(backbone_dim, feature_dim)
            if backbone_dim != feature_dim
            else nn.Identity()
        )

        # Freeze all layers except the last N blocks to reduce overfitting.
        self._freeze_backbone_layers()

    def _build_backbone(self, pretrained: bool) -> nn.Sequential:
        if self.encoder_name == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            return backbone.features
        if self.encoder_name == "efficientnet_b3":
            weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b3(weights=weights)
            return backbone.features
        if self.encoder_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
            return nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            )

        raise ValueError(
            f"Unsupported encoder '{self.encoder_name}'. "
            "Please use efficientnet_b0, efficientnet_b3, or resnet50."
        )

    def _freeze_backbone_layers(self) -> None:
        blocks = list(self.backbone.children())
        if not blocks:
            return

        trainable_blocks = min(self.trainable_blocks, len(blocks))
        trainable_start = len(blocks) - trainable_blocks

        for idx, block in enumerate(blocks):
            requires_grad = idx >= trainable_start
            for param in block.parameters():
                param.requires_grad = requires_grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract spatial feature map before any global pooling.
        features = self.backbone(x)
        batch_size, channels, height, width = features.shape

        # Flatten spatial grid (H*W pixels) while preserving feature dim.
        features = features.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
        features = self.feature_projection(features)

        return features

    def trainable_parameters(self):
        """Convenience helper used by the training script for optimizer groups."""
        return (param for param in self.parameters() if param.requires_grad)
