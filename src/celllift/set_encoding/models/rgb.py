from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
from torch import Tensor, nn

class SICAPFSConv(nn.Module):
    feature_dim = 512

    def __init__(self, num_classes: int=4) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(32, 124, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(124, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError(f'FSConv expects [B, 3, 224, 224], got {tuple(images.shape)}')
        return self.global_max_pool(self.features(images)).flatten(1)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.forward_features(images)
        return {'features': features, 'logits': self.classifier(features)}

def freeze_except_last_parameter_tensors(model: nn.Module, count: int=10) -> tuple[str, ...]:
    named_parameters = list(model.named_parameters())
    if count <= 0 or count > len(named_parameters):
        raise ValueError(f'count must be in [1, {len(named_parameters)}], got {count}')
    for _, parameter in named_parameters:
        parameter.requires_grad_(False)
    trainable = named_parameters[-count:]
    for _, parameter in trainable:
        parameter.requires_grad_(True)
    return tuple((name for name, _ in trainable))

def build_crc_resnet18(*, imagenet_weights: bool=True, trainable_parameter_tensors: int=10) -> tuple[nn.Module, tuple[str, ...]]:
    try:
        from torchvision.models import ResNet18_Weights
        from celllift.pretrained import resnet18
    except ImportError as exc:
        raise RuntimeError('torchvision is required for the CRC RGB baseline') from exc
    weights: Any = ResNet18_Weights.DEFAULT if imagenet_weights else None
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()
    trainable_names = freeze_except_last_parameter_tensors(backbone, trainable_parameter_tensors)
    return (backbone, trainable_names)

class CRCResNet18FeatureExtractor(nn.Module):
    feature_dim = 512

    def __init__(self, *, imagenet_weights: bool=True, trainable_parameter_tensors: int=10) -> None:
        super().__init__()
        self.backbone, trainable_names = build_crc_resnet18(imagenet_weights=imagenet_weights, trainable_parameter_tensors=trainable_parameter_tensors)
        self.trainable_parameter_names = trainable_names

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone(images)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise RuntimeError(f'unexpected ResNet18 output shape {tuple(features.shape)}')
        return features

class CRCTileClassifier(nn.Module):

    def __init__(self, *, imagenet_weights: bool=True, trainable_parameter_tensors: int=10) -> None:
        super().__init__()
        self.extractor = CRCResNet18FeatureExtractor(imagenet_weights=imagenet_weights, trainable_parameter_tensors=trainable_parameter_tensors)
        self.classifier = nn.Linear(self.extractor.feature_dim, 1)

    @property
    def trainable_backbone_parameter_names(self) -> tuple[str, ...]:
        return tuple((f'extractor.backbone.{name}' for name in self.extractor.trainable_parameter_names))

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.extractor(images)
        return {'features': features, 'logits': self.classifier(features).squeeze(-1)}
