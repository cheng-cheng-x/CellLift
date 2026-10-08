from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from torch import Tensor, nn
from celllift.breast_roi_baseline.models import BRACSROIModel, RGBMaskTileEncoder

def freeze_except_last_parameter_tensors(model: nn.Module, count: int=10) -> tuple[str, ...]:
    named = list(model.named_parameters())
    if count <= 0 or count > len(named):
        raise ValueError('invalid trainable RGB parameter-tensor count')
    for _, parameter in named:
        parameter.requires_grad_(False)
    for _, parameter in named[-count:]:
        parameter.requires_grad_(True)
    return tuple((name for name, _ in named[-count:]))

class BRACSRGBMaskROIModel(nn.Module):

    def __init__(self, *, set_encoder: str, num_classes: int, dropout: float=0.1, imagenet_weights: bool=True, trainable_parameter_tensors: int=10) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights
            from celllift.pretrained import resnet18
        except ImportError as exc:
            raise RuntimeError('torchvision is required for BRACS RGB') from exc
        weights = ResNet18_Weights.DEFAULT if imagenet_weights else None
        self.rgb_backbone = resnet18(weights=weights)
        self.rgb_backbone.fc = nn.Identity()
        self.trainable_rgb_parameter_names = freeze_except_last_parameter_tensors(self.rgb_backbone, trainable_parameter_tensors)
        self.roi_model = BRACSROIModel(RGBMaskTileEncoder(set_encoder, rgb_dim=512, dropout=dropout), num_classes=num_classes, dropout=dropout)

    def forward(self, images: Tensor, roi_offsets: Tensor, nucleus_tokens: Tensor, cell_tokens: Tensor, nucleus_mask: Tensor, cell_mask: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError('BRACS RGB images must have shape [tiles,3,224,224]')
        rgb = self.rgb_backbone(images)
        return self.roi_model(roi_offsets, nucleus_tokens=nucleus_tokens, cell_tokens=cell_tokens, nucleus_mask=nucleus_mask, cell_mask=cell_mask, rgb_features=rgb)
