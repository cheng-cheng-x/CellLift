from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Any
from celllift.runtime import torch
from torch import Tensor, nn
from .protocol import TOKEN_DIM
EMBED_DIM = 128

class PaperFSConv(nn.Module):
    feature_dim = 512
    expected_parameters = 630276

    def __init__(self, num_classes: int=4) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(32, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Conv2d(128, 512, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2))
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(512, num_classes)
        self.reset_parameters()
        actual = sum((parameter.numel() for parameter in self.parameters()))
        if actual != self.expected_parameters:
            raise AssertionError(f'FSConv parameter drift: {actual} != {self.expected_parameters}')

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward_features(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1:] != (3, 224, 224):
            raise ValueError(f'PaperFSConv expects [B,3,224,224], got {tuple(images.shape)}')
        return self.global_max_pool(self.features(images)).flatten(1)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        features = self.forward_features(images)
        return {'features': features, 'logits': self.classifier(features)}

def build_paper_crc_resnet18(*, imagenet_weights: bool=True) -> tuple[nn.Module, tuple[str, ...]]:
    try:
        from torchvision.models import ResNet18_Weights
        from celllift.pretrained import resnet18
    except ImportError as exc:
        raise RuntimeError('torchvision is required for the CRC paper baseline') from exc
    weights: Any = ResNet18_Weights.IMAGENET1K_V1 if imagenet_weights else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(512, 2)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.layer4[1].parameters():
        parameter.requires_grad_(True)
    for parameter in model.fc.parameters():
        parameter.requires_grad_(True)
    names = tuple((name for name, parameter in model.named_parameters() if parameter.requires_grad))
    expected_prefixes = ('layer4.1.', 'fc.')
    if any((not name.startswith(expected_prefixes) for name in names)):
        raise AssertionError(f'unexpected trainable CRC parameter: {names}')
    return (model, names)

def crc_adam_groups(model: nn.Module) -> list[dict[str, Any]]:
    backbone = [p for name, p in model.named_parameters() if name.startswith('layer4.1.') and p.requires_grad]
    classifier = [p for name, p in model.named_parameters() if name.startswith('fc.') and p.requires_grad]
    if not backbone or not classifier:
        raise RuntimeError('CRC optimizer groups do not match layer4.1 + classifier')
    return [{'params': backbone, 'lr': 1e-06, 'weight_decay': 0.0001, 'name': 'layer4.1'}, {'params': classifier, 'lr': 2e-06, 'weight_decay': 0.0001, 'name': 'classifier'}]

def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weights = mask.unsqueeze(-1).to(values.dtype)
    count = weights.sum(1)
    return ((values * weights).sum(1) / count.clamp_min(1.0), count.squeeze(-1))

def _mlp(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(EMBED_DIM, output_dim), nn.ReLU(inplace=True))

class FixedSetEncoder(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, kind: str, dropout: float=0.1) -> None:
        super().__init__()
        normalized = kind.lower().replace('_', '')
        if normalized not in {'meanpool', 'deepsets'}:
            raise ValueError('geometry_baselines permits only MeanPool and DeepSets')
        self.kind = normalized
        self.token_mlp = None if normalized == 'meanpool' else _mlp(TOKEN_DIM, EMBED_DIM, dropout)
        pooled_dim = TOKEN_DIM if self.token_mlp is None else EMBED_DIM
        self.set_mlp = _mlp(pooled_dim + 1, EMBED_DIM, dropout)

    def forward(self, tokens: Tensor, mask: Tensor | None=None, count_override: Tensor | None=None) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_DIM:
            raise ValueError(f'expected [B,N,{TOKEN_DIM}] tokens, got {tuple(tokens.shape)}')
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if mask.shape != tokens.shape[:2]:
            raise ValueError('token mask shape mismatch')
        values = tokens if self.token_mlp is None else self.token_mlp(tokens)
        pooled, observed_count = _masked_mean(values, mask.bool())
        count = observed_count if count_override is None else count_override.to(device=tokens.device, dtype=tokens.dtype)
        if count.shape != observed_count.shape:
            raise ValueError('count override shape mismatch')
        if self.token_mlp is not None and count_override is not None:
            raise ValueError('count override is valid only for preaggregated MeanPool')
        output = self.set_mlp(torch.cat((pooled, torch.log1p(count).unsqueeze(-1)), dim=-1))
        return output * (count > 0).unsqueeze(-1).to(output.dtype)

class DualGeometryExpert(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, set_encoder: str, *, dropout: float=0.1) -> None:
        super().__init__()
        self.nucleus_encoder = FixedSetEncoder(set_encoder, dropout)
        self.cell_encoder = FixedSetEncoder(set_encoder, dropout)
        self.fusion = nn.Sequential(nn.Linear(2 * EMBED_DIM, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, *, nucleus_tokens: Tensor, nucleus_mask: Tensor, cell_tokens: Tensor, cell_mask: Tensor, nucleus_count: Tensor | None=None, cell_count: Tensor | None=None) -> Tensor:
        nucleus = self.nucleus_encoder(nucleus_tokens, nucleus_mask, nucleus_count)
        cell = self.cell_encoder(cell_tokens, cell_mask, cell_count)
        return self.fusion(torch.cat((nucleus, cell), dim=-1))

class GeometryClassifier(nn.Module):

    def __init__(self, dataset: str, set_encoder: str, *, dropout: float=0.1) -> None:
        super().__init__()
        self.expert = DualGeometryExpert(set_encoder, dropout=dropout)
        self.head = nn.Linear(EMBED_DIM, 4 if dataset == 'sicapv2' else 1)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        embedding = self.expert(**inputs)
        logits = self.head(embedding)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return {'embedding': embedding, 'logits': logits}
