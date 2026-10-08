from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor, nn
import torch.nn.functional as F

def _conv_bn(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True))

def _dw(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch), nn.ReLU6(inplace=True), nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU6(inplace=True))
MOBILENET_DW = ((32, 1), (64, 2), (64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (256, 1), (256, 1), (256, 1), (256, 1), (512, 2), (512, 1))

class MobileNetV1Half(nn.Module):

    def __init__(self, classes: int=4):
        super().__init__()
        blocks = [_conv_bn(3, 16, 2)]
        in_ch = 16
        for out_ch, stride in MOBILENET_DW:
            blocks.append(_dw(in_ch, out_ch, stride))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(512, classes)

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.embedding(images))

    def embedding(self, images: Tensor) -> Tensor:
        return self.pool(self.features(images)).flatten(1)

class MobileNetV1HalfFCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.backbone = MobileNetV1Half(4)
        self.top_pool = nn.AvgPool2d(7, stride=1, padding=3)
        self.classifier = nn.Conv2d(512, 4, 1)

    def load_classifier_from_dense(self, weight: Tensor, bias: Tensor) -> None:
        self.classifier.weight.data.copy_(weight.detach().reshape(weight.shape[0], weight.shape[1], 1, 1))
        self.classifier.bias.data.copy_(bias)

    def forward(self, images: Tensor) -> Tensor:
        hidden = self.backbone.features(images)
        logits = self.classifier(self.top_pool(hidden))
        logits = F.interpolate(logits, size=(images.shape[-2], images.shape[-1]), mode='nearest')
        return logits.softmax(1)

class DeepSetsWindow(nn.Module):

    def __init__(self, in_dim: int, classes: int=4, width: int=64, dropout: float=0.1):
        super().__init__()
        self.encoder = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, classes))

    def encode(self, tokens: Tensor, mask: Tensor) -> Tensor:
        tokens = torch.nan_to_num(tokens).clamp(-20, 20)
        hidden = self.encoder(tokens) * mask.unsqueeze(-1)
        count = mask.sum(1).clamp_min(1.0)
        mean = hidden.sum(1) / count.unsqueeze(-1)
        return torch.cat((mean, torch.log1p(count).unsqueeze(-1)), -1)

    def forward(self, tokens: Tensor, mask: Tensor) -> Tensor:
        return self.head(self.encode(tokens, mask))

class NucleusMLP(nn.Module):

    def __init__(self, in_dim: int, classes: int=6, width: int=64, dropout: float=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, classes))

    def forward(self, tokens: Tensor) -> Tensor:
        return self.net(tokens)

class RelationExpert(nn.Module):

    def __init__(self, node_dim: int, edge_dim: int, classes: int=6, width: int=64, dropout: float=0.1):
        super().__init__()
        self.node = nn.Sequential(nn.Linear(node_dim, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width))
        self.message = nn.Sequential(nn.Linear(width * 2 + edge_dim, 32), nn.SiLU(), nn.Linear(32, 32))
        self.local = nn.Sequential(nn.Linear(width + 32, width), nn.SiLU())
        self.head = nn.Linear(width, classes)

    def encode(self, nodes: Tensor, edges: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        hidden = self.node(nodes)
        if src.numel() == 0:
            neighbour = hidden.new_zeros((hidden.shape[0], 32))
        else:
            payload = torch.cat((hidden[src], hidden[dst], edges), -1)
            msgs = self.message(payload)
            neighbour = hidden.new_zeros((hidden.shape[0], 32))
            neighbour.index_add_(0, dst, msgs)
            counts = hidden.new_zeros((hidden.shape[0], 1))
            counts.index_add_(0, dst, torch.ones(dst.shape[0], 1, device=hidden.device, dtype=hidden.dtype))
            neighbour = neighbour / counts.clamp_min(1)
        return self.local(torch.cat((hidden, neighbour), -1))

    def forward(self, nodes: Tensor, edges: Tensor, src: Tensor, dst: Tensor) -> Tensor:
        return self.head(self.encode(nodes, edges, src, dst))

class FeatureFusion(nn.Module):

    def __init__(self, image_dim: int, geom_dim: int, classes: int, dropout: float=0.1):
        super().__init__()
        self.image = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.geom = nn.Sequential(nn.Linear(geom_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 64))
        self.norm = nn.LayerNorm(192)
        self.head = nn.Linear(192, classes)

    def forward(self, image: Tensor, geom: Tensor) -> Tensor:
        return self.head(self.norm(torch.cat((self.image(image), self.geom(geom)), -1)))

class WindowFusion(nn.Module):

    def __init__(self, image_dim: int, geom_dim: int, classes: int=4, dropout: float=0.1):
        super().__init__()
        self.sets = DeepSetsWindow(geom_dim, classes=classes, dropout=dropout)
        self.image = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, 128), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.head = nn.Sequential(nn.Linear(128 + 64 + 1, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, classes))

    def forward(self, image: Tensor, tokens: Tensor, mask: Tensor) -> Tensor:
        geom = self.sets.encode(tokens, mask)
        return self.head(torch.cat((self.image(image), geom), -1))

class SpatialFieldNet(nn.Module):

    def __init__(self, classes: int=4, dropout: float=0.1):
        super().__init__()
        self.field = nn.Sequential(nn.Conv2d(10, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU())
        self.dino = nn.Sequential(nn.Conv2d(384, 64, 1), nn.SiLU())
        self.fuse = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Linear(64, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, classes))

    def forward(self, field: Tensor, dino: Tensor) -> Tensor:
        geom = self.field(field)
        if dino.shape[-2:] != geom.shape[-2:]:
            dino = F.interpolate(dino, size=geom.shape[-2:], mode='bilinear', align_corners=False)
        image = self.dino(dino)
        hidden = self.fuse(torch.cat((geom, image), 1)).flatten(1)
        return self.head(hidden)

class ImageNodeExpert(nn.Module):

    def __init__(self, node_dim: int, edge_dim: int, classes: int, width: int=64, dropout: float=0.1, pool: bool=True):
        super().__init__()
        self.pool = pool
        self.appear = nn.Sequential(nn.Linear(384, width), nn.SiLU())
        self.morph = nn.Sequential(nn.Linear(node_dim, width), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Dropout(dropout))
        self.relation = RelationExpert(width, edge_dim, classes=classes, width=width, dropout=dropout)
        self.window = nn.Sequential(nn.Linear(width + 1, width), nn.SiLU(), nn.Linear(width, classes)) if pool else None

    def forward(self, dino: Tensor, nodes: Tensor, edges: Tensor, src: Tensor, dst: Tensor, mask: Tensor | None=None) -> Tensor:
        hidden = self.fuse(torch.cat((self.appear(dino), self.morph(nodes)), -1))
        local = self.relation.encode(hidden, edges, src, dst)
        if not self.pool:
            return self.relation.head(local)
        if local.shape[0] == 0:
            hidden = local.new_zeros(self.window[0].in_features)
            return self.window(hidden.unsqueeze(0)).squeeze(0)
        if mask is None:
            mask = torch.ones(local.shape[0], device=local.device)
        count = mask.sum().clamp_min(1.0)
        mean = (local * mask.unsqueeze(-1)).sum(0) / count
        return self.window(torch.cat((mean, torch.log1p(count).reshape(1))).unsqueeze(0)).squeeze(0)
