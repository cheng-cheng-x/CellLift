from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from typing import Optional
from celllift.runtime import torch
from torch import Tensor, nn
from celllift.conditional_geometry.mask_protocol import MASK_RESIDUAL_TOKEN_DIM
EMBED_DIM = 128

def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weight = mask.unsqueeze(-1).to(values.dtype)
    count = weight.sum(1)
    return ((values * weight).sum(1) / count.clamp_min(1.0), count.squeeze(-1))

def _mlp(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(EMBED_DIM, output_dim), nn.ReLU(inplace=True))

class MaskResidualSetEncoder(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, kind: str, dropout: float) -> None:
        super().__init__()
        normalized = kind.lower().replace('_', '')
        if normalized not in {'meanpool', 'deepsets'}:
            raise ValueError('mask mask_conditioning permits only meanpool or deepsets')
        self.token_mlp = None if normalized == 'meanpool' else _mlp(MASK_RESIDUAL_TOKEN_DIM, EMBED_DIM, dropout)
        pooled_dim = MASK_RESIDUAL_TOKEN_DIM if self.token_mlp is None else EMBED_DIM
        self.set_mlp = _mlp(pooled_dim + 1, EMBED_DIM, dropout)

    def forward(self, tokens: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        if tokens.ndim != 3 or tokens.shape[-1] != MASK_RESIDUAL_TOKEN_DIM:
            raise ValueError(f'expected [B,N,{MASK_RESIDUAL_TOKEN_DIM}] mask-residual tokens')
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if mask.shape != tokens.shape[:2]:
            raise ValueError('mask residual token mask shape mismatch')
        values = tokens if self.token_mlp is None else self.token_mlp(tokens)
        pooled, count = _masked_mean(values, mask.bool())
        output = self.set_mlp(torch.cat((pooled, torch.log1p(count).unsqueeze(-1)), -1))
        return output * (count > 0).unsqueeze(-1).to(output.dtype)

class MaskDualSetFusion(nn.Module):
    output_dim = EMBED_DIM

    def __init__(self, set_encoder: str='meanpool', object_mode: str='both', use_rgb: bool=False, rgb_dim: int=512, dropout: float=0.1) -> None:
        super().__init__()
        if object_mode != 'both' or use_rgb:
            raise ValueError('mask mask_conditioning requires both capacity-matched branches and no RGB')
        self.object_mode, self.use_rgb, self.rgb_dim = (object_mode, use_rgb, rgb_dim)
        self.nucleus_encoder = MaskResidualSetEncoder(set_encoder, dropout)
        self.cell_encoder = MaskResidualSetEncoder(set_encoder, dropout)
        self.fusion = nn.Sequential(nn.Linear(2 * EMBED_DIM, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, *, nucleus_tokens: Tensor, nucleus_mask: Tensor, cell_tokens: Tensor, cell_mask: Tensor) -> Tensor:
        return self.fusion(torch.cat((self.nucleus_encoder(nucleus_tokens, nucleus_mask), self.cell_encoder(cell_tokens, cell_mask)), -1))
