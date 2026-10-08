from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from torch import Tensor, nn
from .fusion import DualSetFusion

class SICAPClassifier(nn.Module):

    def __init__(self, fusion: DualSetFusion) -> None:
        super().__init__()
        self.fusion = fusion
        self.grade_head = nn.Linear(fusion.output_dim, 4)
        self.cribriform_head = nn.Linear(fusion.output_dim, 1)

    def forward(self, **inputs: Tensor) -> dict[str, Tensor]:
        embedding = self.fusion(**inputs)
        return {'embedding': embedding, 'grade_logits': self.grade_head(embedding), 'cribriform_logits': self.cribriform_head(embedding).squeeze(-1)}
