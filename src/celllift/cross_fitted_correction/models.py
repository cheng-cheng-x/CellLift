from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import Tensor, nn
TOKEN_DIM = 41
EMBED_DIM = 128
MULTISTAT_DIM = TOKEN_DIM * 5 + 1

def _mlp(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, EMBED_DIM), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(EMBED_DIM, output_dim), nn.ReLU(inplace=True))

def _masked_mean(values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    weights = mask.unsqueeze(-1).to(values.dtype)
    count = weights.sum(1)
    return ((values * weights).sum(1) / count.clamp_min(1), count.squeeze(-1))

class SetBranch(nn.Module):

    def __init__(self, kind: str, dropout: float=0.1) -> None:
        super().__init__()
        if kind not in {'deepsets', 'multistat', 'balanced_multistat', 'gated_attention'}:
            raise ValueError(f'unsupported cross_fitted_correction pooler: {kind}')
        self.kind = kind
        if kind == 'multistat':
            self.summary_mlp = _mlp(MULTISTAT_DIM, EMBED_DIM, dropout)
        elif kind == 'balanced_multistat':

            def block(input_dim: int) -> nn.Sequential:
                return nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 64), nn.ReLU(inplace=True))
            self.mask_summary_mlp = block(5 * 36 + 1)
            self.three_d_summary_mlp = block(5 * 5 + 1)
        else:
            self.token_mlp = _mlp(TOKEN_DIM, EMBED_DIM, dropout)
            self.set_mlp = _mlp(EMBED_DIM + 1, EMBED_DIM, dropout)
            if kind == 'gated_attention':
                self.attention_v = nn.Linear(EMBED_DIM, EMBED_DIM)
                self.attention_u = nn.Linear(EMBED_DIM, EMBED_DIM)
                self.attention_w = nn.Linear(EMBED_DIM, 1, bias=False)

    def forward(self, tokens: Tensor | None=None, mask: Tensor | None=None, summary: Tensor | None=None) -> Tensor:
        if self.kind in {'multistat', 'balanced_multistat'}:
            if summary is None or summary.ndim != 2 or summary.shape[-1] != MULTISTAT_DIM:
                raise ValueError('MultiStat requires [B,206] precomputed summaries')
            if self.kind == 'multistat':
                return self.summary_mlp(summary)
            statistics = summary[:, :205].reshape(len(summary), 5, 41)
            count = summary[:, -1:]
            mask = torch.cat((statistics[:, :, :36].reshape(len(summary), 180), count), dim=-1)
            three_d = torch.cat((statistics[:, :, 36:].reshape(len(summary), 25), count), dim=-1)
            return torch.cat((self.mask_summary_mlp(mask), self.three_d_summary_mlp(three_d)), dim=-1)
        if tokens is None or mask is None or tokens.ndim != 3 or (tokens.shape[-1] != TOKEN_DIM):
            raise ValueError('DeepSets/attention require padded [B,N,41] tokens and a mask')
        hidden = self.token_mlp(tokens)
        if self.kind == 'deepsets':
            pooled, count = _masked_mean(hidden, mask)
        else:
            score = self.attention_w(torch.tanh(self.attention_v(hidden)) * torch.sigmoid(self.attention_u(hidden))).squeeze(-1)
            score = score.masked_fill(~mask, -torch.inf)
            weight = torch.softmax(score, dim=1)
            pooled = torch.sum(hidden * weight.unsqueeze(-1), dim=1)
            count = mask.sum(1).to(hidden.dtype)
        return self.set_mlp(torch.cat((pooled, torch.log1p(count).unsqueeze(-1)), dim=-1))

class DualSetSummary(nn.Module):

    def __init__(self, pooler: str, dropout: float=0.1) -> None:
        super().__init__()
        self.pooler = pooler
        self.nucleus = SetBranch(pooler, dropout)
        self.cell = SetBranch(pooler, dropout)
        self.fusion = _mlp(2 * EMBED_DIM, EMBED_DIM, dropout)

    def forward(self, **inputs: Tensor) -> Tensor:
        if self.pooler in {'multistat', 'balanced_multistat'}:
            n = self.nucleus(summary=inputs['nucleus_summary'])
            c = self.cell(summary=inputs['cell_summary'])
        else:
            n = self.nucleus(inputs['nucleus_tokens'], inputs['nucleus_mask'])
            c = self.cell(inputs['cell_tokens'], inputs['cell_mask'])
        return self.fusion(torch.cat((n, c), dim=-1))

class OOFCorrectionModel(nn.Module):

    def __init__(self, dataset: str, pooler: str, fusion: str, *, dropout: float=0.1, max_delta_logit: float=2.0) -> None:
        super().__init__()
        if fusion not in {'correction', 'correction_gate'}:
            raise ValueError('invalid correction fusion')
        self.dataset, self.fusion_kind = (dataset, fusion)
        self.geometry = DualSetSummary(pooler, dropout)
        classes = 4 if dataset == 'sicapv2' else 1
        self.delta_head = nn.Linear(EMBED_DIM, classes)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.max_delta_logit = float(max_delta_logit)
        if fusion == 'correction_gate':
            self.gate_beta = nn.Parameter(torch.tensor([-2.0, 0.0, 0.0]))
        else:
            self.register_parameter('gate_beta', None)

    def forward(self, *, rgb_logits: Tensor, **geometry_inputs: Tensor) -> dict[str, Tensor]:
        embedding = self.geometry(**geometry_inputs)
        raw_delta = self.delta_head(embedding)
        if self.dataset == 'sicapv2':
            raw_delta = raw_delta - raw_delta.mean(dim=-1, keepdim=True)
        else:
            raw_delta = raw_delta.squeeze(-1)
        delta = self.max_delta_logit * torch.tanh(raw_delta / self.max_delta_logit)
        if self.gate_beta is None:
            gate = torch.ones(delta.shape[0], device=delta.device, dtype=delta.dtype)
        else:
            if self.dataset == 'sicapv2':
                probability = torch.softmax(rgb_logits, dim=-1)
                entropy = -(probability * torch.log(probability.clamp_min(1e-08))).sum(-1)
                magnitude = torch.linalg.vector_norm(delta, dim=-1)
            else:
                probability = torch.sigmoid(rgb_logits)
                entropy = -(probability * torch.log(probability.clamp_min(1e-08)) + (1 - probability) * torch.log((1 - probability).clamp_min(1e-08)))
                magnitude = delta.abs()
            gate = torch.sigmoid(self.gate_beta[0] + self.gate_beta[1] * entropy + self.gate_beta[2] * magnitude)
        correction = delta * (gate.unsqueeze(-1) if delta.ndim == 2 else gate)
        return {'logits': rgb_logits + correction, 'delta': delta, 'gate': gate, 'embedding': embedding}
