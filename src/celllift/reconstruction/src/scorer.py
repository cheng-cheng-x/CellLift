from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch
from torch import nn
import torch.nn.functional as F

class Scorer(nn.Module):

    def __init__(self, node_dim, candidate_dim, normalization):
        super().__init__()
        for key, value in normalization.items():
            self.register_buffer(key, value)
        self.node = nn.Sequential(nn.Linear(node_dim, 128), nn.GELU(), nn.LayerNorm(128))
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 128)) for _ in range(2)])
        self.query = nn.Sequential(nn.Linear(candidate_dim, 128), nn.GELU(), nn.Linear(128, 128))
        self.head = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, graph):
        h = self.node((graph['node'] - self.node_mean) / self.node_std)
        i, j = graph['edge_index']
        degree = torch.bincount(j, minlength=len(h)).to(h.dtype).clamp_min(1)[:, None]
        for message in self.messages:
            neighbor = torch.zeros_like(h).index_add(0, j, h[i]) / degree
            h = h + message(torch.cat((h, neighbor), -1))
        q = self.query((graph['candidate'] - self.candidate_mean) / self.candidate_std)
        return self.head(torch.cat((h[:, None, :].expand_as(q), q), -1)).squeeze(-1)

def learning_terms(score, observed_cost, observed, valid, mode):
    mask = observed & valid
    target = torch.softmax(-observed_cost, dim=-1)
    if mode == 'compatibility':
        logp = F.log_softmax(score, dim=-1)
        training = -(target * logp).sum(-1)
        predicted_cost = -logp
    else:
        centered = score - score.mean(-1, keepdim=True)
        truth = observed_cost - observed_cost.mean(-1, keepdim=True)
        training = 0.5 * (centered - truth).square().mean(-1)
        logp = F.log_softmax(-score, dim=-1)
        predicted_cost = score
    kl = (target * (target.clamp_min(1e-30).log() - logp)).sum(-1)
    label = predicted_cost.argmin(-1)
    regret = observed_cost.gather(1, label[:, None])[:, 0] - observed_cost.min(-1).values
    return (training * mask, kl * mask, regret * mask, mask, predicted_cost)
