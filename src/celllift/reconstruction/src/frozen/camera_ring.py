from __future__ import annotations
from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import math
from celllift.runtime import torch
from torch import Tensor, nn
from torch.nn import functional as F
RAY_COUNT = 36
RING_CHANNELS = 8

def _circular_conv(cin: int, cout: int, kernel: int, *, groups: int=1) -> nn.Conv1d:
    return nn.Conv1d(cin, cout, kernel, padding=kernel // 2, padding_mode='circular', groups=groups)

class DirectionalMessage(nn.Module):

    def __init__(self, channels: int=RING_CHANNELS) -> None:
        super().__init__()
        self.message = nn.Sequential(nn.Conv1d(2 * channels + 5, 2 * channels, 1), nn.GELU(), _circular_conv(2 * channels, 2 * channels, 5, groups=2 * channels), nn.GELU(), nn.Conv1d(2 * channels, channels, 1))
        self.update = nn.Sequential(nn.Conv1d(2 * channels, channels, 1), nn.GELU(), _circular_conv(channels, channels, 5, groups=channels))

    def forward(self, hidden: Tensor, edge_index: Tensor, position: Tensor) -> Tensor:
        aggregate = torch.zeros_like(hidden)
        degree = hidden.new_zeros((hidden.shape[0], 1, 1))
        if edge_index.numel():
            sender, receiver = edge_index.long()
            messages = self.message(torch.cat((hidden[receiver], hidden[sender], position), dim=1))
            aggregate.index_add_(0, receiver, messages)
            degree.index_add_(0, receiver, hidden.new_ones((receiver.numel(), 1, 1)))
        aggregate = aggregate / degree.clamp_min(1.0)
        return hidden + self.update(torch.cat((hidden, aggregate), dim=1))

class CameraRingMPNN(nn.Module):

    def __init__(self, ray_mean_um: float, ray_std_um: float, radius_um: float=60.0) -> None:
        super().__init__()
        if ray_std_um <= 0 or radius_um <= 0:
            raise ValueError('ray_std_um and radius_um must be positive')
        self.register_buffer('ray_mean_um', torch.tensor(float(ray_mean_um)))
        self.register_buffer('ray_std_um', torch.tensor(float(ray_std_um)))
        self.radius_um = float(radius_um)
        self.input = _circular_conv(1, RING_CHANNELS, 5)
        self.blocks = nn.ModuleList([DirectionalMessage() for _ in range(4)])
        theta = torch.arange(RAY_COUNT) * (2.0 * math.pi / RAY_COUNT)
        self.register_buffer('ray_cos', theta.cos())
        self.register_buffer('ray_sin', theta.sin())
        self.output = nn.Sequential(nn.Linear(RING_CHANNELS * RAY_COUNT, 128), nn.LayerNorm(128), nn.GELU())

    def forward(self, rays_um: Tensor, xy_um: Tensor, edge_index: Tensor) -> Tensor:
        rays = (rays_um.float() - self.ray_mean_um) / self.ray_std_um
        hidden = F.gelu(self.input(rays[:, None]))
        if edge_index.numel():
            sender, receiver = edge_index.long()
            vector = (xy_um.float()[sender] - xy_um.float()[receiver]) / self.radius_um
        else:
            vector = xy_um.new_empty((0, 2), dtype=torch.float32)
        distance = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        scalar = torch.cat((vector, distance), -1)[..., None].expand(-1, -1, RAY_COUNT)
        angles = torch.stack((self.ray_cos, self.ray_sin))[None].expand(vector.shape[0], -1, -1)
        position = torch.cat((scalar, angles), 1)
        for block in self.blocks:
            hidden = block(hidden, edge_index, position)
        return self.output(hidden.flatten(1))
