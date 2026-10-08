from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
from celllift.runtime import torch

def student_t(x):
    return 2.5 * torch.log1p((x / 0.46).square() / 4)
