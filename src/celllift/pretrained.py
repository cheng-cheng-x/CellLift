"""Standard torchvision model initialization."""

def resnet18(*, weights=None, progress=True, **kwargs):
    from torchvision.models import resnet18 as build
    return build(weights=weights, progress=progress, **kwargs)
