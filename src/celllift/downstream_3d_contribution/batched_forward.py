from celllift.runtime import resource_path as _public_resource
import torch

def probabilities(f, inputs, **kwargs):
    batch = max(1, int(getattr(f, 'perturbation_batch_size', 1)))
    results = []
    start = 0
    with torch.no_grad():
        while start < len(inputs):
            count = min(batch, len(inputs) - start)
            x = None
            try:
                x = torch.as_tensor(inputs[start:start + count], device=f.rich.device)
                z = f(x[0], **kwargs) if count == 1 else f(x, **kwargs)
                p = torch.softmax(z, -1)
            except torch.cuda.OutOfMemoryError:
                if count == 1:
                    raise
                del x
                torch.cuda.empty_cache()
                batch = max(1, count // 2)
                continue
            results.extend(p.unbind(0))
            start += count
    return results
