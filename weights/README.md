# Trained CellLift reconstruction model

[`reconstruction/final.pt`](reconstruction/final.pt) contains the validation-selected CellLift reconstruction model used for held-out evaluation. It includes both the geometry network and the candidate scorer. The checkpoint contains tensors and standard Python containers and can be loaded with `weights_only=True`.

| File | Contents |
| --- | --- |
| `reconstruction/final.pt` | Model parameters, scorer normalization, input normalization and configuration |
| `reconstruction/inference_config.json` | Input normalization and model configuration in JSON format |
| `reconstruction/metadata.json` | Architecture, checkpoint size and SHA-256 checksum |

## Load the model

Use the reconstruction environment described in [USAGE](../docs/USAGE.md), including PyTorch and the CUDA/Triton dependencies required by the implementation.

```python
import torch
from celllift.reconstruction.src.model import ConditionalSceneNetwork

checkpoint = torch.load(
    "weights/reconstruction/final.pt",
    map_location="cpu",
    weights_only=True,
)
model = ConditionalSceneNetwork(
    checkpoint["normalization"], checkpoint["config"]
)
model.attach_scorer(checkpoint["scorer_normalization"])
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()
```

The inference script reads `reconstruction/03_training/final.pt` beneath the configured output root. Place this checkpoint there and supply the prepared graph-cache manifest under `reconstruction/02_inputs/manifest.json`. That manifest must use the `full_stats` and `config` fields supplied in `inference_config.json`; select the intended evaluation split explicitly. See [DATA_FORMAT](../docs/DATA_FORMAT.md) for graph and supervision schemas.

The model takes target-section graphs with aligned nuclear contours, positions and 384-dimensional image features. The weights do not replace the segmentation or image-feature extraction steps. Downstream task models and external pretrained encoders have their own artifacts and input contracts.

## Integrity

SHA-256 of `reconstruction/final.pt`:

```text
c3de9f9eb17ebad942207c73117e4d75fab3b8e76fef7aec3087bca4947b4697
```

The checkpoint is distributed under the repository's [MIT License](../LICENSE).
