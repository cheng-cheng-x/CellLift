# Environment and entry points

Install the package in an environment suitable for the selected analysis:

```sh
python -m pip install -e .
```

GPU reconstruction requires CUDA-enabled PyTorch and Triton on Linux. Segmentation additionally requires Cellpose 3.1.1.3. Whole-slide preparation may require OpenSlide, libvips and a compatible registration environment. Optional dependencies are listed in `pyproject.toml`.

## Entry points

| Operation | Python module |
| --- | --- |
| Serial matching | `celllift.serial_preparation.run` |
| Input preparation | `celllift.reconstruction.scripts.prepare` |
| Reconstruction training | `celllift.reconstruction.scripts.train` |
| Reconstruction inference | `celllift.reconstruction.scripts.inference` |
| Reconstruction evaluation | `celllift.reconstruction.scripts.evaluate` |
| Geometric baseline calibration | `celllift.evaluation.geometric_baselines.scripts.calibrate` |
| Geometric baseline inference | `celllift.evaluation.geometric_baselines.scripts.predict` |
| Scene comparison | `celllift.evaluation.scene_metrics.run` |
| Patch prediction | `celllift.benchmark_prediction.run` |
| BRCA prediction | `celllift.breast_patient_prediction.routes.run` |
| Core/nucleus prediction | `celllift.core_and_nucleus_prediction.run` |
| Geometry experts | `celllift.geometry_experts.run` |
| Structure analysis | `celllift.downstream_3d_contribution.phenotype` |
| Model attribution | `celllift.downstream_3d_contribution.model` |

Invoke modules with `python -m MODULE` and the arguments defined in their parsers. Reconstruction inference and evaluation accept shard index and shard count as positional arguments. Configure resources and prepare the corresponding input artifacts before execution. The modules implement individual stages rather than a single image-to-all-results command.

The source collection has not been subjected to a fresh end-to-end execution in this distribution. Installation, data preparation and computational requirements depend on the chosen pipeline.
