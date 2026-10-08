# CellLift

**CellLift infers three-dimensional cell morphology and spatial organisation from single H&E sections.**

CellLift learns nuclear and cell geometry from corresponding observations in adjacent histological sections. At inference, target-section contours, image features and neighbourhood information generate candidate three-dimensional bodies. Candidate scoring and contact-aware scene selection produce nuclear and cell representations in a shared coordinate system.

This repository contains the computational methods for reconstruction, geometric baseline comparisons, downstream prediction, image–geometry fusion and structural interpretation.

## Repository organisation

| Study component | Implementation |
| --- | --- |
| Registration, segmentation and adjacent-section correspondence | `registration`, `segmentation`, `serial_preparation`, `cross_layer_matching` |
| Target-section inputs and graph construction | `model_inputs`, `graph_input_schema` |
| CellLift geometry, training and inference | `reconstruction` |
| Cylindrical, ellipsoidal and quartic baselines; joint scene metrics | `evaluation` |
| Geometry prediction and image–geometry fusion | `benchmark_prediction`, `geometry_baselines`, `geometry_experts`, `breast_patient_prediction`, `breast_roi_baseline`, `core_and_nucleus_prediction` |
| Shared feature, residual and interaction models | `set_encoding`, `conditional_geometry`, `matched_geometry_controls`, `morphology_interaction`, `feature_fusion`, `scene_residual`, `cross_fitted_correction` |
| Structural effects, integrated gradients and neighbourhood replacement | `downstream_3d_contribution` |

Modules are under [`src/celllift`](src/celllift). The [study guide](docs/STUDY_GUIDE.md) connects these implementations to the manuscript's analyses and datasets.

## Documentation

- [Methods and source map](docs/METHODS.md)
- [Dataset and experiment guide](docs/STUDY_GUIDE.md)
- [Inputs, outputs and configuration](docs/DATA_FORMAT.md)
- [Environment and entry points](docs/USAGE.md)
- [External software](THIRD_PARTY.md)

Scientific settings are supplied with their modules under `configs/`. External files are configured through `configs/resources.json`, using [the example](configs/resources.example.json). The [resource index](docs/resource_index.json) identifies each resource's consuming modules.

This is a source-code distribution. Images, subject-level tables, feature caches, predictions and pretrained checkpoints are obtained or generated separately. Dataset-specific pipelines have distinct input contracts; the reconstruction model consumes prepared target-section graphs.

## Research protocol

Adjacent sections provide training supervision. Inference uses target-section inputs. Development and held-out evaluation are separate, and downstream analyses retain their dataset-specific grouping, model-selection and aggregation rules. Structural association, attribution and predictive improvement are evaluated through their respective protocols.
