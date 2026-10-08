# Methods and source map

Paths below are relative to `src/celllift/`.

## Data preparation

`registration/regroi/` contains image registration and spatial correspondence operations. `segmentation/` implements nucleus and cell extraction. `serial_preparation/serial_dataset_builder/` and `cross_layer_matching/` provide object matching, confidence classification, dataset assembly and serial-section splitting. `model_inputs/` builds the target-section features used by reconstruction and downstream models.

The segmentation protocol uses Cellpose 3.1.1.3 with nuclei and cyto3 models. Reliable nuclear observations and consistent cell correspondences supply adjacent-section supervision. Nuclear observations retain their confidence weights when corresponding cell observations are unavailable.

## Three-dimensional reconstruction

| Operation | Source |
| --- | --- |
| Nuclear and conditional-cell parameterisation | `reconstruction/src/geometry.py` |
| Finite-thickness section projections and physical measurements | `reconstruction/src/projection_geometry.py` |
| Target-section encoding and nine-candidate generation | `reconstruction/src/model.py` |
| Observation and shape losses | `reconstruction/src/losses.py`, `learning.py` |
| Contact computations and candidate tables | `reconstruction/src/contact.py`, `candidate_tables.py`, `frozen/` |
| Candidate scoring | `reconstruction/src/scorer.py` |
| Joint scene selection | `reconstruction/src/selector.py` |
| Input graphs and supervision | `reconstruction/src/data.py`, `full_inputs.py`, `upstream/` |
| Training, inference and reconstruction evaluation | `reconstruction/scripts/` |

The training implementation retains the single-body, candidate-geometry and scoring stages. Physical geometry uses micrometre coordinates and a five-micrometre section thickness. The target nuclear projection constrains candidate geometry, while adjacent-section observations supervise axial shape and position.

## Reconstruction comparisons

`evaluation/geometric_baselines/` implements calibrated geometric comparators. `evaluation/scene_metrics/` implements contact, overlap and joint observation–compatibility measurements. Fixed settings include the shrinkage tolerance grid and randomized volume integration. Held-out reconstruction evaluation uses the selected model and frozen baseline calibration.

## Downstream prediction

The task implementations retain separate two-dimensional and three-dimensional geometry inputs, task-specific image reference models, pooling strategies and fusion operators. Shared geometric feature code is under `set_encoding/`, `matched_geometry_controls/` and `conditional_geometry/`; task models and their statistical procedures are listed in the study guide.

## Structure and model interpretation

`downstream_3d_contribution/extract.py` and related extraction modules construct object and neighbourhood descriptors. `phenotype.py` fits adjusted class comparisons. `model.py` and `al_explain.py` implement integrated gradients and model-use analyses; `matching.py`, `paired_relations.py` and `perturb_metrics.py` support matched replacement and grouped comparisons. These analyses retain their own models and observational units.
