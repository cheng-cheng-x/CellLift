# Dataset and experiment guide

## Reconstruction

Serial mouse prostate sections provide training supervision and held-out trajectories for reconstruction evaluation. Training, validation and test membership are defined at the trajectory level. The study compares CellLift with calibrated cylindrical, ellipsoidal and quartic-body baselines, using adjacent-section agreement, spatial compatibility and overlap measures.

## Downstream tasks

| Dataset | Prediction endpoint | Main modules |
| --- | --- | --- |
| SICAPv2 | Prostate patch grading | `benchmark_prediction`, `geometry_baselines` |
| BRACS | Breast ROI classification | `breast_roi_baseline`, `benchmark_prediction` |
| TCGA CRC | Patient-level microsatellite instability | `morphology_interaction`, `geometry_experts`, `benchmark_prediction` |
| TCGA BRCA | Four-class subtype, luminal A/B, ductal/lobular and ER status | `breast_patient_prediction` |
| Arvaniti prostate cores | Core grading with two reader labels | `core_and_nucleus_prediction` |
| Lizard | Nucleus classification | `core_and_nucleus_prediction`, `morphology_interaction` |

The two principal prediction comparisons are matched G2 versus G3 geometry and task-specific image references versus geometry-augmented models. BRCA uses image features and gated-attention pooling. SICAPv2 uses FSConv with pooled geometric experts. BRACS combines image and geometric predictions. CRC uses morphology–appearance interaction, and Lizard and Arvaniti use their corresponding image–node models.

Dataset-specific entry points and configurations retain multiple named arms needed for these comparisons. G2/G3 denotes the manuscript's dimensional comparison; internal arm names are defined in each task's protocol module and should be interpreted within that protocol.

## Structural analysis

Twenty object and neighbourhood properties are summarized by medians and interquartile ranges, giving forty descriptors. Adjusted comparisons account for two-dimensional geometry, object counts and valid three-dimensional coverage. Grouped uncertainty estimates follow the relevant patient or source-group unit.

## Model interpretation

Integrated-gradient contributions are normalized within explanation units and aggregated across source groups, classes, tasks and datasets. The common scalar morphology groups summarize the exported three-dimensional model inputs. Lizard neighbourhood replacement holds model weights fixed, matches on two-dimensional information and repeats the replacement procedure twenty times.

Predictive comparisons, structural effects and interpretation experiments use their respective trained models and sample definitions. The manuscript distinguishes retrospective or exploratory evaluations from validation-based model selection.
