# Inputs, outputs and configuration

## Resource configuration

Copy `configs/resources.example.json` to `configs/resources.json`. Each resource entry is a user-supplied file or directory. Relative paths resolve against the configuration directory. Set `CELLLIFT_CONFIG` to use an alternative configuration file. No resource is resolved through an original workstation path.

`docs/resource_index.json` lists the source modules that consume each resource. Module-level JSON and YAML files contain scientific parameters and `resource://` references. Empty bindings are intentionally unspecified and must be supplied for the selected analysis.

## Reconstruction graphs

A graph cache contains `manifest.json` with an `items` list. Items identify the split, graph identifier, graph file and supervision file. A graph payload contains `graph`, `input` and `metadata`.

- Nuclear identifiers align between graph fields and `input.ids`.
- Image embeddings have one 384-dimensional vector per object.
- Shape coefficients and candidate queries accompany the target-section features.
- Coordinates and physical geometry use micrometres.
- Nuclear and cell observations are stored separately from inference inputs.

The exact tensor schema is defined by `NucleusGraphBatch`, `Graph`, `Coefficients` and `Supervision` in the reconstruction source. Registered correspondence records and LMDB stores are handled by the upstream adapters.

## Model artifacts

Reconstruction checkpoints contain model parameters and the normalization needed by the encoder and scorer. Downstream models use task-specific scaling and aggregation metadata. Checkpoint architecture and preprocessing must agree with the selected implementation. Pickle-based scientific artifacts should be loaded only from trusted sources.

## Outputs

Reconstruction writes predicted nuclear/cell bodies, candidate scores, selected labels and graph-level metadata. Evaluation produces per-object or per-graph records, followed by grouped summaries. Downstream prediction produces probabilities and task metrics; structural analyses produce descriptor tables and adjusted effects; attribution produces model-input contributions and replacement comparisons.

The final CellLift reconstruction checkpoint and its normalization are included under `weights/reconstruction/`; see [the loading instructions](../weights/README.md). Raw images, sample inventories, prepared caches, downstream task checkpoints and third-party pretrained models are obtained separately. Obtain each dataset through its provider and preserve its access and reuse terms.
