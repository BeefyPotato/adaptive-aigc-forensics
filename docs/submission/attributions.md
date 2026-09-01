# Attributions and upstream provenance

This page records dependency and asset provenance for the submission package. It does not select a license for this project's code; project-license choice requires owner approval.

| Asset or dependency | Project use | Recorded upstream license | Provenance / reference |
| --- | --- | --- | --- |
| SID_Set | Controlled development source pool | CC BY 4.0 (dataset card) | [Hugging Face dataset](https://huggingface.co/datasets/saberzl/SID_Set); pinned revision is recorded in `metadata/sid-set-candidate-pool-v1.json`. |
| Community Forensics / OwensLab checkpoints | Frozen RGB expert | MIT | [Community Forensics](https://github.com/JeongsooP/Community-Forensics) and the pinned repos/revisions/checksums in `config/community-forensics-models.json`. |
| PyTorch / torchvision | Tensor computation and image transforms | BSD-3-Clause | [PyTorch license](https://github.com/pytorch/pytorch/blob/main/LICENSE). |
| timm | Community Forensics model construction | Apache-2.0 | [pytorch-image-models license](https://github.com/huggingface/pytorch-image-models/blob/main/LICENSE). |
| NumPy | Numerical operations | BSD-3-Clause | [NumPy license](https://github.com/numpy/numpy/blob/main/LICENSE.txt). |
| Pillow | Image decoding and conversion | HPND | [Pillow license](https://github.com/python-pillow/Pillow/blob/main/LICENSE). |
| safetensors | Checkpoint deserialization | Apache-2.0 | [safetensors license](https://github.com/huggingface/safetensors/blob/main/LICENSE). |
| Hugging Face tooling (`huggingface-hub`) | Checkpoint retrieval | Apache-2.0 | [huggingface_hub license](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE). |
| Node.js | Corruption-harness runtime | MIT | [Node.js license](https://github.com/nodejs/node/blob/main/LICENSE). |
| Sharp | Image transformations / materialization | Apache-2.0 | [Sharp license](https://github.com/lovell/sharp/blob/main/LICENSE). |

## Provenance limitations

The Community Forensics public model metadata identifies upstream collections but does not supply an image-level provenance ledger proving that every organizer demonstration image was excluded. This is the limitation recorded by [ADR 0001](../adr/0001-use-community-forensics-checkpoint-with-provenance-controls.md). The organizer demonstration set is therefore evaluation-only; overlap checks apply to all locally controlled training sources.

SID_Set metadata and individual source records can carry upstream provenance that is more specific than the dataset card. Preserve those declarations in the manifest; do not assume the dataset-card license resolves every downstream asset's rights. Follow the current host terms before downloading data or checkpoints.
