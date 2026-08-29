# Dataset access

This page records the dataset locations listed for Track 5 in the hackathon brief, together with this project's intended use of each dataset. The links below are the hyperlink targets embedded on page 23 of *TikTok TechJam 2026 Tracks & Problem Statements*.

## Development datasets

| Dataset | Access location | Planned role |
| --- | --- | --- |
| SID_Set | [Hugging Face: saberzl/SID_Set](https://huggingface.co/datasets/saberzl/SID_Set) | Primary controlled development dataset. Use the agreed 14,000-source cap and exclude the tampered subset. |
| CIFAKE | [Kaggle: CIFAKE - Real and AI-Generated Synthetic Images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | Small compatibility and pipeline smoke check; not part of the critical training path. |
| WildFake | [ModelScope: WildFake](https://modelscope.cn/datasets/hy2628982280/WildFake/summary) | Optional cross-dataset evaluation if time permits; not required for the primary submission result. |

Follow each host's current access terms and the dataset's own license. Do not commit downloaded images, host credentials, API tokens, or machine-specific dataset paths to this repository. Record the exact dataset revision, download date, file inventory, and checksums in the experiment manifest.

## Organizer demonstration set

The brief also describes an organizer demonstration set containing:

- 4,998 non-AIGC images from COCO val2017.
- 8,843 AIGC images from DALL-E Advanced.

The attached brief does not embed a standalone download URL for this combined demonstration set. Obtain the organizer-curated copy through the official hackathon materials or organizer channel rather than silently substituting a different collection.

This set is evaluation-only: do not use it for training, calibration, model selection, threshold selection, or narrative selection. Report its results separately from the sealed internal test, and identify them as demonstration-only rather than an official competition score.

## Reproducibility rules

- Split at the source-image level before generating any corruption variants.
- Keep every clean and transformed observation from one source in the same partition.
- Run exact and perceptual duplicate checks across partitions and against the organizer demonstration set when it becomes available.
- Preserve the original archives or immutable host revision and verify them before preparing the experiment manifest.
- Keep local dataset roots configurable; never encode a teammate's absolute filesystem path in code or committed configuration.
