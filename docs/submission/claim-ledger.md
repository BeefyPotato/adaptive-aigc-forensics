# Public claim ledger

This ledger is the release gate for public claims. A row is publishable only when a human reviewer confirms it. Never fill a checksum, result, team member, or contribution by inference. All metrics must be labeled **internal validation** unless a separate, evaluation-only organizer result states its scope.

## Candidate identity

| Field | Current record |
| --- | --- |
| Candidate | `learned-static-fusion`: frozen Community Forensics 384 RGB expert plus frozen deterministic 26-value signal expert |
| Design sources | `fusion_pipeline.py`; `rgb_expert.py`; `signal_expert.py`; accepted `submission_inference.py` / `submission_inference_cli.py` at independently reviewed Issue #10 commit `b8982dfb3400fa92fde65cc0ea6f2fe141a4b402` |
| Trusted generation | `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181` |
| Trusted bundle revision | `static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179` |
| Bundle SHA-256 | `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2` |
| Deployment completion receipt | `models/track5/static-fallback.complete.json` SHA-256 `8295d00d0275ee0c06423cd1c31d96e1a16671da21dae1a20aa4dda93ea94112` |
| Manifest SHA-256 | `c9ea2d3b616b37844d21602a95e5f90c824a692ad609d31bbe0b982c5f45228a` |
| RGB checkpoint | revision `6076002bf0d9dd37537f965ee2f06f826c333b61`; SHA-256 `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387` |
| RGB upstream selection evidence | The [Community Forensics paper](https://arxiv.org/html/2411.04125v2) reports end-to-end training on a class-balanced 5.4-million-image corpus: 2.7 million generated images from 4,803 generator models and 2.7 million real images. The [official dataset card](https://huggingface.co/datasets/OwensLab/CommunityForensics) records the generated-image scale and diversity rationale. These are upstream selection findings, not this repository's performance results. |
| Signal checkpoint | profile `hackathon-v1`; revision `signal-checkpoint-v1-4a6b4d974722c9f8729a90d872387bb49e54d01e7bda98ddb69232c28604390e`; normalization revision `signal-normalization-v1-25b16b78f7ecb5e02572e03650537e8b5e266f2f3e49a911a2ae2e2e11d45e80`; model SHA-256 `cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5`; 8,064 training draws; 400 validation sources / 8,000 validation observations |
| Model scale | RGB: 21,811,969 parameters (`config/community-forensics-models.json` → `models.384.parameter_count`); signal MLP: 26→16→1 with 449 trainable scalars, decomposed as 416 + 16 + 16 + 1 (trusted signal model → `weights.input`, `weights.input_bias`, `weights.output`, `weights.output_bias`) |
| Static weight | 0.677 RGB / 0.323 signal; revision `static-fusion-weight-v1-96456ffa07a98fc81ceef01f4cbae62a52b0c07fc1e71c4f5a06b5a06eef1c1b` |
| Final accepted command | `python submission_inference_cli.py --image-dir <directory> --bundle-dir models/track5 --rgb-checkpoint <community-forensics-384.safetensors> --signal-model models/track5/signal-model.json --output predictions.json --device auto --batch-size 8`; independently reviewed and accepted at Issue #10 commit `b8982dfb3400fa92fde65cc0ea6f2fe141a4b402`; `rgb_cli.py` is not a substitute |
| Direct-output schema | JSON array whose records contain exactly `image_path` and `pred`; `pred` is finite and in `[0, 1]` |
| Evidence generation revision | `submission-evidence-generation-v1-b018d8f0326f8a9ed9945b52eda2dadeae659b3827268af69c38aa4c09e27cc1` |
| Report generation revision | `submission-report-generation-v1-411d7380b4667552401f4f751472836d7d3186854f50694dff5039ce0c19e796` |
| Tracked evidence | `docs/submission/evidence/submission-evidence.json` SHA-256 `0c3ed99c9805a4d455502f446637f33d784f541536bb1445b75ef83c6c767f90`; `docs/submission/evidence/submission-evidence.complete.json` SHA-256 `5152f58ad323cb4d4afc57dac8f209c86a7ba56a95bbf7466bb2dbc2589a4c36` |
| Tracked report | `docs/submission/results/robustness-and-errors.md` SHA-256 `9ab6378752417637d6ae3e24443c5c49aff498fbdae11b01f82a1267bf6f486b`; `docs/submission/results/clean-vs-transformed.svg` SHA-256 `8163495995381f52fbccfd754cdb4c3aecfdd918949ee3d5fca0ad6c6fdef3f6`; `docs/submission/results/submission-report.complete.json` SHA-256 `d84773233606bb6f32f3fd6d226155a80d1113ee87c166f3c34f1382190f3072` |
| Runtime acceptance | `docs/submission/runtime-smoke.json`; independently reviewed and accepted same-device record for Issue #10 commit `b8982dfb3400fa92fde65cc0ea6f2fe141a4b402` |
| Repeated-output SHA-256 | Fixture explicit CPU and `auto`: `adcd0528bd98130421385fd7d579ea8ba4ae6aa773f1c4b6e90504a2c749c1b3`; sampled-real SID_Set internal-validation CPU repeat: `21bbd744c94927e674bd9f40b3f56c9ac3188580b49b2d32869cb576e65dd2c2`; both byte-identical with maximum parity delta `0` |

## Claim records

| Claim | Scope and candidate binding | Evidence source / JSON path | Generation revision | Bundle SHA-256 | Required result checksums | Public destination | Publication state |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Selected design uses learned static fusion | Candidate: `learned-static-fusion`; static means the allocation does not change per image | Trusted static-fallback bundle → `static_weight`; `fusion_pipeline.py` | `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181` | `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2` | Static-weight revision, both expert bindings, accepted CLI commit, and runtime acceptance record above | README, Devpost | Architecture and accepted CLI documented; human publication review still required |
| RGB expert selection rationale | Upstream training and evaluation context for the frozen Community Forensics ViT-S/16 384 checkpoint; not an internal-validation result or guarantee on unseen data | [Community Forensics paper](https://arxiv.org/html/2411.04125v2); [official dataset card](https://huggingface.co/datasets/OwensLab/CommunityForensics); [official evaluation card](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval); `config/community-forensics-models.json`; `rgb_expert.py` | N/A — upstream evidence; checkpoint revision is recorded above | N/A — upstream evidence; deployed checkpoint SHA-256 is recorded above | Checkpoint revision and SHA-256, exact local parameter count, upstream paper and cards | README, attribution record | Documented as upstream selection context with provenance caveat |
| Signal expert is a deterministic 26-value representation plus frozen MLP | Core expert in `learned-static-fusion`, not a second RGB backbone | `docs/signal-expert.md`; `signal_expert.py`; trusted bundle provenance | Trusted signal checkpoint revision above | Trusted bundle SHA-256 above | Signal-model SHA-256 and normalization/representation revisions | README, Devpost | Architecture documented; no standalone signal result claimed |
| Robustness metrics and FP/FN analysis | **Internal validation** only; candidate is `learned-static-fusion` | `docs/submission/evidence/submission-evidence.json`; `docs/submission/evidence/submission-evidence.complete.json`; `docs/submission/results/robustness-and-errors.md`; `docs/submission/results/clean-vs-transformed.svg`; `docs/submission/results/submission-report.complete.json` | Evidence/report generation revisions above | Trusted bundle SHA-256 above | Manifest, both experts, evidence, report, and receipt hashes recorded above | README, Devpost, demo | Generated and checksummed; human review still required |
| Organizer demonstration observation | Organizer demonstration only; never model-selection evidence | Organizer-supplied evaluation artifact and overlap audit | N/A — no organizer evaluation generated | N/A — no organizer result | Organizer manifest, both experts, report, and output SHA-256 would be required | Separate labeled result only | Not available / not claimed |

## Metric evidence paths

All performance rows below are **internal validation** and descriptive, not causal. Dot-separated fields are exact JSON paths. The complementarity aggregates are a trusted-bundle-only limitation: they are not copied into the frozen public submission evidence JSON, so they remain bound to the trusted Issue #7 bundle SHA-256 and its exact paths rather than altering the frozen evidence.

| Public value | Evidence JSON and exact path |
| --- | --- |
| Clean AUROC `0.981975` | `docs/submission/evidence/submission-evidence.json` → `metrics.clean_auroc` |
| JPEG family AUROC `0.965512` (six-decimal display of `0.9655125`) | `docs/submission/evidence/submission-evidence.json` → `metrics.corruption_families.jpeg.auroc` |
| Blur family AUROC `0.975375` | `docs/submission/evidence/submission-evidence.json` → `metrics.corruption_families.blur.auroc` |
| Resize family AUROC `0.962363` (six-decimal display of `0.9623625`) | `docs/submission/evidence/submission-evidence.json` → `metrics.corruption_families.resize.auroc` |
| Noise family AUROC `0.883883` (six-decimal display of `0.8838833333333334`) | `docs/submission/evidence/submission-evidence.json` → `metrics.corruption_families.noise.auroc` |
| Color family AUROC `0.975696` (six-decimal display of `0.9756958333333333`) | `docs/submission/evidence/submission-evidence.json` → `metrics.corruption_families.color.auroc` |
| Crop family AUROC `0.977675` | `docs/submission/evidence/submission-evidence.json` → `metrics.corruption_families.crop.auroc` |
| Mean transformed AUROC `0.9567506944444445` | `docs/submission/evidence/submission-evidence.json` → `metrics.mean_corrupted_auroc` |
| All-condition macro AUROC `0.9603541666666667` | `docs/submission/evidence/submission-evidence.json` → `metrics.all_condition_macro_auroc` |
| Worst condition `noise` / `sigma-0.1` / AUROC `0.810425` | `docs/submission/evidence/submission-evidence.json` → `metrics.worst_family_severity.family`, `metrics.worst_family_severity.severity`, `metrics.worst_family_severity.auroc` |
| Brier score `0.09982875909583232` | `docs/submission/evidence/submission-evidence.json` → `metrics.brier_score` |
| Condition-balanced Brier score `0.09680062702417022` | `docs/submission/evidence/submission-evidence.json` → `metrics.condition_balanced_brier_score` |
| Provisional-threshold balanced accuracy `0.87625` | `docs/submission/evidence/submission-evidence.json` → `metrics.threshold_diagnostics.balanced_accuracy` |
| Provisional-threshold FPR `0.08399999999999996` | `docs/submission/evidence/submission-evidence.json` → `metrics.threshold_diagnostics.false_positive_rate` |
| Provisional-threshold FNR `0.16349999999999998` | `docs/submission/evidence/submission-evidence.json` → `metrics.threshold_diagnostics.false_negative_rate` |
| Signal corrected `768/1218 = 0.6305418719211823` calibrated-RGB errors | Trusted Issue #7 `static-fallback-bundle.json` (bundle SHA-256 above) → `evaluation.complementary_value.rgb_errors_corrected_by_signal`, `evaluation.complementary_value.rgb_errors`, `evaluation.complementary_value.correction_rate` |
| Fusion macro-AUROC gain `0.016795535714285936` | Trusted Issue #7 `static-fallback-bundle.json` → `evaluation.selection_evidence.all_condition_macro_auroc_gain` |
| Source-bootstrap interval `[0.011076105794972707, 0.0234869800759804]` | Trusted Issue #7 `static-fallback-bundle.json` → `evaluation.source_bootstrap_all_condition_macro_auroc_gain.lower`, `evaluation.source_bootstrap_all_condition_macro_auroc_gain.upper` |
| Static allocation `0.677` RGB / `0.323` signal | Trusted Issue #7 `static-fallback-bundle.json` → `static_weight.rgb_weight`, `static_weight.signal_weight` |
| Source allocation 8,000 / 2,000 / 2,000 / 2,000 | Trusted `track5-manifest.json` (manifest SHA-256 above) → `selection.split_counts` keys for `expert-training`, `fusion-training`, `internal-validation`, and `sealed-internal-test`, summed across `class-0` and `class-1` |
| Accepted CPU profile `23.18` seconds, `0.086` images/second, `466698240` peak bytes | `docs/submission/runtime-smoke.json` → `runs.fixture.profile.wall_seconds`, `runs.fixture.profile.images_per_second`, `runs.fixture.profile.peak_working_set_bytes`; acceptance fields are `accepted_submission_cli`, `canonical_command`, `independent_review`, and `issue10_commit` |
| Repeated fixture CPU/auto output SHA-256 `adcd0528bd98130421385fd7d579ea8ba4ae6aa773f1c4b6e90504a2c749c1b3`; max delta `0` | `docs/submission/runtime-smoke.json` → `runs.fixture.cpu_output_sha256`, `runs.fixture.auto_output_sha256`, `runs.fixture.repeated_cpu_output_byte_identical`, `runs.fixture.maximum_absolute_parity_delta`, `runs.fixture.device_runs` |
| Sampled-real output SHA-256 `21bbd744c94927e674bd9f40b3f56c9ac3188580b49b2d32869cb576e65dd2c2`; max delta `0` | `docs/submission/runtime-smoke.json` → `runs.sampled_real.output_sha256`, `runs.sampled_real.repeated_cpu_output_byte_identical`, `runs.sampled_real.maximum_absolute_parity_delta`; the four ignored input byte identities are at `runs.sampled_real.inputs`, and `runs.sampled_real.organizer_data` is `false` |
| Signal training profile and frozen revisions | `docs/submission/runtime-smoke.json` → `signal_training.profile`, `signal_training.training_draws`, `signal_training.validation_sources`, `signal_training.validation_observations`, `artifact_bindings.signal_checkpoint_revision`, `artifact_bindings.signal_normalization_revision` |
| Smoke environment versions | `docs/submission/runtime-smoke.json` → `environment.operating_system`, `environment.python`, `environment.pytorch`, `environment.numpy`, `environment.pillow`, `environment.timm`, `environment.safetensors`, `environment.cuda_available` |

## Public destination gates

- Repository: https://github.com/BeefyPotato/adaptive-aigc-forensics — **HUMAN REQUIRED:** make the repository public and verify it while signed out.
- YouTube demo URL: **HUMAN REQUIRED** after recording and upload.
- Devpost project URL: **HUMAN REQUIRED** after submission publication.

## Human contribution record — required before publication

| Person (human-confirmed) | Contribution and reviewed artifact | Approval date | Confirmation source |
| --- | --- | --- | --- |
| _Unassigned — do not infer_ | _Describe a specific reviewed contribution_ | _YYYY-MM-DD_ | _Human confirmation_ |

## Release checklist

- [x] Bind the canonical CLI to the independently reviewed Issue #10 acceptance commit.
- [x] Record every generated result's candidate, scope, artifact JSON path, generation revision, bundle/manifest/checkpoint/output SHA-256 values, and repeated-output SHA-256.
- [x] Mark each metric **internal validation** or organizer demonstration as applicable.
- [ ] Make the GitHub repository public and verify it while signed out.
- [ ] Record the public YouTube and Devpost URLs.
- [ ] Confirm all team-contribution rows with the people named.
- [ ] Obtain owner approval before adding any project code license.
