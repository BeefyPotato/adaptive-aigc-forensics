# Public claim ledger

This ledger is the release gate for public claims. A row is publishable only when a human reviewer confirms it. Never fill a checksum, result, team member, or contribution by inference. All metrics must be labeled **internal validation** unless a separate, evaluation-only organizer result states its scope.

## Candidate identity

| Field | Current record |
| --- | --- |
| Candidate | `learned-static-fusion`: frozen Community Forensics 384 RGB expert plus frozen deterministic 26-value signal expert |
| Design sources | `fusion_pipeline.py`; `rgb_expert.py`; `signal_expert.py`; final accepted Issue #10 CLI revision remains pending |
| Trusted generation | `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181` |
| Trusted bundle revision | `static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179` |
| Bundle SHA-256 | `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2` |
| Manifest SHA-256 | `c9ea2d3b616b37844d21602a95e5f90c824a692ad609d31bbe0b982c5f45228a` |
| RGB checkpoint | revision `6076002bf0d9dd37537f965ee2f06f826c333b61`; SHA-256 `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387` |
| Signal checkpoint | revision `signal-checkpoint-v1-4a6b4d974722c9f8729a90d872387bb49e54d01e7bda98ddb69232c28604390e`; model SHA-256 `cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5` |
| Static weight | 0.677 RGB / 0.323 signal; revision `static-fusion-weight-v1-96456ffa07a98fc81ceef01f4cbae62a52b0c07fc1e71c4f5a06b5a06eef1c1b` |
| Final accepted command | **pending Issue #10 acceptance and final CLI binding**; `rgb_cli.py` is not a substitute for fused inference |
| Direct-output schema | JSON array whose records contain exactly `image_path` and `pred`; `pred` is finite and in `[0, 1]` |
| Evidence generation revision | `submission-evidence-generation-v1-b018d8f0326f8a9ed9945b52eda2dadeae659b3827268af69c38aa4c09e27cc1` |
| Report generation revision | `submission-report-generation-v1-411d7380b4667552401f4f751472836d7d3186854f50694dff5039ce0c19e796` |
| Tracked evidence | `docs/submission/evidence/submission-evidence.json` SHA-256 `0c3ed99c9805a4d455502f446637f33d784f541536bb1445b75ef83c6c767f90`; `docs/submission/evidence/submission-evidence.complete.json` SHA-256 `5152f58ad323cb4d4afc57dac8f209c86a7ba56a95bbf7466bb2dbc2589a4c36` |
| Tracked report | `docs/submission/results/robustness-and-errors.md` SHA-256 `9ab6378752417637d6ae3e24443c5c49aff498fbdae11b01f82a1267bf6f486b`; `docs/submission/results/clean-vs-transformed.svg` SHA-256 `8163495995381f52fbccfd754cdb4c3aecfdd918949ee3d5fca0ad6c6fdef3f6`; `docs/submission/results/submission-report.complete.json` SHA-256 `d84773233606bb6f32f3fd6d226155a80d1113ee87c166f3c34f1382190f3072` |
| Repeated-output SHA-256 | Pending the accepted command's same-device repeat run |

## Claim records

| Claim | Scope and candidate binding | Evidence source / JSON path | Generation revision | Bundle SHA-256 | Required result checksums | Public destination | Publication state |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Selected design uses learned static fusion | Candidate: `learned-static-fusion`; static means the allocation does not change per image | Trusted static-fallback bundle → `static_weight`; `fusion_pipeline.py` | `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181` | `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2` | Static-weight revision and both expert bindings above | README, Devpost | Architecture documented; final CLI acceptance pending |
| Signal expert is a deterministic 26-value representation plus frozen MLP | Core expert in `learned-static-fusion`, not a second RGB backbone | `docs/signal-expert.md`; `signal_expert.py`; trusted bundle provenance | Trusted signal checkpoint revision above | Trusted bundle SHA-256 above | Signal-model SHA-256 and normalization/representation revisions | README, Devpost | Architecture documented; no standalone signal result claimed |
| Robustness metrics and FP/FN analysis | **Internal validation** only; candidate is `learned-static-fusion` | `docs/submission/evidence/submission-evidence.json`; `docs/submission/evidence/submission-evidence.complete.json`; `docs/submission/results/robustness-and-errors.md`; `docs/submission/results/clean-vs-transformed.svg`; `docs/submission/results/submission-report.complete.json` | Evidence/report generation revisions above | Trusted bundle SHA-256 above | Manifest, both experts, evidence, report, and receipt hashes recorded above | README, Devpost, demo | Generated and checksummed; human review still required |
| Organizer demonstration observation | Organizer demonstration only; never model-selection evidence | Organizer-supplied evaluation artifact and overlap audit | N/A — no organizer evaluation generated | N/A — no organizer result | Organizer manifest, both experts, report, and output SHA-256 would be required | Separate labeled result only | Not available / not claimed |

## Human contribution record — required before publication

| Person (human-confirmed) | Contribution and reviewed artifact | Approval date | Confirmation source |
| --- | --- | --- | --- |
| _Unassigned — do not infer_ | _Describe a specific reviewed contribution_ | _YYYY-MM-DD_ | _Human confirmation_ |

## Release checklist

- [ ] Replace the pending CLI field only with an independently reviewed Issue #10 acceptance receipt.
- [ ] Record every generated result's candidate, scope, artifact JSON path, generation revision, bundle/manifest/checkpoint/output SHA-256 values, and repeated-output SHA-256.
- [ ] Mark each metric **internal validation** or organizer demonstration as applicable.
- [ ] Confirm all team-contribution rows with the people named.
- [ ] Obtain owner approval before adding any project code license.
