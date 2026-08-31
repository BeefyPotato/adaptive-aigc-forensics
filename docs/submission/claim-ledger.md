# Public claim ledger

This ledger is the release gate for public claims. A row is publishable only when a human reviewer confirms it. Never fill a checksum, result, team member, or contribution by inference. All metrics must be labeled **internal validation** unless a separate, evaluation-only organizer result states its scope.

## Candidate identity

| Field | Current record |
| --- | --- |
| Candidate | Raw RGB-only, frozen Community Forensics 384 |
| Candidate source | `rgb_cli.py`; `rgb_expert.py` |
| Candidate revision | `6fbfb1d` base revision; replace with the exact reviewed submission commit before publication |
| Trusted generation | `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181` |
| Trusted bundle revision | `static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179` |
| Checkpoint source | `config/community-forensics-models.json`, `models.384` |
| Checkpoint SHA-256 | `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387` |
| Bundle SHA-256 | `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2` |
| Canonical command | `python rgb_cli.py --input-dir ./images --output ./predictions.json --resolution 384 --device auto --batch-size 8` |
| Direct-output schema | JSON array of exactly `{ "image_path": string, "pred": finite number in [0, 1] }` |
| Repeated-output SHA-256 | Required before publishing an inference result; not generated or claimed in this repository |

## Claim records

| Claim | Scope and candidate binding | Evidence source / JSON path | Generation revision | Bundle SHA-256 | Required result checksums | Public destination | Publication state |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Current candidate uses the frozen 384-pixel RGB expert | Candidate: raw RGB-only; not a fusion claim. The trusted static-fallback provenance record does not make fusion an inference or result claim. | `config/community-forensics-models.json` → `models.384`; `rgb_cli.py`; trusted bundle record | `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181` | `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2` (`static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179`) | Checkpoint SHA-256 above; result-specific output SHA-256 remains required before publication | README, Devpost | Documented; human review still required |
| Signal path is deterministic 26-value representation plus frozen MLP | Research architecture only; not the current candidate | `docs/signal-expert.md`; `signal_cli.py` | `6fbfb1d` base; replace on release | Required if a signal bundle is cited | Manifest, checkpoint, and output SHA-256 required | Devpost architecture | Documented, no result claimed |
| A robustness metric or FP/FN analysis | **Internal validation** only; candidate must be named | `artifacts/rgb-baseline/rgb-internal-validation-metrics.json` (local, untracked) | Fill exact generated revision | Fill exact bundle SHA-256 or N/A direct-run rationale | Materialized manifest, checkpoint, report, and repeated-output SHA-256 | Devpost/demo only after review | No public result currently claimed |
| Organizer demonstration observation | Organizer demonstration only; never model-selection evidence | Organizer-supplied evaluation artifact and overlap audit | Fill exact generated revision | Fill exact bundle SHA-256 | Organizer manifest, checkpoint, report, and output SHA-256 | Separate labeled result only | Not available / not claimed |

## Human contribution record — required before publication

| Person (human-confirmed) | Contribution and reviewed artifact | Approval date | Confirmation source |
| --- | --- | --- | --- |
| _Unassigned — do not infer_ | _Describe a specific reviewed contribution_ | _YYYY-MM-DD_ | _Human confirmation_ |

## Release checklist

- [ ] Replace base revision placeholders with the exact reviewed commit.
- [ ] Record every generated result's candidate, scope, artifact JSON path, generation revision, bundle/manifest/checkpoint/output SHA-256 values, and repeated-output SHA-256.
- [ ] Mark each metric **internal validation** or organizer demonstration as applicable.
- [ ] Confirm all team-contribution rows with the people named.
- [ ] Obtain owner approval before adding any project code license.
