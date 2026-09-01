# 120-second demo script

This recording plan shows the **learned-static-fusion** design: frozen Community Forensics 384 RGB evidence plus the frozen deterministic 26-value signal expert at **0.677 RGB / 0.323 signal**. Issue #10's command was independently reviewed and accepted at commit `b8982dfb3400fa92fde65cc0ea6f2fe141a4b402`. Quote a metric or show a result only after the tracked evidence receipts are claim-ledger-complete and human-reviewed.

| Time | Screen / narration |
| --- | --- |
| 0:00–0:15 (15s) | Show a clean source image and its declared real-world corruption variants. Say: “Authenticity evidence often weakens after delivery transformations. Our harness keeps every variant attached to its source image and its split.” |
| 0:15–0:35 (20s) | Show `README.md`, `config/community-forensics-models.json`, and the trusted bundle receipt bound to generation `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181`. Say: “The selected design is learned static fusion: frozen Community Forensics 384 and a frozen deterministic 26-value signal expert, combined at 0.677 RGB and 0.323 signal. The allocation is fixed, not a per-image degradation gate.” |
| 0:35–1:00 (25s) | Run `python submission_inference_cli.py --image-dir <directory> --bundle-dir models/track5 --rgb-checkpoint <community-forensics-384.safetensors> --signal-model models/track5/signal-model.json --output predictions.json --device auto --batch-size 8`, then show `predictions.json`. Say: “Both expert artifacts are verified before scoring. Each sorted record has only `image_path` and a finite fusion probability `pred`; it is not a provenance verdict.” Do not substitute `rgb_cli.py`. |
| 1:00–1:25 (25s) | Show `docs/submission/results/clean-vs-transformed.svg` beside `docs/submission/results/robustness-and-errors.md`. Narrate the six declared corruption families, keep “internal validation” visible, and state that the report is not an official organizer score. Use only the values bound in the claim ledger. |
| 1:25–1:45 (20s) | Show the report's claim-ledger-complete sanitized cases covering clean/transformed false positives and false negatives, with corruption-family labels and the provisional-threshold trade-off. Say: “On internal validation, signal evidence corrected 768 of 1,218 calibrated-RGB errors; fusion's macro-AUROC gain was 0.016795535714285936 with source-bootstrap interval 0.011076105794972707 to 0.0234869800759804. This is descriptive, not causal. Static fusion does not adapt trust per image.” |
| 1:45–2:00 (15s) | Show the limitations section, claim ledger, attributions, and repository URL. Say: “Organizer data is evaluation-only. Every published result needs a checksum, revision, candidate binding, and human-confirmed contribution record.” |

## Recording checklist

- Activate a local environment; do not show machine-specific paths, credentials, downloaded dataset paths, or organizer labels.
- Verify both expert artifacts, bundle, accepted command revision, and output SHA-256 values before recording a result screen.
- Use only repository-created fixtures or artifact paths named in the claim ledger.
- Keep “internal validation” visible with any result; omit a result screen if its provenance record is incomplete.
