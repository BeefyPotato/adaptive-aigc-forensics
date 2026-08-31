# 120-second demo script

This recording plan shows the raw RGB-only candidate. Do not substitute a fusion output, quote a metric, or show a result artifact until its claim-ledger row is completed and human-reviewed.

| Time | Screen / narration |
| --- | --- |
| 0:00–0:15 (15s) | Show a clean source image and its declared real-world corruption variants. Say: “Authenticity evidence often weakens after delivery transformations. Our harness keeps every variant attached to its source image and its split.” |
| 0:15–0:35 (20s) | Show `README.md` architecture and `config/community-forensics-models.json`. Say: “The current candidate is raw RGB-only: frozen Community Forensics 384. The deterministic 26-value signal representation and fusion work are research paths, not this submitted inference candidate.” |
| 0:35–1:00 (25s) | Run `python rgb_cli.py --input-dir ./images --output ./predictions.json --resolution 384 --device auto --batch-size 8`, then show `predictions.json`. Say: “The checkpoint is verified before loading. Each sorted record has only `image_path` and a probability `pred`; it is not a calibrated decision or a provenance verdict.” |
| 1:00–1:25 (25s) | Show the locally repository-created internal-validation robustness SVG only after its claim-ledger row lists candidate name, generation revision, manifest/checkpoint/output SHA-256 values. Narrate the six declared corruption families and say “internal validation” on screen. If that SVG has not been generated and checksummed, show `rgb-internal-validation-metrics.json` schema instead and make no numerical claim. |
| 1:25–1:45 (20s) | Show two claim-ledger-complete and human-reviewed internal-validation FP/FN records with source IDs redacted as necessary, their corruption-family labels, and a provisional threshold note. Say: “We inspect errors only on source-disjoint internal validation. The trade-off is that raw RGB inference is reproducible but does not adapt trust by degradation.” |
| 1:45–2:00 (15s) | Show the limitations section, claim ledger, attributions, and repository URL. Say: “Organizer data is evaluation-only. Every published result needs a checksum, revision, candidate binding, and human-confirmed contribution record.” |

## Recording checklist

- Activate a local environment; do not show machine-specific paths, credentials, downloaded dataset paths, or organizer labels.
- Verify the checkpoint and output SHA-256 values before recording a result screen.
- Use only repository-created fixtures or artifact paths named in the claim ledger.
- Keep “internal validation” visible with any result; omit a result screen if its provenance record is incomplete.
