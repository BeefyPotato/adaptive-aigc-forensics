# Public submission report and deferred sealed evaluation

The committed Section 5.5 report is candidate-bound development evidence for
`learned-static-fusion`. It summarizes the completed `internal-validation`
split that influenced candidate and threshold selection. It is not a sealed,
organizer, official, independent-test, or unbiased performance estimate.

The public artifacts are limited to aggregate robustness metrics and four
deterministically selected, sanitized error representatives. They contain no
image paths or bytes, raw labels, logits, or probabilities:

```text
docs/submission/evidence/submission-evidence.json
docs/submission/evidence/submission-evidence.complete.json
docs/submission/results/robustness-and-errors.md
docs/submission/results/clean-vs-transformed.svg
docs/submission/results/submission-report.complete.json
```

## Reproduce the public report

From the repository root, with the trusted Issue #7 generation available at
`artifacts/issue-7-fusion-v2`, run:

```powershell
python submission_evidence.py `
  --generation-dir artifacts/issue-7-fusion-v2 `
  --candidate learned-static-fusion `
  --expected-generation-revision static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181 `
  --expected-bundle-sha256 9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2 `
  --output-dir docs/submission/evidence

python submission_report.py `
  docs/submission/evidence `
  docs/submission/results
```

Both publishers validate exact inventories and completion bindings. Repeating
the commands against the committed output succeeds only when every generated
byte matches; stale, mutated, partial, or extra files fail closed.

## Deferred evaluation paths

Sealed internal-test and organizer-cohort evaluation remain deferred. No
sealed or organizer result is included in this submission, and neither path
contributed metrics or narrative to the public internal-validation report.
