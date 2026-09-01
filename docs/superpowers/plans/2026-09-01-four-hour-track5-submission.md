# Four-Hour Track 5 Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all Section 5.5 submission deliverables within four wall-clock hours using the completed Issue #7 evidence, without making Issue #10 or unavailable organizer bytes a hard blocker.

**Architecture:** Use three parallel lanes with separate files: Issue #9 generates a checksummed clean-versus-transformed report and deterministic FP/FN note; Issue #10 attempts the learned-static directory predictor; Issue #11 prepares README, Devpost, attribution, and video assets. At T+2:00, freeze either the fully accepted Issue #10 fusion predictor or the already tested `rgb_cli.py` deadline path, then bind the report and narrative to that exact candidate.

**Tech Stack:** Python 3.12, `unittest`, NumPy, PyTorch, `fusion_pipeline`, `rgb_expert`, Node.js 22, Sharp 0.35.4, static Markdown/SVG, GitHub Issues, Devpost, and YouTube.

**Spec:** Section 5.5 on pages 23-24 of `[Early Bird Access] TikTok TechJam 2026 Tracks & Problem Statements - Feishu Docs.pdf`; [Issue #9](https://github.com/BeefyPotato/adaptive-aigc-forensics/issues/9), [Issue #10](https://github.com/BeefyPotato/adaptive-aigc-forensics/issues/10), and [Issue #11](https://github.com/BeefyPotato/adaptive-aigc-forensics/issues/11).

## Global Constraints

- Hard deadline: T+4:00 includes implementation, tests, merges, recording, upload, Devpost, and public-link verification.
- Required prediction JSON is an array of records containing exactly `image_path` and `pred`; `pred` is finite and in `[0, 1]`.
- Models must have fewer than 2B parameters.
- COCO val2017 / DALL-E Advanced remains evaluation-only and cannot influence training, calibration, selection, weights, thresholds, templates, or narrative.
- Internal-validation results must be labeled as development evidence, never as sealed, organizer, official, independent-test, or unbiased performance.
- Trusted generation: `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181`.
- Trusted bundle revision: `static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179`.
- Trusted bundle SHA-256: `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2`.
- Do not commit datasets, checkpoints, raw caches, credentials, machine-specific paths, or image bytes.
- Keep work on issue branches. Merge Issue #10 only if fully accepted, then Issue #9, then Issue #11. Never merge because the clock is low.
- Repository publication, YouTube upload, Devpost entry, and contribution confirmation are human actions requiring the owner.

## Section 5.5 Audit

| Deliverable | Existing evidence | Missing submission artifact |
| --- | --- | --- |
| Devpost description | Architecture/model/data facts exist across docs | `docs/submission/devpost.md` and final Devpost paste |
| Public repository | Code is structured; `rgb_cli.py` already emits exact schema | Repository is private; README is incomplete; Issue #10 fusion CLI is absent |
| Demo video | Existing fixture images and CLI | Script, recording, public YouTube upload, Devpost link |
| Robustness summary | Completed Issue #7 bundle has clean and six-family metrics | Public table/SVG |
| Error analysis | 8,000 validated rows and frozen threshold exist | Deterministic FP/FN note and trade-off summary |

## Four-Hour Schedule

| Time | Issue #9 lane | Inference lane | Packaging/human lane |
| --- | --- | --- | --- |
| 0:00-0:15 | Freeze schema/scope | Preflight artifacts/checkpoint | Confirm roles; start common copy |
| 0:15-1:15 | Evidence builder + tests | Issue #10 implementation/tests | README, Devpost, attribution, video script |
| 1:15-1:45 | Markdown/SVG renderer + tests | Full gate + real parity smoke | Rehearse intro/results |
| 1:45-2:00 | Claim review | Submit acceptance evidence | Prepare two explicit command variants |
| **2:00** | **Freeze one candidate; no late switching** |
| 2:00-2:40 | Generate real report; verify/merge #9 | Merge #10 only if fully green | Finalize candidate-specific copy |
| 2:40-3:25 | Integration support | Repeat final demo inference | Record/review/upload video |
| 3:25-3:50 | Full test/claim gate | Clean-worktree smoke | Public-tree scan, GitHub/Devpost publication |
| 3:50-4:00 | Code freeze | Code freeze | Verify public URLs and submit |

## Deadline Scope Cuts

- COCO/DALL-E acquisition/evaluation is stretch work if exact organizer-curated bytes are not already available.
- SID_Set sealed/custodian evaluation remains post-submission.
- Reuse the persisted Issue #7 source-bootstrap gain interval; do not build a new bootstrap system.
- Defer organizer cohort tooling, new full corruption materialization, dashboards, thumbnails, and dataset redistribution.
- No partial Issue #10 merge. If any current criterion is unmet at T+2:00, use `rgb_cli.py` honestly.

---

### Task 1: Freeze Scope and the Candidate Gate

**Time box:** 15 minutes.

**Files:**
- Modify through GitHub: Issue #9 and Issue #11.
- Record final decision in: `docs/submission/claim-ledger.md`.

**Interfaces:**
- Consumes: Issue #7 generation and Issue #10 acceptance evidence.
- Produces: one immutable `system_id` at T+2:00.

- [ ] **Step 1: Rescope Issue #9**

  Make the Section 5.5 clean-versus-transformed summary and deterministic FP/FN note from completed Issue #7 internal validation the acceptance target. Retain sealed code as deferred capability and move COCO/DALL-E to stretch work.

- [ ] **Step 2: Correct Issue #11**

  Require every public number to trace to a checksummed completed generation and name its split. Remove the obsolete requirement that all numbers come from a sealed evaluator.

- [ ] **Step 3: Freeze the gate rule**

  - `learned-static-fusion` is legal only when every current Issue #10 criterion and real parity smoke pass.
  - Otherwise use `raw-rgb-only` through existing `rgb_cli.py` and report the bundle's matching candidate.

- [ ] **Step 4: Record why the gate is not cherry-picking**

  Freeze at T+2:00 based only on engineering readiness. Do not switch after seeing report formatting or demo output.

---

### Task 2: Build Candidate-Bound Evidence

**Time box:** 60 minutes.

**Files:**
- Create: `submission_evidence.py`
- Create: `tests/test_submission_evidence.py`

**Interfaces:**
- Consumes: `fusion_pipeline.read_static_fallback_generation(output_directory, *, expected_provenance=None, expected_generation_revision)`, trusted revisions, and explicit candidate.
- Produces: `build_submission_evidence(generation_directory, *, candidate, expected_generation_revision, expected_bundle_sha256) -> dict` and atomic `publish_submission_evidence(generation_directory, *, candidate, expected_generation_revision, expected_bundle_sha256, output_directory) -> dict`.

- [ ] **Step 1: Write the failing trusted-reader test**

```python
@patch("submission_evidence.read_static_fallback_generation")
def test_evidence_binds_generation_bundle_and_candidate(reader):
    reader.return_value = completed_generation_fixture()
    evidence = build_submission_evidence(
        "frozen-generation",
        candidate="learned-static-fusion",
        expected_generation_revision=TRUSTED_GENERATION_REVISION,
        expected_bundle_sha256=TRUSTED_BUNDLE_SHA256,
    )
    reader.assert_called_once_with(
        "frozen-generation",
        expected_generation_revision=TRUSTED_GENERATION_REVISION,
    )
    assert evidence["schema_version"] == "submission-evidence-v1"
    assert evidence["system_id"] == "learned-static-fusion"
    assert evidence["bindings"]["bundle_sha256"] == TRUSTED_BUNDLE_SHA256
    assert evidence["evaluation_scope"] == "internal-validation"
```

- [ ] **Step 2: Run RED**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' -m unittest tests.test_submission_evidence -v
```

  Expected: `ModuleNotFoundError: No module named 'submission_evidence'`.

- [ ] **Step 3: Implement trusted loading and candidate mapping**

```python
CANDIDATE_SCORE_FIELDS = {
    "raw-rgb-only": "rgb_logit",
    "learned-static-fusion": "selected_fallback_logit",
}


def build_submission_evidence(
    generation_directory,
    *,
    candidate,
    expected_generation_revision,
    expected_bundle_sha256,
):
    if candidate not in CANDIDATE_SCORE_FIELDS:
        raise ValueError(
            "Submission candidate must be raw-rgb-only or learned-static-fusion."
        )
    generation = read_static_fallback_generation(
        generation_directory,
        expected_generation_revision=expected_generation_revision,
    )
    completion = generation["completion"]
    if completion["bundle_sha256"] != expected_bundle_sha256:
        raise ValueError("Submission evidence bundle SHA-256 is incompatible.")
    bundle = generation["bundle"]
    return _evidence_from_validated_inputs(
        completion=completion,
        bundle=bundle,
        candidate=candidate,
        metrics=bundle["evaluation"]["candidates"][candidate],
        rows=generation["calibrated_internal_validation_cache"]["records"],
    )
```

- [ ] **Step 4: Add literal extraction tests**

  For `learned-static-fusion` assert 400 sources, 8,000 observations, clean AUROC `0.981975`, mean transformed AUROC `0.9567506944444445`, macro AUROC `0.9603541666666667`, and worst condition noise / `sigma-0.1` / `0.810425`. For `raw-rgb-only` assert its own candidate object from the validated bundle.

- [ ] **Step 5: Write failing deterministic-error tests**

```python
assert set(evidence["error_analysis"]["representative_cases"]) == {
    "clean-false-positive",
    "clean-false-negative",
    "transformed-false-positive",
    "transformed-false-negative",
}
assert evidence["error_analysis"]["selection_rule"] == "submission-error-hash-rank-v1"
assert build_from_rows(rows) == build_from_rows(list(reversed(rows)))
```

- [ ] **Step 6: Implement source-unique hash ranking**

  For each wrong prediction, hash canonical JSON containing ranking version, stratum, source ID, and variant ID; keep the lowest-ranked case per source, then the lowest-ranked source per stratum. Publish only IDs, family, severity, error kind, expert agreement/correction status, and rank. Exclude image paths, bytes, labels, logits, and probabilities.

- [ ] **Step 7: Persist limitations and trade-offs**

  State that internal validation influenced candidate/threshold selection; results are not organizer, sealed, official, independent-test, or unbiased estimates; upstream checkpoint overlap cannot be disproven; and the weakest condition/FPR/FNR discussion is descriptive, not causal.

- [ ] **Step 8: Publish exact atomic inventory**

```text
submission-evidence.json
submission-evidence.complete.json
```

  Completion binds generation/bundle revisions, bundle SHA-256, `system_id`, evidence SHA-256, and `submission-evidence-generation-v1-<sha256>`. Exact reruns reuse byte-identical output; mutation or extra files fail.

- [ ] **Step 9: Run GREEN and commit**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' -m unittest tests.test_submission_evidence -v
git add submission_evidence.py tests/test_submission_evidence.py
git commit -m "feat: publish frozen submission evidence"
```

---

### Task 3: Render the Section 5.5 Report

**Time box:** 30 minutes.

**Files:**
- Create: `submission_report.py`
- Create: `tests/test_submission_report.py`

**Interfaces:**
- Consumes: completed `submission-evidence-v1` directory only.
- Produces: `render_submission_report(evidence_directory, output_directory) -> dict` and no metric recomputation.

The report output inventory is exactly:

```text
robustness-and-errors.md
clean-vs-transformed.svg
submission-report.complete.json
```

- [ ] **Step 1: Write the failing renderer test**

```python
def test_report_contains_required_summary_errors_and_disclosure(self):
    completion = render_submission_report(self.evidence_dir, self.output_dir)
    markdown = (self.output_dir / "robustness-and-errors.md").read_text("utf-8")
    svg = ElementTree.parse(self.output_dir / "clean-vs-transformed.svg")
    self.assertIn("Clean versus transformed", markdown)
    self.assertIn("False positives", markdown)
    self.assertIn("False negatives", markdown)
    self.assertIn("internal validation", markdown)
    self.assertIn("not an official organizer score", markdown)
    self.assertEqual(svg.getroot().tag.rsplit("}", 1)[-1], "svg")
    self.assertEqual(completion["system_id"], self.evidence["system_id"])
```

- [ ] **Step 2: Run RED**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' -m unittest tests.test_submission_report -v
```

- [ ] **Step 3: Implement one table and one SVG**

  Render clean, JPEG, blur, resize, noise, color, crop, and mean transformed AUROC. Also render macro AUROC, worst family/severity, Brier, balanced accuracy, FPR, and FNR. SVG bars use only persisted values.

- [ ] **Step 4: Render all four error strata**

  Show one sanitized case per available stratum; explicitly say `No case in the frozen evaluation` if empty. Add weakest-condition and threshold trade-offs without inventing visual causes.

- [ ] **Step 5: Test escaping and determinism**

  Inject HTML/XML metacharacters, render twice and compare bytes, mutate evidence, add an extra file, and require each invalid input to fail.

- [ ] **Step 6: Run GREEN and commit**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' -m unittest tests.test_submission_evidence tests.test_submission_report -v
git add submission_report.py tests/test_submission_report.py
git commit -m "feat: render submission robustness evidence"
```

---

### Task 4: Run Inference and Enforce the T+2:00 Gate

**Time box:** Parallel T+0:00 to T+2:00.

**Files on Issue #10 branch:**
- Create if pursuing fusion: `submission_inference.py`
- Create if pursuing fusion: `submission_inference_cli.py`
- Create if pursuing fusion: `tests/test_submission_inference.py`
- Create if pursuing fusion: `tests/test_submission_inference_cli.py`
- Create if pursuing fusion: `docs/submission-inference.md`
- Existing fallback: `rgb_cli.py`, `rgb_expert.py`, `tests/test_rgb_expert.py`, `docs/rgb-expert.md`

**Interfaces:**
- Fusion: `run_submission_inference(image_directory, *, frozen_generation_directory, rgb_checkpoint, signal_model, device, batch_size) -> list[dict]`.
- Either path emits deterministically sorted `[{"image_path": "relative/path.png", "pred": 0.5}]`.

- [ ] **Step 1: Preflight artifacts**

  Verify the 384 RGB checkpoint, Issue #7 generation, and signal-model SHA-256 `cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5`. Start checkpoint acquisition immediately if absent.

- [ ] **Step 2: Prove the existing deadline path**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' -m unittest tests.test_rgb_expert.RgbExpertTests.test_directory_and_experiment_paths_share_preprocessing_and_score_direction tests.test_rgb_expert.RgbExpertTests.test_checkpoint_checksum_mismatch_fails_before_model_loading -v
```

- [ ] **Step 3: Implement Issue #10 with core TDD**

  Prove calibrated `0.677 / 0.323` fusion, pre-score checksum validation, mandatory experts, recursive deterministic paths, atomic corrupt-image failure, CPU mode, repeated bytes, and evaluation-path parity.

- [ ] **Step 4: Run the complete gate at T+1:40**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' -m unittest tests.test_submission_inference tests.test_submission_inference_cli -v
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' -m unittest discover -s tests -p 'test*.py' -v
npm test
git diff --check origin/main...HEAD
git status --short --branch
```

- [ ] **Step 5: Freeze one candidate at T+2:00**

  Merge/select fusion only if all current Issue #10 criteria and real parity pass. Otherwise preserve Issue #10 unmerged and use:

```powershell
python rgb_cli.py --input-dir ./fixtures/experiment/images --output ./artifacts/demo/predictions.json --device cpu --batch-size 2 --cache-dir ./artifacts/checkpoints
```

  Record the result in the claim ledger. Never describe RGB output as fusion.

---

### Task 5: Generate the Real Public Evidence

**Time box:** 40 minutes, after the candidate gate.

**Files:**
- Generate: `docs/submission/evidence/submission-evidence.json`
- Generate: `docs/submission/evidence/submission-evidence.complete.json`
- Generate: `docs/submission/results/robustness-and-errors.md`
- Generate: `docs/submission/results/clean-vs-transformed.svg`
- Generate: `docs/submission/results/submission-report.complete.json`
- Modify: `docs/sealed-report.md` to distinguish public internal-validation evidence.

**Interfaces:**
- Consumes: final `system_id` and `artifacts/issue-7-fusion-v2`.
- Produces: public aggregates only.

- [ ] **Step 1: Generate candidate-bound evidence**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' submission_evidence.py --generation-dir '..\..\issue-7-fusion-v2' --candidate learned-static-fusion --expected-generation-revision 'static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181' --expected-bundle-sha256 '9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2' --output-dir 'docs\submission\evidence'
```

  Use `--candidate raw-rgb-only` instead when the deadline path is frozen.

- [ ] **Step 2: Render and repeat**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' submission_report.py 'docs\submission\evidence' 'docs\submission\results'
```

  Repeat under `artifacts/submission-repeat` and require matching SHA-256 files.

- [ ] **Step 3: Verify learned-static values when applicable**

```text
sources: 400
observations: 8000
clean AUROC: 0.981975
mean transformed AUROC: 0.9567506944444445
all-condition macro AUROC: 0.9603541666666667
worst condition: noise / sigma-0.1 / AUROC 0.810425
degradation retention: 0.9743126805106489
Brier score: 0.09982875909583232
balanced accuracy: 0.87625
RGB errors corrected by signal: 768 / 1218 (0.6305418719211823)
fusion macro-AUROC gain: 0.016795535714285936
source-bootstrap 95% gain interval: [0.011076105794972707, 0.0234869800759804]
```

  Do not present these as submitted-system metrics when `raw-rgb-only` is selected.

- [ ] **Step 4: Inspect SVG, test, and commit**

```powershell
& '..\..\venvs\signal-issue6-clean-20260831\Scripts\python.exe' -m unittest tests.test_submission_evidence tests.test_submission_report tests.test_fusion_pipeline -v
git diff --check
git add submission_evidence.py submission_report.py tests/test_submission_evidence.py tests/test_submission_report.py docs/submission docs/sealed-report.md
git commit -m "docs: publish Track 5 submission evidence"
```

  Visually verify unclipped labels, bars, disclosures, and worst-condition annotation.

---

### Task 6: Assemble README, Devpost, Attribution, and Video Script

**Time box:** Parallel T+0:00 to T+2:40 on Issue #11.

**Files:**
- Modify: `README.md`
- Create: `docs/submission/devpost.md`
- Create: `docs/submission/demo-script.md`
- Create: `docs/submission/attributions.md`
- Create: `docs/submission/claim-ledger.md`
- Create: `tests/test_submission_docs.py`

**Interfaces:**
- Consumes: architecture facts immediately and frozen candidate/results at T+2:00.
- Produces: all written Section 5.5 content and a claim ledger.

- [ ] **Step 1: Complete README**

  Add overview, architecture, Python/Node setup, dependency install, data preparation, corruption generation, training/calibration order, one canonical inference command, result reproduction, limitations/improvements, and contributions.

- [ ] **Step 2: Draft Devpost with exact required headings**

```markdown
# Adaptive AIGC Forensics
## Problem and solution
## Technical implementation
## Development tools
## Models and APIs
## Libraries and frameworks
## Datasets and assets
## Robustness and error analysis
## Innovation and complementary value
## Impact and feasibility
## Limitations and next steps
## Team contributions
## Demo and repository
```

  State that RGB is Community Forensics 384 and signal is a deterministic 26-value representation plus frozen MLP. Mention `0.677 / 0.323` only if fusion ships. Label every result internal validation.

- [ ] **Step 3: Add attribution**

  List SID_Set, Community Forensics/OwensLab, PyTorch, NumPy, Pillow, Hugging Face tooling, Node.js, and Sharp with recorded licenses. Include upstream provenance limitations. Do not choose a project code license without owner approval.

- [ ] **Step 4: Write a 90-120 second demo**

  Use repository-created fixtures: 15 seconds problem; 20 seconds architecture/candidate; 25 seconds command and JSON; 25 seconds robustness SVG; 20 seconds FP/FN and trade-off; 15 seconds limitations/repository.

- [ ] **Step 5: Build the claim ledger**

  Record every number/architecture claim, source file and JSON path, generation revision, bundle SHA-256, public destination, canonical command, and repeated-output SHA-256.

- [ ] **Step 6: Test documentation and commit**

  Assert README contains one canonical command, exact output schema, setup, reproduction, limitations, contributions, candidate, and evidence link; assert flags match CLI `--help`.

```powershell
& '.\artifacts\python-3.12.10\python.exe' -m unittest tests.test_submission_docs -v
git add README.md docs/submission tests/test_submission_docs.py
git commit -m "docs: assemble Track 5 submission package"
```

---

### Task 7: Verify, Merge, Record, and Publish

**Time box:** T+2:40 to T+4:00; stop feature work at T+3:25.

**Files:**
- Verify all selected Issue #9/#10/#11 files.
- External: GitHub visibility, YouTube, Devpost.

**Interfaces:**
- Consumes: predictor, evidence, docs, demo script, and claim ledger.
- Produces: public repository/video/Devpost and verified `main`.

- [ ] **Step 1: Run predictor twice**

  Use the README command on `fixtures/experiment/images`. Require exact keys, finite predictions, stable ordering, and identical same-device hashes.

- [ ] **Step 2: Run full gate**

```powershell
& '.\artifacts\python-3.12.10\python.exe' -m unittest discover -s tests -p 'test*.py' -v
npm test
git diff --check
git status --short --branch
```

  Do not rely on `npm run verify`; it omits fusion, sealed, evidence, and submission tests.

- [ ] **Step 3: Scan the public tree**

  Check tracked files for secrets, absolute user paths, data/checkpoint bytes, private caches, unsupported claims, and missing attribution. Confirm `artifacts/` and `datasets/` remain ignored.

- [ ] **Step 4: Merge in order**

  Merge Issue #10 only if selected and fully accepted, then verified Issue #9, then update/test/merge Issue #11. Push `main` only after the final gate.

- [ ] **Step 5: Record and review video**

  The owner records final inference, combines rehearsed footage, verifies every spoken number against the ledger, and excludes third-party UI/logos and unlicensed images.

- [ ] **Step 6: Perform publication actions**

  After the scan, the owner changes GitHub from private to public, uploads the video publicly to YouTube, pastes public repository/video URLs and Devpost copy, confirms contributions, and submits.

- [ ] **Step 7: Verify public surfaces**

  While signed out, clone/open GitHub, play YouTube in a private window, and open Devpost. Confirm setup, links, SVG, FP/FN note, and predictor command work before T+4:00.

## Final Acceptance Checklist

- [ ] Devpost covers problem fit, tools, models/APIs, libraries/frameworks, datasets/assets, limitations, contributions, repository, and video.
- [ ] GitHub is public and contains one working directory-to-JSON command.
- [ ] Public code is structured, documented at its module boundaries, and free of dead submission paths.
- [ ] Prediction records contain exactly `image_path` and `pred`.
- [ ] README contains overview, setup, inference, reproduction, limitations/improvements, and contributions.
- [ ] Public YouTube demonstrates inference, JSON, robustness, and error analysis without unauthorized content.
- [ ] Table/SVG compares clean with JPEG, blur, resize, noise, color, and crop.
- [ ] Error note contains deterministic clean/transformed FP/FN cases and threshold trade-off.
- [ ] Every number is candidate-bound, split-labeled, checksummed, and in the claim ledger.
- [ ] Full Python/Node suites pass; diffs and worktrees are clean before merges.
