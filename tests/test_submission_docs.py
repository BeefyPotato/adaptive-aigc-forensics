"""Public submission documentation contract tests.

The agreed public seam is the repository's submission-facing Markdown: it
identifies the fusion design and its JSON contract without presenting an
unaccepted implementation command as the submitted system.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _canonical_lf_bytes(path: Path) -> bytes:
    """Return repository-canonical text bytes independent of Git checkout EOLs."""

    return path.read_bytes().replace(b"\r\n", b"\n")


class SubmissionReadmeTests(unittest.TestCase):
    def test_readme_publishes_the_fusion_design_without_claiming_an_unaccepted_cli(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("## Submission inference status", readme)
        self.assertIn("learned-static-fusion", readme)
        self.assertIn("0.677 RGB / 0.323 signal", readme)
        self.assertIn("pending Issue #10 acceptance and final CLI binding", readme)
        self.assertIn("JSON array", readme)
        self.assertIn("exactly this per-image schema", readme)
        self.assertIn('"image_path": "relative/path/to/image.png"', readme)
        self.assertIn('"pred": 0.73', readme)
        self.assertIn("finite probability in `[0, 1]`", readme)
        self.assertIn("## Setup", readme)
        self.assertIn("## Training, calibration, and reporting order", readme)
        self.assertIn("## Limitations and improvements", readme)
        self.assertIn("## Contributions", readme)
        self.assertIn("docs/submission/claim-ledger.md", readme)
        self.assertIn(
            "training, calibration, any selection, weights, thresholds, templates, or narrative",
            readme,
        )
        self.assertNotIn("current raw-RGB candidate", readme)
        self.assertNotIn("sole public inference command", readme)
        self.assertNotIn("python rgb_cli.py --input-dir ./images --output ./predictions.json", readme)

    def test_readme_discloses_scale_splits_compute_and_reproduction(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        model_config = json.loads((ROOT / "config/community-forensics-models.json").read_text("utf-8"))

        self.assertEqual(model_config["models"]["384"]["parameter_count"], 21_811_969)
        self.assertEqual(26 * 16 + 16 + 16 + 1, 449)
        for disclosure in (
            "21,811,969 parameters",
            "449 trainable scalar parameters",
            "416 + 16 + 16 + 1",
            "8,000 expert-training sources",
            "2,000 fusion-training sources",
            "2,000 internal-validation sources",
            "2,000 sealed-internal-test sources",
            "23.18 seconds",
            "466,698,240 bytes",
            "adcd0528bd98130421385fd7d579ea8ba4ae6aa773f1c4b6e90504a2c749c1b3",
            "same-device smoke evidence, not independent Issue #10 acceptance",
        ):
            self.assertIn(disclosure, readme)
        for command in (
            "python submission_evidence.py --generation-dir ./artifacts/issue-7-fusion-v2",
            "python submission_report.py ./artifacts/submission-reproduction/evidence "
            "./artifacts/submission-reproduction/results",
            'python -m unittest discover -s tests -p "test*.py" -v',
            "npm test",
        ):
            self.assertIn(command, readme)
        self.assertNotIn("npm run verify", readme)

        devpost = (ROOT / "docs/submission/devpost.md").read_text(encoding="utf-8")
        for document in (readme, devpost):
            for result in (
                "768/1218 = 0.6305418719211823",
                "0.016795535714285936",
                "[0.011076105794972707, 0.0234869800759804]",
            ):
                self.assertIn(result, document)
            self.assertIn("descriptive", document)
            self.assertIn("not a causal", document)


class SubmissionPackageTests(unittest.TestCase):
    def test_devpost_uses_required_headings_and_selected_fusion_design(self) -> None:
        devpost = (ROOT / "docs" / "submission" / "devpost.md").read_text(encoding="utf-8")

        for heading in (
            "# Adaptive AIGC Forensics",
            "## Problem and solution",
            "## Technical implementation",
            "## Development tools",
            "## Models and APIs",
            "## Libraries and frameworks",
            "## Datasets and assets",
            "## Robustness and error analysis",
            "## Innovation and complementary value",
            "## Impact and feasibility",
            "## Limitations and next steps",
            "## Team contributions",
            "## Demo and repository",
        ):
            self.assertIn(heading, devpost)
        self.assertIn("learned-static-fusion", devpost)
        self.assertIn("0.677 RGB / 0.323 signal", devpost)
        self.assertIn("cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5", devpost)
        self.assertIn("pending Issue #10 acceptance and final CLI binding", devpost)
        self.assertIn("internal validation", devpost)
        self.assertIn("training, calibration, any selection, weights, thresholds, templates, or narrative", devpost)
        self.assertNotIn("current raw RGB-only candidate", devpost)
        self.assertNotIn("python rgb_cli.py --input-dir ./images --output ./predictions.json", devpost)

    def test_pending_fusion_interface_preserves_the_submission_output_contract(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        devpost = (ROOT / "docs" / "submission" / "devpost.md").read_text(encoding="utf-8")
        ledger = (ROOT / "docs" / "submission" / "claim-ledger.md").read_text(encoding="utf-8")

        for document in (readme, devpost, ledger):
            self.assertIn("learned-static-fusion", document)
            self.assertIn("pending Issue #10 acceptance", document)
        self.assertIn('"image_path": "relative/path/to/image.png"', readme)
        self.assertIn('`{ "image_path": string, "pred": number }`', devpost)
        self.assertIn("exactly `image_path` and `pred`", ledger)

    def test_supporting_documents_preserve_provenance_and_human_review_gates(self) -> None:
        attributions = (ROOT / "docs" / "submission" / "attributions.md").read_text(encoding="utf-8")
        demo = (ROOT / "docs" / "submission" / "demo-script.md").read_text(encoding="utf-8")
        ledger = (ROOT / "docs" / "submission" / "claim-ledger.md").read_text(encoding="utf-8")

        for name in (
            "SID_Set", "Community Forensics", "PyTorch", "NumPy", "Pillow",
            "Hugging Face", "timm", "safetensors", "Node.js", "Sharp",
        ):
            self.assertIn(name, attributions)
        self.assertIn("120-second", demo)
        self.assertIn("learned-static-fusion", demo)
        self.assertIn("0.677 RGB / 0.323 signal", demo)
        self.assertIn("Do not record inference", demo)
        self.assertIn("internal validation", demo)
        self.assertIn("Human contribution record", ledger)
        self.assertIn("Bundle SHA-256", ledger)
        self.assertIn("Final accepted command", ledger)
        self.assertIn("Evidence generation revision", ledger)
        self.assertIn("Tracked report", ledger)
        self.assertIn("pending Issue #10 acceptance", ledger)
        self.assertIn(
            "static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181",
            ledger,
        )
        self.assertIn(
            "static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179",
            ledger,
        )
        self.assertIn("9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2", ledger)
        self.assertIn("cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5", ledger)
        self.assertIn("static-fusion-weight-v1-96456ffa07a98fc81ceef01f4cbae62a52b0c07fc1e71c4f5a06b5a06eef1c1b", ledger)
        self.assertIn("claim-ledger-complete and human-reviewed", demo)

    def test_final_fusion_evidence_is_linked_and_bound_without_claiming_cli_acceptance(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        devpost = (ROOT / "docs" / "submission" / "devpost.md").read_text(encoding="utf-8")
        demo = (ROOT / "docs" / "submission" / "demo-script.md").read_text(encoding="utf-8")
        ledger = (ROOT / "docs" / "submission" / "claim-ledger.md").read_text(encoding="utf-8")

        for path in (
            "docs/submission/evidence/submission-evidence.json",
            "docs/submission/evidence/submission-evidence.complete.json",
            "docs/submission/results/robustness-and-errors.md",
            "docs/submission/results/clean-vs-transformed.svg",
            "docs/submission/results/submission-report.complete.json",
        ):
            self.assertIn(path, ledger)
        self.assertIn("docs/submission/results/robustness-and-errors.md", readme)
        self.assertIn("docs/submission/results/clean-vs-transformed.svg", devpost)
        self.assertIn("docs/submission/results/clean-vs-transformed.svg", demo)

        for metric in (
            "0.981975", "0.9567506944444445", "0.9603541666666667",
            "noise / sigma-0.1 / 0.810425", "0.09982875909583232",
            "0.09680062702417022", "0.87625", "0.08399999999999996",
            "0.16349999999999998",
        ):
            self.assertIn(metric, devpost)
        for binding in (
            "submission-evidence-generation-v1-b018d8f0326f8a9ed9945b52eda2dadeae659b3827268af69c38aa4c09e27cc1",
            "submission-report-generation-v1-411d7380b4667552401f4f751472836d7d3186854f50694dff5039ce0c19e796",
            "0c3ed99c9805a4d455502f446637f33d784f541536bb1445b75ef83c6c767f90",
            "5152f58ad323cb4d4afc57dac8f209c86a7ba56a95bbf7466bb2dbc2589a4c36",
            "9ab6378752417637d6ae3e24443c5c49aff498fbdae11b01f82a1267bf6f486b",
            "8163495995381f52fbccfd754cdb4c3aecfdd918949ee3d5fca0ad6c6fdef3f6",
            "d84773233606bb6f32f3fd6d226155a80d1113ee87c166f3c34f1382190f3072",
        ):
            self.assertIn(binding, ledger)
        self.assertIn("pending Issue #10 acceptance", ledger)
        self.assertNotIn("Final evidence/report receipts | **pending", ledger)

    def test_public_evidence_bytes_receipts_metrics_and_ledger_agree(self) -> None:
        expected_hashes = {
            "docs/submission/evidence/submission-evidence.json": "0c3ed99c9805a4d455502f446637f33d784f541536bb1445b75ef83c6c767f90",
            "docs/submission/evidence/submission-evidence.complete.json": "5152f58ad323cb4d4afc57dac8f209c86a7ba56a95bbf7466bb2dbc2589a4c36",
            "docs/submission/results/robustness-and-errors.md": "9ab6378752417637d6ae3e24443c5c49aff498fbdae11b01f82a1267bf6f486b",
            "docs/submission/results/clean-vs-transformed.svg": "8163495995381f52fbccfd754cdb4c3aecfdd918949ee3d5fca0ad6c6fdef3f6",
            "docs/submission/results/submission-report.complete.json": "d84773233606bb6f32f3fd6d226155a80d1113ee87c166f3c34f1382190f3072",
        }
        for relative_path, expected_hash in expected_hashes.items():
            with self.subTest(relative_path=relative_path):
                artifact = ROOT / relative_path
                self.assertTrue(artifact.is_file(), f"missing merged Issue #9 artifact: {relative_path}")
                self.assertEqual(hashlib.sha256(_canonical_lf_bytes(artifact)).hexdigest(), expected_hash)

        evidence = json.loads((ROOT / "docs/submission/evidence/submission-evidence.json").read_text("utf-8"))
        evidence_receipt = json.loads(
            (ROOT / "docs/submission/evidence/submission-evidence.complete.json").read_text("utf-8")
        )
        report_receipt = json.loads(
            (ROOT / "docs/submission/results/submission-report.complete.json").read_text("utf-8")
        )
        ledger = (ROOT / "docs/submission/claim-ledger.md").read_text(encoding="utf-8")

        self.assertEqual(evidence["system_id"], "learned-static-fusion")
        self.assertEqual(evidence["evaluation_scope"], "internal-validation")
        self.assertEqual(evidence_receipt["system_id"], evidence["system_id"])
        self.assertEqual(report_receipt["system_id"], evidence["system_id"])
        self.assertIn(evidence["system_id"], ledger)
        for binding in ("generation_revision", "bundle_revision", "bundle_sha256"):
            self.assertEqual(evidence["bindings"][binding], evidence_receipt[binding])
            self.assertIn(evidence_receipt[binding], ledger)
        self.assertEqual(evidence_receipt["evidence_sha256"], expected_hashes[
            "docs/submission/evidence/submission-evidence.json"
        ])
        self.assertEqual(report_receipt["evidence_sha256"], evidence_receipt["evidence_sha256"])
        self.assertEqual(
            report_receipt["evidence_generation_revision"],
            evidence_receipt["evidence_generation_revision"],
        )
        self.assertEqual(
            report_receipt["files"],
            {
                "clean-vs-transformed.svg": expected_hashes[
                    "docs/submission/results/clean-vs-transformed.svg"
                ],
                "robustness-and-errors.md": expected_hashes[
                    "docs/submission/results/robustness-and-errors.md"
                ],
            },
        )

        metrics = evidence["metrics"]
        expected_metrics = {
            "metrics.clean_auroc": (metrics["clean_auroc"], 0.981975),
            "metrics.mean_corrupted_auroc": (metrics["mean_corrupted_auroc"], 0.9567506944444445),
            "metrics.all_condition_macro_auroc": (metrics["all_condition_macro_auroc"], 0.9603541666666667),
            "metrics.worst_family_severity.auroc": (metrics["worst_family_severity"]["auroc"], 0.810425),
            "metrics.brier_score": (metrics["brier_score"], 0.09982875909583232),
            "metrics.condition_balanced_brier_score": (
                metrics["condition_balanced_brier_score"], 0.09680062702417022,
            ),
            "metrics.threshold_diagnostics.balanced_accuracy": (
                metrics["threshold_diagnostics"]["balanced_accuracy"], 0.87625,
            ),
            "metrics.threshold_diagnostics.false_positive_rate": (
                metrics["threshold_diagnostics"]["false_positive_rate"], 0.08399999999999996,
            ),
            "metrics.threshold_diagnostics.false_negative_rate": (
                metrics["threshold_diagnostics"]["false_negative_rate"], 0.16349999999999998,
            ),
        }
        for json_path, (actual, expected) in expected_metrics.items():
            with self.subTest(json_path=json_path):
                self.assertEqual(actual, expected)
                self.assertIn(json_path, ledger)
                self.assertIn(str(expected), ledger)

        for revision in (
            "submission-evidence-generation-v1-b018d8f0326f8a9ed9945b52eda2dadeae659b3827268af69c38aa4c09e27cc1",
            "submission-report-generation-v1-411d7380b4667552401f4f751472836d7d3186854f50694dff5039ce0c19e796",
        ):
            self.assertIn(revision, ledger)
        self.assertEqual(
            evidence_receipt["evidence_generation_revision"],
            "submission-evidence-generation-v1-b018d8f0326f8a9ed9945b52eda2dadeae659b3827268af69c38aa4c09e27cc1",
        )
        self.assertEqual(
            report_receipt["report_generation_revision"],
            "submission-report-generation-v1-411d7380b4667552401f4f751472836d7d3186854f50694dff5039ce0c19e796",
        )

        for complementary_path in (
            "evaluation.complementary_value.rgb_errors_corrected_by_signal",
            "evaluation.complementary_value.rgb_errors",
            "evaluation.complementary_value.correction_rate",
            "evaluation.selection_evidence.all_condition_macro_auroc_gain",
            "evaluation.source_bootstrap_all_condition_macro_auroc_gain.lower",
            "evaluation.source_bootstrap_all_condition_macro_auroc_gain.upper",
        ):
            self.assertIn(complementary_path, ledger)

        runtime = json.loads((ROOT / "docs/submission/runtime-smoke.json").read_text("utf-8"))
        self.assertEqual(runtime["schema_version"], "submission-runtime-smoke-v1")
        self.assertFalse(runtime["accepted_submission_cli"])
        self.assertFalse(runtime["canonical_command"])
        self.assertEqual(runtime["independent_review"], "pending")
        self.assertEqual(runtime["profile"]["wall_seconds"], 23.18)
        self.assertEqual(runtime["profile"]["peak_working_set_bytes"], 466_698_240)
        self.assertEqual(
            runtime["output_sha256"],
            "adcd0528bd98130421385fd7d579ea8ba4ae6aa773f1c4b6e90504a2c749c1b3",
        )
        for runtime_path in (
            "profile.wall_seconds", "profile.images_per_second", "profile.peak_working_set_bytes",
            "accepted_submission_cli", "canonical_command", "output_sha256", "device_runs",
        ):
            self.assertIn(runtime_path, ledger)


if __name__ == "__main__":
    unittest.main()
