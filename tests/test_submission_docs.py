"""Public submission documentation and acceptance-record contract tests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_COMMAND = (
    "python submission_inference_cli.py --image-dir <directory> "
    "--bundle-dir artifacts/issue-7-fusion-v2 "
    "--rgb-checkpoint <community-forensics-384.safetensors> "
    "--signal-model <signal-model.json> --output predictions.json "
    "--device auto --batch-size 8"
)
ISSUE10_COMMIT = "b8982dfb3400fa92fde65cc0ea6f2fe141a4b402"
SIGNAL_PROFILE = "hackathon-v1"
SIGNAL_CHECKPOINT_REVISION = (
    "signal-checkpoint-v1-4a6b4d974722c9f8729a90d872387bb49e54d01e7bda98ddb69232c28604390e"
)
SIGNAL_NORMALIZATION_REVISION = (
    "signal-normalization-v1-25b16b78f7ecb5e02572e03650537e8b5e266f2f3e49a911a2ae2e2e11d45e80"
)


def _canonical_lf_bytes(path: Path) -> bytes:
    """Return repository-canonical text bytes independent of Git checkout EOLs."""

    return path.read_bytes().replace(b"\r\n", b"\n")


class SubmissionReadmeTests(unittest.TestCase):
    def test_readme_publishes_the_accepted_fusion_cli_and_contract(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("## Submission inference status", readme)
        self.assertIn("learned-static-fusion", readme)
        self.assertIn("0.677 RGB / 0.323 signal", readme)
        self.assertEqual(readme.count(CANONICAL_COMMAND), 1)
        self.assertIn(ISSUE10_COMMIT, readme)
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
            "21bbd744c94927e674bd9f40b3f56c9ac3188580b49b2d32869cb576e65dd2c2",
            "independently reviewed",
            SIGNAL_PROFILE,
            "8,064 training draws",
            "400 validation sources",
            "8,000 validation observations",
            SIGNAL_CHECKPOINT_REVISION,
            SIGNAL_NORMALIZATION_REVISION,
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

    def test_publication_links_keep_only_human_actions_open(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        devpost = (ROOT / "docs/submission/devpost.md").read_text(encoding="utf-8")
        ledger = (ROOT / "docs/submission/claim-ledger.md").read_text(encoding="utf-8")

        for document in (readme, devpost, ledger):
            self.assertIn("https://github.com/BeefyPotato/adaptive-aigc-forensics", document)
            self.assertIn("HUMAN REQUIRED", document)
            self.assertIn("YouTube", document)
            self.assertIn("Devpost", document)
        self.assertIn("make the repository public", readme)
        self.assertIn("_Unassigned — do not infer_", ledger)

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
        self.assertEqual(devpost.count(CANONICAL_COMMAND), 1)
        self.assertIn(ISSUE10_COMMIT, devpost)
        self.assertIn("internal validation", devpost)
        self.assertIn("training, calibration, any selection, weights, thresholds, templates, or narrative", devpost)
        self.assertNotIn("current raw RGB-only candidate", devpost)
        self.assertNotIn("python rgb_cli.py --input-dir ./images --output ./predictions.json", devpost)
        self.assertNotIn("](attributions.md)", devpost)
        self.assertNotIn("](../data-sources.md)", devpost)
        self.assertNotIn("](claim-ledger.md)", devpost)
        self.assertNotIn("](demo-script.md)", devpost)
        for public_link in (
            "https://github.com/BeefyPotato/adaptive-aigc-forensics/blob/main/docs/submission/attributions.md",
            "https://github.com/BeefyPotato/adaptive-aigc-forensics/blob/main/docs/data-sources.md",
            "https://github.com/BeefyPotato/adaptive-aigc-forensics/blob/main/docs/submission/claim-ledger.md",
            "https://github.com/BeefyPotato/adaptive-aigc-forensics/blob/main/docs/submission/demo-script.md",
        ):
            self.assertIn(public_link, devpost)

    def test_accepted_fusion_interface_preserves_the_submission_output_contract(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        devpost = (ROOT / "docs" / "submission" / "devpost.md").read_text(encoding="utf-8")
        ledger = (ROOT / "docs" / "submission" / "claim-ledger.md").read_text(encoding="utf-8")

        for document in (readme, devpost, ledger):
            self.assertIn("learned-static-fusion", document)
            self.assertIn(CANONICAL_COMMAND, document)
            self.assertIn(ISSUE10_COMMIT, document)
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
        self.assertIn(CANONICAL_COMMAND, demo)
        self.assertIn("internal validation", demo)
        self.assertIn("Human contribution record", ledger)
        self.assertIn("Bundle SHA-256", ledger)
        self.assertIn("Final accepted command", ledger)
        self.assertIn("Evidence generation revision", ledger)
        self.assertIn("Tracked report", ledger)
        self.assertIn("independently reviewed and accepted", ledger)
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

    def test_final_fusion_evidence_is_linked_and_bound_to_cli_acceptance(self) -> None:
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
        self.assertIn(ISSUE10_COMMIT, ledger)
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
        self.assertIn("trusted-bundle-only limitation", ledger)

        runtime = json.loads((ROOT / "docs/submission/runtime-smoke.json").read_text("utf-8"))
        self.assertEqual(runtime["schema_version"], "submission-runtime-acceptance-v1")
        self.assertTrue(runtime["accepted_submission_cli"])
        self.assertEqual(runtime["canonical_command"], CANONICAL_COMMAND)
        self.assertEqual(runtime["independent_review"], "accepted")
        self.assertEqual(runtime["issue10_commit"], ISSUE10_COMMIT)
        for binding in ("generation_revision", "bundle_revision", "bundle_sha256"):
            self.assertEqual(runtime["artifact_bindings"][binding], evidence["bindings"][binding])
        self.assertEqual(runtime["artifact_bindings"]["signal_profile"], SIGNAL_PROFILE)
        self.assertEqual(
            runtime["artifact_bindings"]["signal_checkpoint_revision"],
            SIGNAL_CHECKPOINT_REVISION,
        )
        self.assertEqual(
            runtime["artifact_bindings"]["signal_normalization_revision"],
            SIGNAL_NORMALIZATION_REVISION,
        )
        fixture = runtime["runs"]["fixture"]
        self.assertEqual(fixture["profile"]["wall_seconds"], 23.18)
        self.assertEqual(fixture["profile"]["peak_working_set_bytes"], 466_698_240)
        self.assertEqual(fixture["cpu_output_sha256"], fixture["auto_output_sha256"])
        self.assertEqual(
            fixture["cpu_output_sha256"],
            "adcd0528bd98130421385fd7d579ea8ba4ae6aa773f1c4b6e90504a2c749c1b3",
        )
        self.assertEqual(fixture["maximum_absolute_parity_delta"], 0)
        sampled_real = runtime["runs"]["sampled_real"]
        self.assertFalse(sampled_real["organizer_data"])
        self.assertEqual(sampled_real["input_image_count"], 4)
        self.assertEqual(sampled_real["maximum_absolute_parity_delta"], 0)
        self.assertEqual(
            sampled_real["output_sha256"],
            "21bbd744c94927e674bd9f40b3f56c9ac3188580b49b2d32869cb576e65dd2c2",
        )
        expected_inputs = [
            ("authentic/0002d5c6b40edcd4.jpg", 147_109, "238ac3d883867d9253b9f66953ccca969793b1644d08df53f907bf2c400c54c6"),
            ("authentic/00032d5bb63c29eb.jpg", 136_504, "4a7a351d5fff74295074834efbdad92b53d41754ed2bde70a9e0f3871abc4a5b"),
            ("full-synthetic/full_synthetic_000021.jpg", 173_637, "ed04d319ba8c9d4dd688393e2b10dbe0172deffc10f1ccb0d4387744384fa9b0"),
            ("full-synthetic/full_synthetic_000022.jpg", 176_449, "4e287901a8ecb69f783223e05d59af141f3d69e92bc3b7bc2a06c9c73cca835d"),
        ]
        self.assertEqual(
            [(item["image_path"], item["bytes"], item["sha256"]) for item in sampled_real["inputs"]],
            expected_inputs,
        )
        for runtime_path in (
            "runs.fixture.profile.wall_seconds", "runs.fixture.profile.images_per_second",
            "runs.fixture.profile.peak_working_set_bytes", "accepted_submission_cli",
            "canonical_command", "runs.fixture.cpu_output_sha256",
            "runs.sampled_real.output_sha256", "runs.sampled_real.inputs",
        ):
            self.assertIn(runtime_path, ledger)

    def test_canonical_cli_help_exposes_every_documented_option(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "submission_inference_cli.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for option in (
            "--image-dir", "--bundle-dir", "--rgb-checkpoint", "--signal-model",
            "--output", "--device", "--batch-size",
        ):
            self.assertIn(option, completed.stdout)


if __name__ == "__main__":
    unittest.main()
