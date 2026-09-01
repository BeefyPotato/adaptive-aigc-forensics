"""Public submission documentation contract tests.

The agreed public seam is the repository's submission-facing Markdown: it
identifies the fusion design and its JSON contract without presenting an
unaccepted implementation command as the submitted system.
"""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()
