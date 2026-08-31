"""Public submission documentation contract tests.

The agreed public seam is the repository's submission-facing Markdown: a
participant can copy the single canonical RGB command and understand its JSON
contract without relying on internal implementation details.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class SubmissionReadmeTests(unittest.TestCase):
    def test_readme_publishes_the_candidate_bound_rgb_inference_contract(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("## Canonical inference (current candidate)", readme)
        self.assertIn(
            "python rgb_cli.py --input-dir ./images --output ./predictions.json "
            "--resolution 384 --device auto --batch-size 8",
            readme,
        )
        self.assertIn('"image_path"', readme)
        self.assertIn('"pred"', readme)
        self.assertIn("## Setup", readme)
        self.assertIn("## Training, calibration, and reporting order", readme)
        self.assertIn("## Limitations and improvements", readme)
        self.assertIn("## Contributions", readme)
        self.assertIn("docs/submission/claim-ledger.md", readme)
        self.assertEqual(readme.count("## Canonical inference (current candidate)"), 1)
        self.assertEqual(readme.count("python rgb_cli.py --input-dir"), 1)


class SubmissionPackageTests(unittest.TestCase):
    def test_devpost_uses_required_headings_and_current_candidate(self) -> None:
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
        self.assertIn("raw RGB-only", devpost)
        self.assertIn("internal validation", devpost)
        self.assertNotIn("0.677 / 0.323", devpost)

    def test_documented_rgb_flags_are_supported_by_cli_help(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        help_output = subprocess.run(
            [sys.executable, "rgb_cli.py", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout

        command = (
            "python rgb_cli.py --input-dir ./images --output ./predictions.json "
            "--resolution 384 --device auto --batch-size 8"
        )
        for flag in ("--input-dir", "--output", "--resolution", "--device", "--batch-size"):
            self.assertIn(flag, command)
            self.assertIn(flag, help_output)
        self.assertIn(command, readme)

    def test_supporting_documents_preserve_provenance_and_human_review_gates(self) -> None:
        attributions = (ROOT / "docs" / "submission" / "attributions.md").read_text(encoding="utf-8")
        demo = (ROOT / "docs" / "submission" / "demo-script.md").read_text(encoding="utf-8")
        ledger = (ROOT / "docs" / "submission" / "claim-ledger.md").read_text(encoding="utf-8")

        for name in ("SID_Set", "Community Forensics", "PyTorch", "NumPy", "Pillow", "Hugging Face", "Node.js", "Sharp"):
            self.assertIn(name, attributions)
        self.assertIn("120-second", demo)
        self.assertIn("internal validation", demo)
        self.assertIn("Human contribution record", ledger)
        self.assertIn("Bundle SHA-256", ledger)
        self.assertIn("Repeated-output SHA-256", ledger)


if __name__ == "__main__":
    unittest.main()
