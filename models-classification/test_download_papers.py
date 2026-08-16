from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import download_papers  # noqa: E402
from download_papers import PaperRecord, download_paper, iter_paper_records, load_configured_models  # noqa: E402


class DownloadPapersTests(unittest.TestCase):
    def test_groups_are_streamed_and_filtered_to_configured_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yaml"
            groups = root / "groups.yaml"
            config.write_text("- local: docs/selected.md\n  title: Selected Model\n", encoding="utf-8")
            groups.write_text(
                """version: 3
groups:
  Group name:
    description: A nested value that should be skipped.
    documents:
    - local: docs/selected.md
      title: Different checkpoint title
      paper_id: '2508.10104'
      techniques:
      - a large field not retained
    - local: docs/not-in-config.md
      paper_id: '2501.00001'
    - local: docs/selected.md
      paper_id: null
""",
                encoding="utf-8",
            )

            configured_models = load_configured_models(config)
            with patch.object(download_papers.yaml, "safe_load", side_effect=AssertionError):
                records = list(iter_paper_records(groups, configured_models))

            self.assertEqual(
                records,
                [
                    PaperRecord("Group name", "Selected Model", "2508.10104"),
                    PaperRecord("Group name", "Selected Model", None),
                ],
            )

    def test_hf_stdout_is_written_directly_to_the_model_markdown_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = PaperRecord("Group name", "Model name", "2508.10104")
            calls: list[tuple[list[str], Path, bool]] = []

            def fake_run(command, *, stdout, cwd, check):
                calls.append((command, cwd, check))
                stdout.write(b"# Downloaded paper\n")
                return SimpleNamespace(returncode=0)

            with patch.object(download_papers.subprocess, "run", side_effect=fake_run):
                result = download_paper(record, output)

            target = output / "Group name" / "2508.10104_Model name.md"
            self.assertEqual(result, "downloaded")
            self.assertEqual(target.read_bytes(), b"# Downloaded paper\n")
            self.assertEqual(
                calls,
                [(["uv", "run", "hf", "papers", "read", "2508.10104"], download_papers.PROJECT_DIRECTORY.parent, False)],
            )


if __name__ == "__main__":
    unittest.main()
