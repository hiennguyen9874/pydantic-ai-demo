from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

import download_papers  # noqa: E402
from download_papers import (  # noqa: E402
    PaperRecord,
    download_paper,
    download_paper_source,
    iter_paper_records,
    load_configured_models,
)


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

    def test_arxiv_source_is_downloaded_to_a_temporary_tar_gz_and_extracted(self) -> None:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as source:
            contents = b"\\documentclass{article}\\n"
            member = tarfile.TarInfo("main.tex")
            member.size = len(contents)
            source.addfile(member, io.BytesIO(contents))

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            record = PaperRecord("Document information extraction", "LayoutLM", "1912.13318")
            with patch.object(
                download_papers, "urlopen", return_value=closing(io.BytesIO(archive.getvalue()))
            ) as urlopen:
                result = download_paper_source(record, output)

            target = output / "Document information extraction" / "1912.13318_LayoutLM"
            self.assertEqual(result, "downloaded")
            self.assertEqual((target / "main.tex").read_bytes(), b"\\documentclass{article}\\n")
            urlopen.assert_called_once_with("https://arxiv.org/src/1912.13318", timeout=60)

    def test_existing_source_directory_is_skipped_without_a_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            target = output / "Group name" / "2508.10104_Model name"
            target.mkdir(parents=True)
            record = PaperRecord("Group name", "Model name", "2508.10104")

            with patch.object(download_papers, "urlopen") as urlopen:
                result = download_paper_source(record, output)

            self.assertEqual(result, "skipped")
            urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
