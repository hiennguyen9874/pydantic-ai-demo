"""Download paper Markdown or arXiv LaTeX sources for classified models.

Examples:
    uv run python models-classification/download_papers.py
    uv run python models-classification/download_papers.py --source

The large groups checkpoint is parsed as YAML events, so only one document record is
kept in memory at a time. Markdown and arXiv source downloads are streamed directly
to temporary destinations before being published.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal
from urllib.request import urlopen

import yaml
from yaml.events import (
    DocumentEndEvent,
    DocumentStartEvent,
    MappingEndEvent,
    MappingStartEvent,
    ScalarEvent,
    SequenceEndEvent,
    SequenceStartEvent,
    StreamEndEvent,
    StreamStartEvent,
)


PROJECT_DIRECTORY = Path(__file__).parent
DEFAULT_CONFIG = PROJECT_DIRECTORY / "Multimodal" / "config.yaml"
DEFAULT_GROUPS = PROJECT_DIRECTORY / "Multimodal" / "groups.yaml"


@dataclass(frozen=True)
class PaperRecord:
    group_name: str
    model_name: str
    paper_id: str | None


def load_configured_models(path: Path) -> dict[str, str]:
    """Return config document paths mapped to their canonical model titles."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config {path}: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError(f"Config {path} must be a YAML list")

    models: dict[str, str] = {}
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Config {path} entry {index} must be a mapping")
        local, title = item.get("local"), item.get("title")
        if not isinstance(local, str) or not local or not isinstance(title, str) or not title:
            raise ValueError(f"Config {path} entry {index} must have non-empty local and title values")
        if local in models:
            raise ValueError(f"Config {path} contains duplicate local path {local!r}")
        models[local] = title
    return models


def _next(events: Iterator[object], expected: type[object]) -> object:
    event = next(events, None)
    if not isinstance(event, expected):
        actual = type(event).__name__ if event is not None else "end of file"
        raise ValueError(f"Invalid groups YAML: expected {expected.__name__}, got {actual}")
    return event


def _is_yaml_null(event: ScalarEvent) -> bool:
    return event.tag == "tag:yaml.org,2002:null" or (
        event.style is None and event.value in {"~", "null", "Null", "NULL"}
    )


def _skip_node(events: Iterator[object], first_event: object) -> None:
    """Consume an already-started YAML node without constructing its value."""
    if not isinstance(first_event, (MappingStartEvent, SequenceStartEvent)):
        return

    depth = 1
    while depth:
        event = next(events, None)
        if event is None:
            raise ValueError("Invalid groups YAML: unexpected end of file")
        if isinstance(event, (MappingStartEvent, SequenceStartEvent)):
            depth += 1
        elif isinstance(event, (MappingEndEvent, SequenceEndEvent)):
            depth -= 1


def _read_document_with_start(events: Iterator[object]) -> dict[str, str | None]:
    """Parse one document after its MappingStartEvent has been consumed."""
    document: dict[str, str | None] = {}
    while True:
        event = next(events, None)
        if isinstance(event, MappingEndEvent):
            return document
        if not isinstance(event, ScalarEvent):
            raise ValueError("Invalid groups YAML: expected a document field name")
        value_event = next(events, None)
        if value_event is None:
            raise ValueError("Invalid groups YAML: unexpected end of file")
        if event.value in {"local", "paper_id"}:
            if not isinstance(value_event, ScalarEvent):
                raise ValueError(f"Invalid groups YAML: {event.value} must be a scalar")
            document[event.value] = None if _is_yaml_null(value_event) else value_event.value
        else:
            _skip_node(events, value_event)


def iter_paper_records(groups_path: Path, configured_models: dict[str, str]) -> Iterator[PaperRecord]:
    """Yield configured models in the checkpoint without loading its full YAML tree."""
    try:
        source = groups_path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read groups {groups_path}: {exc}") from exc

    with source:
        events = iter(yaml.parse(source))
        _next(events, StreamStartEvent)
        _next(events, DocumentStartEvent)
        _next(events, MappingStartEvent)

        while True:
            event = next(events, None)
            if isinstance(event, MappingEndEvent):
                break
            if not isinstance(event, ScalarEvent):
                raise ValueError("Invalid groups YAML: expected a top-level field name")
            value_event = next(events, None)
            if value_event is None:
                raise ValueError("Invalid groups YAML: unexpected end of file")
            if event.value != "groups":
                _skip_node(events, value_event)
                continue
            if not isinstance(value_event, MappingStartEvent):
                raise ValueError("Invalid groups YAML: groups must be a mapping")
            while True:
                group_event = next(events, None)
                if isinstance(group_event, MappingEndEvent):
                    break
                if not isinstance(group_event, ScalarEvent):
                    raise ValueError("Invalid groups YAML: expected a group name")
                group_name = group_event.value
                group_value = next(events, None)
                if not isinstance(group_value, MappingStartEvent):
                    raise ValueError("Invalid groups YAML: each group must be a mapping")
                for document in _read_group_documents_with_start(events, group_value):
                    local = document.get("local")
                    if isinstance(local, str) and local in configured_models:
                        paper_id = document.get("paper_id")
                        yield PaperRecord(
                            group_name=group_name,
                            model_name=configured_models[local],
                            paper_id=paper_id if isinstance(paper_id, str) and paper_id else None,
                        )

        _next(events, DocumentEndEvent)
        _next(events, StreamEndEvent)


def _read_group_documents_with_start(
    events: Iterator[object], _start_event: MappingStartEvent
) -> Iterator[dict[str, str | None]]:
    """Parse a group after its MappingStartEvent has already been consumed."""
    while True:
        event = next(events, None)
        if isinstance(event, MappingEndEvent):
            return
        if not isinstance(event, ScalarEvent):
            raise ValueError("Invalid groups YAML: expected a group field name")
        value_event = next(events, None)
        if value_event is None:
            raise ValueError("Invalid groups YAML: unexpected end of file")
        if event.value != "documents":
            _skip_node(events, value_event)
            continue
        if not isinstance(value_event, SequenceStartEvent):
            raise ValueError("Invalid groups YAML: documents must be a sequence")
        while True:
            document_event = next(events, None)
            if isinstance(document_event, SequenceEndEvent):
                break
            if not isinstance(document_event, MappingStartEvent):
                raise ValueError("Invalid groups YAML: each document must be a mapping")
            yield _read_document_with_start(events)


def _path_component(value: str, kind: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"Invalid {kind} for output path: {value!r}")
    return value


def download_paper(
    record: PaperRecord, output_directory: Path, *, overwrite: bool = False
) -> Literal["downloaded", "skipped", "missing-paper-id", "failed"]:
    """Run hf with stdout directed to an atomic temporary destination file."""
    if record.paper_id is None:
        return "missing-paper-id"

    target = (
        output_directory
        / _path_component(record.group_name, "group name")
        / f"{_path_component(record.paper_id, 'paper ID')}_{_path_component(record.model_name, 'model name')}.md"
    )
    if target.exists() and not overwrite:
        return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            result = subprocess.run(
                ["uv", "run", "hf", "papers", "read", record.paper_id],
                stdout=temporary,
                cwd=PROJECT_DIRECTORY.parent,
                check=False,
            )
        if result.returncode != 0:
            return "failed"
        os.replace(temporary_path, target)
        temporary_path = None
        return "downloaded"
    except OSError as exc:
        print(f"failed: {record.model_name} ({record.paper_id}): {exc}")
        return "failed"
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _source_target(record: PaperRecord, output_directory: Path) -> Path:
    """Return the output directory for an arXiv source archive."""
    assert record.paper_id is not None
    return (
        output_directory
        / _path_component(record.group_name, "group name")
        / f"{_path_component(record.paper_id, 'paper ID')}_{_path_component(record.model_name, 'model name')}"
    )


def _extract_tar_safely(archive: Path, destination: Path) -> None:
    """Extract archive without allowing member paths to escape destination."""
    destination_root = destination.resolve()
    with tarfile.open(archive, mode="r:*") as source:
        members = source.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if member_path != destination_root and destination_root not in member_path.parents:
                raise ValueError(f"Archive member escapes destination: {member.name!r}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"Archive member has unsupported type: {member.name!r}")
        source.extractall(destination, members=members)


def _download_to_file(response: BinaryIO, destination: Path) -> None:
    with destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def download_paper_source(
    record: PaperRecord, output_directory: Path, *, overwrite: bool = False
) -> Literal["downloaded", "skipped", "missing-paper-id", "failed"]:
    """Download and safely extract an arXiv source archive into the model directory."""
    if record.paper_id is None:
        return "missing-paper-id"

    target = _source_target(record, output_directory)
    if target.exists() and not overwrite:
        return "skipped"

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(dir=target.parent, prefix=f".{target.name}.") as temporary:
            temporary_directory = Path(temporary)
            archive = temporary_directory / f"{record.paper_id}.tar.gz"
            extracted = temporary_directory / "source"
            with urlopen(f"https://arxiv.org/src/{record.paper_id}", timeout=60) as response:
                _download_to_file(response, archive)
            extracted.mkdir()
            _extract_tar_safely(archive, extracted)

            if target.exists():
                shutil.rmtree(target)
            os.replace(extracted, target)
        return "downloaded"
    except (OSError, tarfile.TarError, ValueError) as exc:
        print(f"failed: {record.model_name} ({record.paper_id}): {exc}")
        return "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--groups", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Root for group folders (default: the config directory)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing downloaded output")
    parser.add_argument(
        "--source",
        action="store_true",
        help="Download and extract arXiv LaTeX source archives instead of Markdown",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configured_models = load_configured_models(args.config)
    output_directory = args.output_dir or args.config.parent
    counts: dict[str, int] = {"downloaded": 0, "skipped": 0, "missing-paper-id": 0, "failed": 0}

    download = download_paper_source if args.source else download_paper
    output_suffix = "/" if args.source else ".md"
    for record in iter_paper_records(args.groups, configured_models):
        outcome = download(record, output_directory, overwrite=args.overwrite)
        counts[outcome] += 1
        if outcome == "downloaded":
            print(f"downloaded: {record.group_name}/{record.paper_id}_{record.model_name}{output_suffix}")
        elif outcome == "missing-paper-id":
            print(f"no paper id: {record.group_name}/{record.model_name}")
        elif outcome == "failed":
            print(f"failed: {record.group_name}/{record.model_name} ({record.paper_id})")

    print(
        "done: " + ", ".join(f"{count} {name}" for name, count in counts.items())
    )


if __name__ == "__main__":
    main()
