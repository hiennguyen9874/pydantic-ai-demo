from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from classifier import (  # noqa: E402
    AgentDocumentClassifier,
    ClassificationState,
    GroupDecision,
    SourceDocument,
    classify_manifest,
    extract_hf_paper,
    extract_publication_year,
    load_state,
    read_manifest,
)


class FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, list[str]]] = []

    def classify(self, document: SourceDocument, content: str, groups):
        self.calls.append((document.title, content, list(groups)))
        if document.title == "First":
            return GroupDecision(
                group_name="Object detection",
                new_group_description="Models primarily designed to detect and localize objects in images.",
                techniques=["encoder-decoder transformer", "object queries"],
                domains=["computer vision"],
                tasks=["object detection"],
            )
        return GroupDecision(
            group_name="object detection",
            techniques=["convolutional neural network"],
            domains=["computer vision"],
            tasks=["object detection"],
        )


class ClassifierTests(unittest.TestCase):
    def test_local_openai_compatible_agent_constructs_without_a_network_request(self) -> None:
        classifier = AgentDocumentClassifier(
            "openai:gpt-5.6-luna",
            base_url="http://localhost:8317/v1",
            api_key="test-key",
        )
        self.assertEqual(classifier.agent.name, "model_document_classifier")

    def test_manifest_is_the_expected_list_shape(self) -> None:
        documents = read_manifest(Path(__file__).parent / "Multimodal" / "config.yaml")
        self.assertEqual(len(documents), 149)
        self.assertEqual(documents[0].title, "ALIGN")

    def test_extract_publication_year_only_from_explicit_hf_papers_header(self) -> None:
        self.assertEqual(
            extract_publication_year("This model was published in HF papers on 2020-05-26."),
            2020,
        )
        self.assertIsNone(extract_publication_year("Released in 2020, source unavailable."))

    def test_extract_hf_paper_returns_first_link_and_its_identifier(self) -> None:
        self.assertEqual(
            extract_hf_paper(
                "[ALBERT](https://huggingface.co/papers/1909.11942) and "
                "https://huggingface.co/papers/2401.12345"
            ),
            ("https://huggingface.co/papers/1909.11942", "1909.11942"),
        )
        self.assertEqual(
            extract_hf_paper("https://huggingface.co/papers/2401.12345"),
            ("https://huggingface.co/papers/2401.12345", "2401.12345"),
        )
        self.assertEqual(extract_hf_paper("No paper link."), (None, None))

    def test_legacy_checkpoint_is_rebuilt_for_metadata_cutover(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "groups.yaml"
            output_path.write_text(
                "version: 1\nsource_config: old.yaml\ngroups:\n  Vision transformer models:\n"
                "    description: Old grouping\n    documents: []\n",
                encoding="utf-8",
            )

            state = load_state(output_path, Path("new.yaml"))

            self.assertEqual(state.version, 3)
            self.assertEqual(state.source_config, str(Path("new.yaml").resolve()))
            self.assertEqual(state.groups, {})

    def test_batches_checkpoint_groups_and_retry_unreadable_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.md"
            second_path = root / "second.md"
            first_path.write_text(
                "This model was published in HF papers on 2020-05-26. "
                "[First](https://huggingface.co/papers/2005.12345) architecture",
                encoding="utf-8",
            )
            second_path.write_text("second architecture", encoding="utf-8")
            missing_path = root / "missing.md"
            output_path = root / "groups.yaml"
            documents = [
                SourceDocument(local=str(first_path), title="First"),
                SourceDocument(local=str(second_path), title="Second"),
                SourceDocument(local=str(missing_path), title="Missing"),
            ]
            state = ClassificationState(source_config="test-config.yaml")
            fake = FakeClassifier()

            classified, unreadable = classify_manifest(
                documents,
                state,
                fake,
                output_path=output_path,
                batch_size=2,
                max_content_chars=200,
            )

            self.assertEqual((classified, unreadable), (2, 1))
            self.assertEqual(list(state.groups), ["Object detection"])
            documents_in_group = state.groups["Object detection"].documents
            self.assertEqual([d.title for d in documents_in_group], ["First", "Second"])
            self.assertEqual(documents_in_group[0].year, 2020)
            self.assertEqual(documents_in_group[0].paper_link, "https://huggingface.co/papers/2005.12345")
            self.assertEqual(documents_in_group[0].paper_id, "2005.12345")
            self.assertIsNone(documents_in_group[1].year)
            self.assertIsNone(documents_in_group[1].paper_link)
            self.assertIsNone(documents_in_group[1].paper_id)
            self.assertEqual(documents_in_group[0].techniques, ["encoder-decoder transformer", "object queries"])
            self.assertEqual(documents_in_group[0].domains, ["computer vision"])
            self.assertEqual(documents_in_group[0].tasks, ["object detection"])
            self.assertEqual(len(state.read_errors), 1)
            self.assertEqual(state.read_errors[0].local, str(missing_path))
            self.assertEqual(fake.calls[0][2], [])
            self.assertEqual(fake.calls[1][2], ["Object detection"])

            checkpoint = load_state(output_path, Path("test-config.yaml"))
            self.assertEqual(checkpoint, state)

            second_run = FakeClassifier()
            classified, unreadable = classify_manifest(
                documents,
                checkpoint,
                second_run,
                output_path=output_path,
                batch_size=2,
                max_content_chars=100,
            )
            self.assertEqual((classified, unreadable), (0, 1))
            self.assertEqual(second_run.calls, [])


if __name__ == "__main__":
    unittest.main()
