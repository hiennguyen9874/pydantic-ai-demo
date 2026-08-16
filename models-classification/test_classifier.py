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
                group_name="Vision encoder",
                new_group_description="Image encoders trained for visual understanding tasks.",
            )
        return GroupDecision(group_name="vision encoder")


class ClassifierTests(unittest.TestCase):
    def test_local_openai_compatible_agent_constructs_without_a_network_request(self) -> None:
        classifier = AgentDocumentClassifier(
            "openai:gpt-5.6-luna",
            base_url="http://localhost:8317/v1",
            api_key="test-key",
        )
        self.assertEqual(classifier.agent.name, "model_document_classifier")

    def test_manifest_is_the_expected_list_shape(self) -> None:
        documents = read_manifest(Path(__file__).with_name("config.yaml"))
        self.assertEqual(len(documents), 149)
        self.assertEqual(documents[0].title, "ALIGN")

    def test_batches_checkpoint_groups_and_retry_unreadable_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "first.md"
            second_path = root / "second.md"
            first_path.write_text("first architecture", encoding="utf-8")
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
                max_content_chars=100,
            )

            self.assertEqual((classified, unreadable), (2, 1))
            self.assertEqual(list(state.groups), ["Vision encoder"])
            self.assertEqual([d.title for d in state.groups["Vision encoder"].documents], ["First", "Second"])
            self.assertEqual(len(state.read_errors), 1)
            self.assertEqual(state.read_errors[0].local, str(missing_path))
            self.assertEqual(fake.calls[0][2], [])
            self.assertEqual(fake.calls[1][2], ["Vision encoder"])

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
