from unittest.mock import MagicMock, patch

import main


def test_cli_single_question(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text("Title,Date,Rating,Review\nGood,2024-01-01,5,Tasty pizza\n")

    fake_pipeline = MagicMock()
    fake_pipeline.add_documents.return_value = 1
    fake_pipeline.get_context.return_value = "Good Tasty pizza"
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = "Pizza is tasty."

    with patch("main.Settings"), patch("main.RAGPipeline", return_value=fake_pipeline), patch(
        "main.OllamaLLM"
    ), patch("main.ChatPromptTemplate") as prompt:
        prompt.from_template.return_value.__or__.return_value = fake_chain
        code = main.main(["--data", str(csv), "-q", "How is the pizza?", "-k", "2"])

    assert code == 0
    fake_pipeline.add_documents.assert_called_once()
    fake_pipeline.get_context.assert_called_once_with("How is the pizza?", k=2)
    fake_chain.invoke.assert_called_once_with(
        {"reviews": "Good Tasty pizza", "question": "How is the pizza?"}
    )


def test_cli_reports_missing_data_file(capsys):
    with patch("main.Settings"), patch("main.RAGPipeline"):
        code = main.main(["--data", "/does/not/exist.csv", "-q", "hi"])
    assert code == 1
    assert "not found" in capsys.readouterr().err
