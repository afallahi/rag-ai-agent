from main.llm.ollama_client import OllamaClient
import pytest


def test_generate_answer():
    """Test that Ollama LLM client generates a valid answer."""

    llm = OllamaClient()

    if not llm.is_running():
        # Skip the test if Ollama is not running
        pytest.skip("Ollama is not running. Skipping LLM test.")

    prompt = (
        "You are a helpful assistant.\n\n"
        "Answer the question below using ONLY the context provided.\n\n"
        "Context:\n"
        "Python is a programming language known for its simplicity and readability.\n\n"
        "Question: What is Python known for?"
    )

    response = llm.generate_answer(prompt)

    assert isinstance(response, str), "Response should be a string"
    assert len(response.strip()) > 0, "Response should not be empty"
    assert "simplicity" in response.lower() or "readability" in response.lower() or "easy" in response.lower()
