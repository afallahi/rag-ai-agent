
from main.llm import llm_client

def test_generate_answer():
    """Test the LLM client generates a valid answer."""
    prompt = (
        "Context:\n"
        "Python is a programming language known for its simplicity.\n\n"
        "Question: What is Python known for?"
    )

    response = llm_client.generate_answer(prompt)

    assert isinstance(response, str)
    assert len(response.strip()) > 0
    assert "simplicity" in response.lower() or "easy" in response.lower()
