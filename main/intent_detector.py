from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from main.config import Config


class IntentDetector:
    def __init__(self, model=None):
        self.model = model or ChatOllama(
            model=Config.OLLAMA_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            streaming=True
        )
        self.intents = [
            "greeting",
            "thanks",
            "goodbye",
            "help",
            "chitchat",
            "question",
            "unclear"
        ]
        self.prompt_template = PromptTemplate.from_template(
            "Classify the following user message into one of these intents: {intents}.\n\n"
            "User Message: {text}\nIntent:"
        )

    def detect(self, text: str) -> str:
        prompt = self.prompt_template.format(
            text=text.strip(),
            intents=", ".join(self.intents)
        )

        # Stream the response and accumulate content chunks
        response_chunks = self.model.stream(prompt)
        full_response = ""
        for chunk in response_chunks:
            full_response += chunk.content

        intent = full_response.strip().lower()
        if intent not in self.intents:
            intent = "unclear"
        return intent
