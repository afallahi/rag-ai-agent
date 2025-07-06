import streamlit as st

from pipeline import build_global_index, build_prompt
from main.llm.ollama_client import OllamaClient
from main.intent_detector import IntentDetector
from main.vector_store import faiss_indexer
from main.embedder import embedder

@st.cache_resource
def load_components():
    llm = OllamaClient()
    index = build_global_index(force=False)
    intent_detector = IntentDetector()
    return llm, index, intent_detector

llm, index, intent_detector = load_components()

st.set_page_config(page_title="Armstrong AI Agent", page_icon="")
st.title("Armstrong AI Assistant")
st.caption("Ask technical questions. I'll pull relevant context and answer using local LLM.")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Your question", placeholder="e.g. What is Fire Manager?")

def respond_to_query(query_text: str):
    intent = intent_detector.detect(query_text)

    if intent == "greeting":
        return "Hello! I'm your Armstrong assistant. Ask me a technical question and I’ll look it up for you."
    elif intent == "thanks":
        return "You're welcome! Let me know if you have more questions."
    elif intent == "goodbye":
        return "Goodbye! Feel free to come back with more questions anytime."
    elif intent == "help":
        return "I can help answer questions about Armstrong products. Ask me something specific!"
    elif intent == "vague":
        return "Could you please rephrase your question or ask something more specific?"
    elif intent == "empty":
        return None
    else:
        top_chunks = faiss_indexer.query_faiss_index(index, query_text, embedder.get_model(), k=4)
        if not top_chunks:
            return "Sorry, I couldn't find relevant information in the documents."

        max_score = max(score for _, score in top_chunks)
        if max_score < 0.2:
            return "I looked through the documents but didn't find anything helpful for that question."

        context = "\n\n".join(chunk for chunk, _ in top_chunks)
        prompt = build_prompt(context, query_text, st.session_state.history[-3:])  # limit history
        return llm.generate_answer(prompt)

if st.button("Submit") and query.strip():
    response = respond_to_query(query)
    if response:
        st.session_state.history.append((query, response))
        st.rerun()
    else:
        st.warning("Please enter a valid question.")


if st.button("Reset"):
    st.session_state.history.clear()
    st.rerun()

if st.session_state.history:
    st.markdown("---")
    st.markdown("### Chat History")
    for q, a in reversed(st.session_state.history):
        st.write(f"**You:** {q}")
        st.write(f"**Assistant:** {a}")
