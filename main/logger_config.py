import logging

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Suppress noisy loggers
    noisy_loggers = ["httpx", "httpcore", "ollama", "langchain_ollama"]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)
