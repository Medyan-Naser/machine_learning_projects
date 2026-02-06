from langchain_ollama import ChatOllama
from config.settings import settings
import re
import logging

logger = logging.getLogger(__name__)

class RelevanceChecker:
    def __init__(self):
        """Initialize the relevance checker with Ollama."""
        print("Initializing RelevanceChecker with Ollama...")
        self.model = ChatOllama(
            model="llama3.2",
            temperature=0.0,
            num_predict=10,
        )
        print("Ollama model initialized successfully.")
