from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with Ollama embeddings."""
        print("Initializing RetrieverBuilder with Ollama embeddings...")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )
        print("Ollama embeddings initialized successfully.")
        
  