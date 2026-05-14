from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RetrieverBuilder:
    def __init__(self):
        """Initialize the retriever builder with embeddings."""
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
        }

        watsonx_embedding = WatsonxEmbeddings(
            model_id="ibm/slate-125m-english-rtrvr-v2",
            url="https://us-south.ml.cloud.ibm.com",
            project_id="skills-network",
            params=embed_params
        )
        self.embeddings = watsonx_embedding
        
