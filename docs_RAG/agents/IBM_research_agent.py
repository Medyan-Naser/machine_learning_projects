from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials, APIClient
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import json

credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )
client = APIClient(credentials)


class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with the IBM WatsonX ModelInference.
        """
        # Initialize the WatsonX ModelInference
        print("Initializing ResearchAgent with IBM WatsonX ModelInference...")
        self.model = ModelInference(
            model_id="meta-llama/llama-3-2-90b-vision-instruct", 
            credentials=credentials,
            project_id="skills-network",
            params={
                "max_tokens": 300,            # Adjust based on desired response length
                "temperature": 0.3,           # Controls randomness; lower values make output more deterministic
            }
        )
        print("ModelInference initialized successfully.")
