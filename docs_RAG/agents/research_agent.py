from langchain_ollama import ChatOllama
from typing import Dict, List
from langchain.schema import Document
from config.settings import settings
import json

class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with Ollama ChatOllama.
        """
        print("Initializing ResearchAgent with Ollama...")
        self.model = ChatOllama(
            model="llama3.2",
            temperature=0.3,
            num_predict=300,
        )
        print("Ollama model initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        print(f"ResearchAgent.generate called with question='{question}' and {len(documents)} documents.")

        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        prompt = self.generate_prompt(question, context)
        print("Prompt created for the LLM.")

        try:
            print("Sending prompt to the model...")
            response = self.model.invoke(prompt)
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        try:
            llm_response = response.content.strip()
            print(f"Raw LLM response:\n{llm_response}")
        except (IndexError, KeyError, AttributeError) as e:
            print(f"Unexpected response structure: {e}")
            llm_response = "I cannot answer this question based on the provided documents."

        draft_answer = self.sanitize_response(llm_response) if llm_response else "I cannot answer this question based on the provided documents."

        print(f"Generated answer: {draft_answer}")

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }
