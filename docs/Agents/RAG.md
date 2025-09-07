# Retrieval-Augmented Generation (RAG)

## Introduction
RAG is an AI framework that improves the accuracy of LLMs by combining them with an external knowledge base.  
It avoids retraining by retrieving relevant information dynamically and injecting it into the prompt.

---

## Why RAG?
- LLMs are powerful but limited to their training data.  
- Domain-specific queries (e.g., company policies) may be **inaccurate** if not in training.  
- RAG solves this by **retrieving external documents** and augmenting the query.

---

## RAG Workflow
1. **Prompt Encoding**
   - The input question is converted into a vector using a **question encoder**.
   - Example: BERT, GPT, or DPR encoders.
   - Token embedding + vector averaging are used for compact representations.

2. **Knowledge Base Preparation**
   - Domain documents are split into chunks.  
   - Each chunk is embedded into a vector and stored in a **vector database** (e.g., FAISS).  
   - Each embedding is linked to a **chunk ID**.

3. **Retrieval**
   - The system compares the query embedding with stored vectors.  
   - Distance metrics:
     - **Dot Product:** Considers direction + magnitude.  
     - **Cosine Similarity:** Considers only angular similarity.  
   - Top **K most relevant chunks** are selected.

4. **Augmented Query Creation**
   - The retrieved context chunks are combined with the original query.  
   - This creates an **augmented input** for the LLM.

5. **Generation**
   - The LLM uses the augmented query to generate a context-aware response.  
   - Example: Using BART or GPT-based models.

---

## Encoders and FAISS
- **Context Encoder:** Converts documents/passages into embeddings.  
- **Question Encoder:** Converts queries into embeddings for comparison.  
- **FAISS:** A Facebook AI library for fast similarity search over high-dimensional vectors.  

Workflow:
- Encode documents with the context encoder → Store in FAISS index.  
- Encode question with question encoder → Search in FAISS index.  
- Retrieve nearest embeddings → Use them as context.  

---

## Example: Company Policy
- Input question: "What is the mobile policy?"  
- Documents: Company HR policy split into 7 chunks.  
- FAISS retrieves top 3 relevant chunks.  
- LLM generates a precise, policy-specific answer.

---

## Key Takeaways
- **Retriever + Generator** = RAG.  
- Encoders convert text to embeddings for efficient search.  
- FAISS enables fast similarity search across large corpora.  
- RAG ensures **accurate, domain-specific responses** without retraining.  
