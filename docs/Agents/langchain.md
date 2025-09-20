# LangChain

## Introduction
LangChain is an **open-source framework** for building AI applications with Large Language Models (LLMs).  
It provides a modular environment to connect **LLMs, retrieval systems (RAG), and tools** into production-ready workflows.  

It facilitates a **structured way to integrate LLMs** into use cases such as **NLP, chatbots, and data retrieval**.  
LangChain = **reason + retrieve + act**.

---

## Why LangChain?
- LLMs alone generate text but lack grounding.  
- Retrieval (RAG) improves factual accuracy.  
- **LangChain orchestrates reasoning + retrieval + tools** into pipelines.  

**Benefits**:  
- Simplifies integration of models like OpenAI, IBM, Google, Meta.  
- Enables workflow automation and content management.  
- Unlocks flexible AI apps (chatbots, summarization, Q&A, analytics).  

---

## Core Components
1. **Language Models (LMs)** – Foundation LLMs (IBM WatsonX, OpenAI, Google, Meta). Generate text from prompts.  
2. **Chat Models** – Optimized for conversations; handle dialogue and context flow.  
3. **Chat Messages** – Roles in chat:
      - *Human Message* → user input  
      - *AI Message* → model response  
      - *System Message* → instructions  
      - *Function Message* → call outcomes  
      - *Tool Message* → tool interaction  
   Each message = **role + content**.  
4. **Prompt Templates** – Structure prompts with placeholders for dynamic inputs.  
      - String Prompt Template (single format)  
      - Chat Prompt Template (multi-role messages)  
      - Example Selector – chooses best examples (Semantic Similarity, MMR, N-Gram).  
5. **Output Parsers** – Convert model output into structured formats (JSON, CSV, XML, Pandas).  
6. **Chains** – Sequences of LLM calls, retrievers, and tools.  
7. **Agents** – LLM-powered decision makers choosing actions.  
8. **Memory** – Keeps past interactions for continuity.  
9. **Tools** – External APIs, databases, search engines.  
10. **LangChain Core** – Expression language and abstractions.  
11. **LangChain Community** – Third-party integrations.  

---

## Prompting & In-Context Learning
LangChain supports **prompt engineering** to guide LLM behavior.  

- **Zero-Shot** – Only task instructions.  
- **One-Shot** – One example included.  
- **Few-Shot** – Multiple examples for generalization.  
- **In-Context Learning** – Model adapts from examples in prompt.  
- **Chain-of-Thought (CoT)** – Step-by-step reasoning.  
- **Self-Consistency** – Multiple outputs → pick most consistent.  

---

## Example Workflow
1. User asks: *"Summarize the company’s mobile policy."*  
2. Pipeline:  
      - Encode query → Search FAISS index → Retrieve top chunks  
      - Add retrieved context to prompt → Send to LLM  
      - Generate structured summary  

---

## Use Case: Content Management
A company managing blogs, videos, podcasts, and articles can use LangChain to:  
- Automate retrieval and summarization.  
- Classify and tag large content libraries.  
- Build agents for recommendations and distribution.  

---
## TODO move this section to RAG
## Documents for Building RAG Applications

LangChain provides **document tools** that simplify building **Retrieval Augmented Generation (RAG) applications**.  
RAG combines **retrieval of external data** with LLM reasoning to deliver grounded responses.

### Document Workflow in LangChain
1. **Document Object**  
      - Core container for data.  
      - Attributes:  
         - `page_content` → document text (string).  
         - `metadata` → extra info (e.g., `document_id`, filename, tags).  

2. **Document Loaders**  
      - Load data from 100+ sources (Airbyte, Unstructured, S3, websites, PDFs, code).  
      - Example: Web loader fetches content directly from a URL.  

3. **Text Splitters**  
      - Break large documents into smaller chunks for retrieval.  
      - Examples:  
         - `CharacterTextSplitter` → recursive character-based splitting.  
         - `MarkdownHeaderTextSplitter` → split by markdown headers.  

4. **Embeddings**  
      - Convert document chunks into vector representations capturing semantic meaning.  
      - Example: Embedding model from WatsonX.ai.  

5. **Vector Databases**  
      - Store embeddings for fast similarity search.  
      - Example: **Chroma** for storing embeddings and retrieving nearest neighbors.  

6. **Retrievers**  
      - Extract relevant chunks from vector stores.  
      - Types:  
         - **Vector Store Retriever** – similarity search.  
         - **Parent Document Retriever** – searches within parent chunks.  
         - **Self-Query Retriever** – advanced filtering with metadata. 

---

## Chains, Memory, and Agents

LangChain uses **chains, memory, and agents** to build dynamic applications that combine reasoning, context, and tool usage.  

### Chains
- A **chain** is a sequence of calls where the **output of one step becomes the input for the next**.  
- **Sequential Chains** create smooth flows:
      1. Chain 1 → Selects a famous dish for a given location.  
      2. Chain 2 → Generates the recipe for that dish.  
      3. Chain 3 → Estimates cooking time from the recipe.  
- Each chain is defined by:
      - **Prompt Template** (task instruction).  
      - **LLM Chain** (model + template).  
      - **Output Key** (where result is stored).  
- Sequential chains combine steps into a **unified process** with verbose tracing for debugging.

### Memory
- **Memory** allows chains to **store and recall context** across interactions.  
- Each chain:
      - Reads from memory → enhances input with past history.  
      - Writes to memory → saves current input/output for future use.  
- **ChatMessageHistory** stores:
      - *Human messages* (user inputs).  
      - *AI messages* (model responses).  
- Enables continuity in dialogue (e.g., remembering prior questions/answers).

### Agents
- **Agents** are dynamic systems where an LLM decides which actions to take.  
- Agents use **reasoning + tools** to fulfill user requests.  
- Unlike chains, agents can **choose actions dynamically** at runtime.  
- Tools integrated with agents:  
      - Search engines, databases, APIs, websites.  
- Example: **Pandas DataFrame Agent**  
      - Created with `create_pandas_dataframe_agent`.  
      - Transforms natural language queries into Python code.  
      - Executes on a DataFrame (e.g., "How many rows?" → returns 139).  
      - `verbose=True` shows model reasoning process.  

---

## Generative models
Generative models understand and capture the underlying patterns and data distribution to resemble the given data sets. Generative models are applicable in generating images, text, and music, augmenting data, discovering drugs, and detecting anomalies. 

Types of generative models are: 

- Gaussian mixture models (GMMs)
- Hidden Markov models (HMMs)
- Restricted Boltzmann machines (RBMs)
- Variational autoencoders (VAEs)
- Generative adversarial networks (GANs)
- Diffusion models

---