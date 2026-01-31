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
9. **Tools** – External utilities that models can call (see Tools section below).  
10. **Toolkits** – Collections of related tools for a common purpose or integration.  
11. **LangChain Core** – Expression language and abstractions.  
12. **LangChain Community** – Third-party integrations.  

---

## Tools

Tools are utilities designed to be called by a model—inputs are structured so models can generate them, and outputs are passed back to the model. They perform actions like searching the web, querying databases, or executing code.

### Ways to Initialize Tools

1. **Built-in Tools** – Use pre-built tools like `WikipediaQueryRun` for common tasks.  
2. **`load_tools` Function** – Load multiple tools at once (e.g., `wikipedia`, `serpapi`, `llm-math`).  
3. **Custom Tools with `@tool` Decorator** – Wrap any function as a tool the LLM can invoke. This gives you manual control over tool inputs and validation.  
4. **Tools as OpenAI Functions** – Convert tools using `convert_to_openai_function`, or use `bind_functions`/`bind_tools` to bind tools to OpenAI chat models.

### Popular External Tools & Toolkits

- **Wikipedia** – Fetch summaries and article information.  
- **Search Engines** – Bing, Google, DuckDuckGo for real-time search results.  
- **APIs** – Weather data, financial data, and more.

---

## Prompting & In-Context Learning
LangChain supports **prompt engineering** to guide LLM behavior.  

- **Zero-Shot** – Only task instructions.  
- **One-Shot** – One example included.  
- **Few-Shot** – Multiple examples for generalization.  
- **In-Context Learning** – Model adapts from examples in prompt.  
- **Chain-of-Thought (CoT)** – Step-by-step reasoning.  
- **Self-Consistency** – Generate Multiple outputs → pick most consistent or accurate.

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
      - Break large documents into smaller chunks for retrieval. Then combine the small chunks intoa larger chunk aiming at a specific size. also include some overlap.
      - Text is split can be at a specific, character, word, sentence, or token. (common ones are bu paragraph change, line change, space, ...)
      - Chunk size is measured by counting characters, words, tokens, or metrics
      - Examples:  
         - `CharacterTextSplitter` → recursive character-based splitting.  
         - `MarkdownHeaderTextSplitter` → split by markdown headers.
      - **Recursive Character Text Splitter**
         - Uses recursion to split large text until chunks fit size.  
         - Default separators: paragraph (`\n\n`), new line (`\n`), word, char.
         - It processes the large text by attempting to split it by the first character. If the first split results in chunks that are still too large, it moves to the next character and attempts to split by it. This process continues through the list of characters until the chunks are less than the specified chunk size.
         - Example: split by paragraph → then if still above chunk size then by sentence (keep using the list of seperators if needed) → merge chunks under limit if possible.  
      - **Code Text Splitter**
         - Based on recursive splitter, specialized for source code.  
         - Supports multiple languages.  
         - Requires specifying `language` param.  
      - **Markdown Header Text Splitter**
         - Splits markdown by headers, preserving structure.  
         - Example: groups content under headers like `bar`, `baz`.  

4. **Embeddings**  
      - Convert document chunks into vector representations capturing semantic meaning.  
      - Example: Embedding model from WatsonX.ai.  

5. **Vector Databases**  
      - Store embeddings for fast similarity search.  
      - Example: **Chroma DB** and **FIASS DB** for storing embeddings and retrieving nearest neighbors.  

6. **Retrievers**
      - interface to query the DB (or other sources)
      - Extract relevant chunks from vector stores.  
      - Types:  
         - **Vector Store Retriever**
            - Similarity search.
            - Uses the search methods implemented by a vector store, like similarity search and MMR (Maximum marginal relevance), to query the texts in the vector store.
                - MMR in vector stores is a technique used to balance the relevance and diversity of retrieved results. It selects documents that are both highly relevant to the query and minimally similar to previously selected documents. This approach helps to avoid redundancy and ensures a more comprehensive coverage of different aspects of the query.
         - **Parent Document Retriever** 
            - Handles small chunk embeddings while preserving larger context.
            - Searches within parent chunks
            - Uses two text splitters: child splitter for embeddings, parent splitter for retrieval.
            - During retrieval, it first fetches the small chunks but then looks up the parent IDs for them and returns those larger documents.
         - **Self-Query Retriever** 
            - Uses metadata filters along with semantic search.  
            - Converts a query into two parts: a semantic string and a metadata filter.  
            - Enables retrieval based on both text and metadata, e.g., movies with rating > 8.5.  
         - **Multi-Query retriever**
            - Builds on vector retrievers by generating multiple query variations using an LLM.  
            - Overcomes differences in query wording or embedding limitations.  
            - Retrieves documents for each query and returns the **unique union** for richer results. 

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
- **Memory** allows chains to **store and recall context** across interactions. So to remember old prompts in the chat.
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

## Natural Language Interfaces for Data

LangChain enables querying data using natural language, making data analysis accessible without deep technical skills.

### Pandas DataFrame Agent

- Analyze and visualize data by asking natural language questions.  
- The agent generates Python code that directly interacts with your DataFrame—filtering, aggregating, and visualizing.  
- Created with `create_pandas_dataframe_agent`.

### SQL Agent

- Translates natural language queries to SQL and retrieves results from databases.  
- Setup: Create a Python virtual environment → install LangChain and LLM libraries → launch SQL server → build a database connector.

### Best Practices

- Use **sandboxed environments** for code execution.  
- Design **clear prompts** and **validate LLM analysis** with human expertise.  
- Iteratively refine queries for accurate results.

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