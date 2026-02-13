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

### The `@tool` Decorator

The recommended way to create custom tools is using the `@tool` decorator. This decorator simplifies tool creation by wrapping any function into a tool that implements the Tool Interface.

```python
from langchain_core.tools import tool
import re

@tool
def add_numbers(inputs: str) -> dict:
    """
    Adds a list of numbers provided in the input string.
    
    Parameters:
    - inputs (str): A string containing numbers to be extracted and summed.
    
    Returns:
    - dict: A dictionary with key "result" containing the sum.
    
    Example Input: "Add the numbers 10, 20, and 30."
    Example Output: {"result": 60}
    """
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    result = sum(numbers)
    return {"result": result}
```

**Key Points:**
- The docstring is critical—it tells the LLM when and how to use the tool
- Input/output types should be clearly defined
- The function name becomes the tool name

### Popular Built-in Tools

| Tool Name               | Description                                      |
|-------------------------|--------------------------------------------------|
| `WikipediaQueryRun`     | Search Wikipedia for factual information         |
| `GoogleSearchRun`       | Perform web searches using Google's API          |
| `PythonREPLTool`        | Execute Python code in a safe environment        |
| `OpenWeatherMapQueryRun`| Fetch real-time weather data                     |
| `YouTubeSearchTool`     | Search for YouTube videos                        |
| `DuckDuckGoSearchRun`   | Privacy-focused web search                       |
| `BingSearchRun`         | Microsoft Bing web search                        |

### Dataset Caching Tools

When building complex data-driven agents, efficiently managing datasets in memory is essential. Since LLMs communicate via text, sending entire datasets in each response wastes tokens and context window space.

**Solution:** Create a global cache that stores DataFrames after they're first loaded.

**Benefits:**
1. **Reduces token usage** – Reference datasets by name rather than content
2. **Improves performance** – Load data only once
3. **Maintains availability** – Datasets persist between different tool calls

**Metadata Tools:** Provide dataset summaries with key statistical information, giving the agent a quick overview without transferring entire content. These tools examine CSV structure and return metadata about available data.

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

## Generative AI vs Agentic AI

- **Generative AI** – Reactive system that creates content (text, images) based on prompts.  
- **Agentic AI** – Proactive system that uses prompts to pursue goals autonomously.

---

## LangGraph

LangGraph is an advanced framework for building **stateful, multiagent applications**. It extends LangChain's capabilities by enabling complex workflows with branching, looping, and persistent state.

### Core Concepts

| Component | Description |
|-----------|-------------|
| **Nodes** | Functions that perform computation and may modify state |
| **Edges** | Define execution flow between nodes, passing updated state |
| **State** | Shared memory that persists across nodes (inputs, intermediate values, outputs) |
| **Conditional Edges** | Enable dynamic routing based on runtime decisions |

### Key Capabilities

- **Looping & Branching** – Dynamic decision-making at runtime  
- **State Persistence** – Maintains context over long interactions  
- **Human-in-the-Loop** – Pause for human input and resume execution  
- **Time Travel** – Debug by replaying past states  
- **Workflow Visualization** – View graphs as Mermaid diagrams

### Building a LangGraph Application

1. Create a `StateGraph` object  
2. Add nodes (functions that process state)  
3. Connect nodes with edges  
4. Set an entry point  
5. Compile into a runnable application  
6. Invoke with initial state

---

## Agent Types

### Reflection Agents

Reflection agents iteratively improve outputs through internal critique using a feedback loop.

**Components:**
- **Generator** – Produces initial content  
- **Reflector** – Provides critical feedback (often role-played as a teacher/critic)

**Process:** Generate → Reflect → Revise → Repeat until satisfactory.

Agent state is defined using `MessageGraph`, tracking conversation and accumulating messages across iterations. Graph construction involves defining nodes, connecting with edges, setting entry points, and using router nodes for dynamic decisions.

### Reflexion Agents

Reflexion agents extend reflection by adding **external grounding with citations and real-time data**.

**Three-Step Cycle:**

| Step | Description |
|------|-------------|
| **Draft** | Generate answer and propose search queries/tool calls |
| **Execute Tools** | Run queries (e.g., web search), add results to context |
| **Revise** | Analyze draft + fetched info, list missing/incorrect parts, add references |

**Key Differences from Reflection:**
- Uses external tools (e.g., Tavily search) for real-time data  
- Produces structured schema-based output (response, critique, query fields)  
- Includes citations and references to support claims  
- **Responder** generates initial structured output → **Revisor** refines with tool outputs

**Schemas:** `AnswerQuestion` and `Reflection` schemas capture answers, flag missing details, and generate queries.

### ReAct Agents

ReAct (Reason + Act) agents interleave thinking and action in a single workflow.

- Alternates between **internal reasoning** (chain-of-thought) and **actions** (tool/function calls)  
- Each cycle: decide action → execute → reason on updated state  
- No separate reflector step—reasoning and acting are integrated

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