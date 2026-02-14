# AI Agents Overview

## What Are AI Agents?
AI Agents are systems that combine Large Language Models (LLMs) with external tools, data sources, and frameworks to accomplish tasks.  
Unlike standalone LLMs, agents can:

- Retrieve external knowledge
- Use memory and reasoning steps
- Interact with APIs, databases, or company-specific policies

This makes them more adaptable to **real-world, domain-specific tasks**.

---

## Docs Map
- `RAG.md` (Retrieval-Augmented Generation)
- `langchain.md` (LangChain + LangGraph + agent types)
- `workflow-patterns.md` (Reflection, Routing, Parallelization)
- `crewai.md` (CrewAI framework)
- `ag2-autogen.md` (AG2 / AutoGen framework)

## Why RAG and LangChain?
- **RAG (Retrieval-Augmented Generation):** Improves accuracy by injecting external knowledge into LLM outputs.  
- **LangChain:** A framework that orchestrates LLMs, retrieval systems, and external tools into modular pipelines.  

Together, they form the foundation of **AI Agents**: intelligent, flexible systems that can reason and act with up-to-date knowledge.

---

## Core Components of AI Agents
1. **LLM (Generator):** Produces natural language responses.  
2. **Retriever / Knowledge Base:** Finds relevant context (e.g., company policies, documents).  
3. **Framework (LangChain):** Manages the flow between prompts, retrieval, and generation.  

---

## Workflow Patterns
Agentic systems often combine multiple workflow patterns:

- **Reflection**: critique + revise loops to improve quality
- **Routing**: dispatch inputs to specialized handlers
- **Parallelization**: run independent subtasks concurrently and merge results

See: `workflow-patterns.md`.

---

## Key Takeaways
- LLMs alone struggle with **domain-specific knowledge** and **stale training data**.  
- RAG adds retrieval to reduce hallucinations and improve factual accuracy.  
- LangChain provides the **infrastructure** for building AI agents that integrate multiple components.  
