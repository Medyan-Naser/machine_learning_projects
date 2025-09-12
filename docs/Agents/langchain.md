# LangChain

## Introduction
LangChain is a framework for building AI agents that integrate:
- LLMs (generators)  
- Retrieval systems (RAG, FAISS, vector DBs)  
- Tools (APIs, calculators, databases)  

It provides modular components to design workflows where an LLM can **reason, retrieve, and act**.

---

## Core Concepts
1. **Chains**
   - A sequence of calls linking LLMs, retrievers, and tools.
   - Example: Prompt → RAG Retrieval → LLM Generation.

2. **Agents**
   - Decision-making entities powered by LLMs.
   - They select which tools/chains to use based on the task.

3. **Memory**
   - Mechanism to store previous interactions.
   - Supports context continuity across multiple queries.

4. **Tools**
   - External capabilities (APIs, DB queries, search engines).
   - LangChain integrates them directly into agent workflows.

---

## Why LangChain?
- LLMs alone = isolated text generators.  
- RAG improves factual accuracy but is still static.  
- LangChain orchestrates everything into **dynamic pipelines**:
  - Query external APIs
  - Retrieve documents
  - Use reasoning steps
  - Generate structured outputs

---

## Example Workflow
1. User asks: *"Summarize the company’s mobile policy."*  
2. LangChain pipeline:  
   - Encode query → Search FAISS index → Retrieve top chunks.  
   - Augment prompt with retrieved context.  
   - Pass to LLM → Generate answer.  
3. Output: Accurate, policy-specific summary.

---

## Key Takeaways
- LangChain makes it easy to build **production-ready AI agents**.  
- It provides abstractions for **chains, agents, memory, and tools**.  
- Best used with RAG to enhance **retrieval + reasoning**.  
