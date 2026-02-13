# **DocChat** 📝🤖  
🚀 **AI-powered Multi-Agent RAG system for intelligent document querying with fact verification**  

![DocChat Cover Image](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/zSuj0yrlvjcVkkbW4frkNA/docchat-landing-page.png)

---

## **📌 Overview**  

**DocChat** is a **multi-agent Retrieval-Augmented Generation (RAG) system** designed to help users query **long, complex documents** with **accurate, fact-verified answers**. Unlike traditional chatbots like **ChatGPT or DeepSeek**, which **hallucinate responses and struggle with structured data**, DocChat **retrieves, verifies, and corrects** answers before delivering them.  

💡 **Key Features:**  
✅ **Fully Local AI** – Powered by **Ollama** (no API costs, complete privacy, offline capability)  
✅ **Multi-Agent System** – A **Research Agent** generates answers, while a **Verification Agent** fact-checks responses.  
✅ **Hybrid Retrieval** – Uses **BM25 and vector search** to find the most relevant content.  
✅ **Handles Multiple Documents** – Selects the most relevant document even when multiple files are uploaded.  
✅ **Scope Detection** – Prevents hallucinations by **rejecting irrelevant queries**.  
✅ **Fact Verification** – Ensures responses are accurate before presenting them to the user.  
✅ **Web Interface with Gradio** – Allowing seamless document upload and question-answering.  

---

## **🎥 Demo Video**  

📹 **[Click here to watch the DocChat demo](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/zyARt3f3bnm5T-6C4AE3mw/docchat-demo.mp4)**  
*(Opens in a new tab)*

---

## **🛠️ How DocChat Works**  

### **1️⃣ Query Processing & Scope Analysis**  
- Users **upload documents** and **ask a question**.  
- DocChat **analyzes query relevance** and determines if the question is **within scope**.  
- If the query is **irrelevant**, DocChat **rejects it** instead of generating hallucinated responses.  

### **2️⃣ Multi-Agent Research & Retrieval**  
- **Docling** parses documents into a structured format (Markdown, JSON).  
- **LangChain & ChromaDB** handle **hybrid retrieval** (BM25 + vector embeddings).  
- Even when **multiple documents** are uploaded, **DocChat finds the most relevant sections** dynamically.  

### **3️⃣ Answer Generation & Verification**  
- **Research Agent** generates an answer using retrieved content.  
- **Verification Agent** cross-checks the response against the source document.  
- If **verification fails**, a **self-correction loop** re-runs retrieval and research.  

### **4️⃣ Response Finalization**  
- **If the answer passes verification**, it is displayed to the user.  
- **If the question is out of scope**, DocChat informs the user instead of hallucinating.  

---



