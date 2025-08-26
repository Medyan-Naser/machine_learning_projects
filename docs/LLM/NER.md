# NER: Named Entity Recognition

Named Entity Recognition (NER) is an **NLP task** that identifies and classifies entities in text into predefined categories like **Person, Organization, Location, Date, Monetary Value, and Percentages**.

---

## 1. Task Overview

- **Input:** Unstructured text.  
- **Output:** Structured labels for tokens corresponding to entities.  
- **Goal:** Extract key information to make text machine-readable.

**Example:**  
Sentence: `"Apple announced a partnership with Microsoft in San Francisco on August 25, 2025."`  
Output:  
- Apple → Organization  
- Microsoft → Organization  
- San Francisco → Location  
- August 25, 2025 → Date  

---

## 2. Approaches

- **Rule-Based Systems:** Early NER used regex patterns and dictionaries.  
- **Classical ML:** Algorithms like **CRFs** and **SVMs** using handcrafted features.  
- **Deep Learning:** RNNs, CNNs, and especially **Transformers (BERT, GPT)** dominate today.  

---

## 3. Model Output

- Each token is assigned a label:  
  - **B-PER, I-PER** → Beginning / Inside of a Person entity  
  - **B-ORG, I-ORG** → Organization  
  - **B-LOC, I-LOC** → Location  
  - **O** → Not an entity  

**Example:**  
Sentence: `"Barack Obama visited Paris."`  
Tags: `[B-PER, I-PER, O, B-LOC, O]`

---

## 4. Strengths

- Converts unstructured text into structured data.  
- Generalizes across multiple domains (finance, healthcare, legal, etc.).  
- Can be fine-tuned on domain-specific datasets.  

---

## 5. Common Use Cases

- **Information Extraction:** Pulling names, dates, and locations from documents.  
- **Search Engines:** Enhancing indexing and semantic retrieval.  
- **Business Intelligence:** Analyzing news, legal, or medical reports.  
- **Chatbots & QA:** Understanding user queries by recognizing key entities.  

---

## 6. Notes

- Performance depends heavily on **training data quality**.  
- Domain adaptation may require **fine-tuning** with domain-specific examples.  
- Pretrained transformer models (e.g., **BERT**) achieve **state-of-the-art** results in NER.
