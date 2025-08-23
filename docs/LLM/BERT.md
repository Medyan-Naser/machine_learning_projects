# BERT: Bidirectional Encoder Representations from Transformers

BERT is an **encoder-only Transformer model** designed to deeply understand language. It differs from decoder-based models like GPT by focusing on context from **both directions** rather than generating text.

---

## 1. Architecture Overview

- **Transformer Encoder Only:** Uses stacked self-attention layers.  
- **Bidirectional Attention:** Each token can attend to all other tokens in the input sequence.  
- **Input Representation:**
    - Tokens are represented using **token embeddings**, **segment embeddings**, and **positional embeddings**.
    - Input format includes `[CLS]` at the start (for classification tasks) and `[SEP]` to separate sentences.

**Example:**  
Input: `"The cat sat on the mat."`  
Tokenized: `[CLS] The cat sat on the mat [SEP]`

---

## 2. Training Objectives

BERT uses two main objectives to learn rich contextual embeddings:

### Masked Language Modeling (MLM)
- **Goal:** Predict randomly masked words in a sentence.  
- **How it works:**
    1. Randomly mask 15% of input tokens.  
    2. Model predicts the original tokens from context.  
- **Strength:** Learns bidirectional context, understanding the relationship between words and surrounding text.  

**Example:**  
Input: `"The [MASK] sat on the mat."`  
Prediction: `[MASK] → cat`

---

### Next Sentence Prediction (NSP)
- **Goal:** Understand relationships between sentences.  
- **How it works:**
    1. Model is fed pairs of sentences.  
    2. Task: predict if the second sentence **follows logically** after the first.  
- **Strength:** Helps with tasks requiring sentence-level understanding like QA and entailment.  

**Example:**  
- Sentence A: `"The cat sat on the mat."`  
- Sentence B: `"It was purring happily."` → label: **IsNext**  
- Sentence B: `"The stock market crashed yesterday."` → label: **NotNext**

---

## 3. Model Output

- **Contextual Embeddings:** Each token gets a vector capturing its meaning in context.  
- **[CLS] Token:** Aggregated representation of the entire sequence, often used for classification tasks.  
- **[SEP] Token:** Marks sentence boundaries, aiding NSP and sentence-level tasks.

---

## 4. Strengths

- Captures **deep contextual relationships** bidirectionally.  
- Excellent for **understanding language**, not generating it.  
- Can be fine-tuned for a wide variety of NLP tasks with relatively little additional data.

---

## 5. Common Use Cases

- **Text Classification:** Sentiment, spam detection, intent recognition  
- **Named Entity Recognition (NER):** Extracting names, locations, and organizations  
- **Question Answering (extractive QA):** Finding answers in passages  
- **Semantic Search:** Matching queries with relevant documents  

---

## 6. Key Variants

- **BERT-Base:** 12 layers, 768 hidden size, 12 attention heads  
- **BERT-Large:** 24 layers, 1024 hidden size, 16 attention heads  
- **DistilBERT:** Smaller, faster, with ~40% fewer parameters, retaining 97% performance  

---

## 7. Notes

- Unlike decoder models, BERT **cannot generate text** naturally.  
- Pretraining is **resource-intensive**, but fine-tuning is efficient.  
- Masked token prediction and NSP together enable **strong transfer learning** across NLP tasks.
