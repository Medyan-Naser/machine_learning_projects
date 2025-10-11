# XLNet: Generalized Autoregressive Pretraining for Language Understanding

**XLNet** is a Transformer-based language model developed by Google and Carnegie Mellon University in 2019.  
It was designed to **combine the strengths of BERT and GPT**, fixing limitations in BERT’s training approach while keeping bidirectional context.

---

## 1. Why XLNet Was Created

**BERT’s limitations:**
- Uses **Masked Language Modeling (MLM)** — predicts masked words independently.
- Sees **[MASK] tokens** during pretraining, which never appear during fine-tuning.
- Can’t capture full **dependency between masked tokens**.

**XLNet’s solution:**
- Removes `[MASK]` tokens entirely.
- Uses a **Permutation Language Modeling (PLM)** objective to capture bidirectional context *without masking*.
- Builds on **Transformer-XL** to handle longer sequences efficiently.

---

## 2. Core Idea: Permutation Language Modeling

Instead of masking tokens, XLNet predicts tokens in **random orderings** of a sentence.  
This allows the model to learn from **both left and right context** during training.

**Example:**

Sentence:  
The cat sat on the mat.

XLNet can train on multiple permutations:  
- Order 1: The → cat → sat → on → the → mat  
- Order 2: mat → the → cat → on → sat → the  

Each token gets predicted with both left and right neighbors in some permutation —  
giving **bidirectional understanding** like BERT, but through an **autoregressive** approach.

---

## 3. Architecture Highlights

- **Based on Transformer-XL:**  
  - Adds **segment recurrence** (memory of previous text).  
  - Uses **relative positional encoding** to track token relationships.  
  - Handles longer context than BERT or GPT.

- **Two-stream attention:**  
  - One stream for content, another for prediction — allowing flexible permutation training.

---

## 4. How It Differs

| Feature | **BERT** | **GPT** | **XLNet** |
|----------|-----------|----------|------------|
| Objective | Masked LM (autoencoder) | Left-to-right LM | Permutation LM |
| Direction | Bidirectional | Unidirectional | Bidirectional |
| [MASK] Tokens | Yes | No | No |
| Long Context Support | No | Limited | Yes (Transformer-XL) |
| Pretrain–Finetune Gap | High | Low | Low |

**In short:**  
- XLNet keeps BERT’s bidirectional understanding.  
- Keeps GPT’s autoregressive nature.  
- Adds long-context modeling from Transformer-XL.

---

## 5. Strengths and Use Cases

**Strengths:**
- Models dependencies between tokens better than BERT.  
- Handles longer text sequences efficiently.  
- No `[MASK]` token mismatch during fine-tuning.

**Use cases:**
- Question answering  
- Text classification  
- Natural language inference  
- Long document understanding

---

## 6. Limitations

- More complex and computationally expensive training.  
- Later models (e.g., RoBERTa, T5) achieved similar or better performance with simpler objectives.

---

## 7. Summary

**XLNet bridges BERT and GPT — combining bidirectional context with autoregressive prediction.**  
It was created to overcome BERT’s masked language modeling issues and to handle longer dependencies effectively.
