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

