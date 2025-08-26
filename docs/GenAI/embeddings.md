## Converting Words to Features

Before using text in an NLP or GenAI model, it must be converted into numerical features that a neural network can process.

### 1. One-Hot Encoding
- Represents each token as a binary vector the size of the vocabulary.  
- Only the position for that token is `1`; all others are `0`.  
- Dimension = **vocabulary size**; sparse and high-dimensional.  
- **Note:** Mostly replaced by embeddings in modern NLP.

### 2. Bag-of-Words (BoW)
- Aggregates one-hot vectors for all tokens in a document by summing or averaging.  
- Ignores word order; useful for basic text classification.

### 3. Embedding
- Dense, low-dimensional vector for each token (learned during training).  
- Stored in an **embedding matrix**:  
    - Rows = tokens  
    - Columns = embedding dimensions  
- Captures semantic relationships between words.  
- When using BoW as input, the hidden layer output is the sum of embeddings for present tokens.  
- **Modern NLP approach:** Embeddings are the standard replacement for one-hot encoding.

### 4. Embedding Bag
- Directly takes token indexes and outputs the **sum or average** of their embeddings in one step.  
- More efficient than manually summing/averaging.  
- Supports an offset parameter to indicate document boundaries when input is a flat list of token IDs.

---

## Word2Vec and the Continuous Bag of Words (CBOW) Model

### 1. Overview of Word2Vec
- **Word2Vec** stands for *word to vector*.
- Produces **word embeddings** — dense numerical representations capturing the meaning and relationships of words.
- Embeddings reflect semantic relationships:
    - *king* − *man* + *woman* ≈ *queen*
- Improves NLP tasks by replacing randomly initialized embeddings with trained, meaningful vectors.

### 2. How Word2Vec Works
- Implemented using a simple neural network:
    - **Input layer**: size = vocabulary size (one-hot representation of a word).
    - **Embedding layer**: size = chosen embedding dimension, which is the word vector dimensions.
    - **Output layer**: size = vocabulary size (predicts target or context words).
- The model learns by adjusting:
    - **W** (hidden layer weights) → embeddings.
    - **W′** (output layer weights).
- After training, embeddings place semantically similar words closer together in vector space.

---

## Continuous Bag of Words (CBOW) Model

### 1. Concept
- **Goal**: Use **context words** to predict a **target word**.
- **Context window**: Defines how many words before and after the target are considered.
    - Example (window size = 1): In *she exercises every morning*:
        - Context: *she*, *every* → Target: *exercises*.
        - Context: *exercises*, *morning* → Target: *every*.
- **Input representation**:
    - Context words converted to one-hot vectors.
    - Combined into a bag-of-words vector (order ignored).
- **Output**:
    - Softmax over vocabulary — highest probability corresponds to the predicted target.

### 2. Architecture
1. **Embedding layer**: Maps context words to embeddings and averages them.
2. **Hidden layer**: Holds the learned embeddings (W matrix).
3. **Output layer**: Predicts the likelihood of each word being the target.
4. **Training objective**: Maximize the probability of the correct target word given its context.

### 3. Key Takeaways
- **CBOW** predicts a target word from surrounding context words.
- Embeddings learned by CBOW capture semantic relationships and can be reused for other NLP tasks.

---

## Skip-Gram Model

### 1. Concept
- **Reverse of CBOW**: Predicts **context words** from a given **target word**.
- For each target word, the model predicts surrounding words within the context window.
    - Example (window size = 1): In *she exercises every morning*:
        - Target: *exercises* → Predict *she* (preceding) and *every* (following).
        - Target: *every* → Predict *exercises* and *morning*.
- **Prediction style**:
    - Instead of predicting all context words at once, Skip-Gram predicts **one context word at a time**.
    - Breaks a multi-word context into smaller, easier prediction tasks.

### 2. Architecture
1. **Input layer**: One-hot vector of the target word.
2. **Embedding layer**: Maps the target word to its dense vector.
3. **Output layer**: Predicts one context word from the target.
4. **Training**: For each target-context pair, maximize the probability of predicting the correct context word.

### 3. Key Takeaways
- Skip-Gram is better for capturing rare word representations because it trains on more individual (target, context) pairs.
- Works well for large datasets and can produce high-quality embeddings.

---

## Pre-Trained Word Embeddings

### 1. Concept
- Instead of training embeddings from scratch, you can use **pre-trained vectors** built on massive corpora.
- Example: **GloVe** (*Global Vectors for Word Representation*):
    - Trained on large datasets like Wikipedia or Common Crawl.
    - Captures broad semantic and syntactic relationships.

### 2. Benefits
- **Faster training**: Start with meaningful embeddings rather than random ones.
- **Better performance**: Leverages patterns learned from billions of words.
- Can be used **as-is** (frozen) or **fine-tuned** on your specific dataset.

### 3. Integration in NLP Models
- Load pre-trained vectors (e.g., GloVe) into your embedding layer.
- Optionally set `freeze = False` to allow fine-tuning during training.
- Particularly effective for tasks like text classification, sentiment analysis, and named entity recognition.

---

## GloVe (Global Vectors for Word Representation)

### 1. Overview
- **GloVe** is a word embedding method developed at Stanford.
- Learns embeddings from **global co-occurrence statistics** across a large corpus.
- Captures semantic relationships similar to Word2Vec:
    - *king* − *man* + *woman* ≈ *queen*

### 2. How It Works
- Builds a **co-occurrence matrix**: how often each word appears near another.
- Factorizes this matrix to learn embeddings where similarity reflects meaning.
- Unlike Word2Vec’s predictive training, GloVe is **count-based** and global.

### 3. Key Benefits
- Embeddings reflect both local and global word statistics.
- Pre-trained GloVe vectors (Wikipedia, Common Crawl) are widely available.
- Can be used directly or fine-tuned for NLP tasks.

---

## Summary
- **Word2Vec**: Framework for learning dense word embeddings.
- **CBOW**: Predicts a target word from its context; efficient for frequent words.
- **Skip-Gram**: Predicts context words from a target; better for rare words.
- **Pre-Trained Embeddings**: Use large-scale pre-learned vectors (like GloVe) to improve model accuracy and training speed.
- **GloVe**: Learns embeddings from global co-occurrence statistics, combining the strengths of count-based and predictive models.

---