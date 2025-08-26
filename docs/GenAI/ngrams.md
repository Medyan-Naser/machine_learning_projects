# Language Modeling with N-Grams

## Key Concepts
- **N-Gram**: A sequence of N consecutive words used to model language context.
- **Bi-Gram Model**: Predicts the next word using only the immediate previous word (context size = 1).
- **Tri-Gram Model**: Predicts the next word using the two previous words (context size = 2).
- **N-Gram Generalization**: Extends the concept to any arbitrary context size `N`.

## How It Works
1. **Conditional Probability**:
    - Bi-Gram: \( P(\text{word}_t | \text{word}_{t-1}) \)
    - Tri-Gram: \( P(\text{word}_t | \text{word}_{t-2}, \text{word}_{t-1}) \)
2. **Counting Approach**:
    - Use frequency counts from training data to compute probabilities.
    - Example: "I like" → "vacations" (probability 1.0), "I like" → "surgery" (probability 0).
3. **Argmax Prediction**:
    - Choose the word with the highest probability given the context.

## From N-Grams to Neural Networks
- **Challenge**: Large context sizes make probability tables sparse and complex.
- **Neural Network Approach**:
    - Represent words as **one-hot vectors** or **embeddings**.
    - **Context Vector** = concatenation of embedding vectors for the context words.
    - Input dimension = vocabulary size × context size.
    - Feedforward NN predicts the next word using a softmax output.

## Limitations
- Traditional N-grams ignore long-term dependencies.
- Feedforward N-gram neural networks still lack positional awareness compared to modern architectures (e.g., RNNs, Transformers).

## Main Idea
N-gram models predict the next word using a fixed-size context of previous words. Increasing `N` captures more context but becomes computationall

---