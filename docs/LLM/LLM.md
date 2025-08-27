# Large Language Models (LLMs)

Large Language Models (LLMs) are at the heart of today’s AI revolution.  
They are built on **transformer architectures**, which enable them to  
process and generate human-like language — and even extend beyond text  
into images, audio, and multimodal data.

---

## 1. The Premise of Deep Learning

**Deep learning** is a branch of machine learning where instead of coding  
explicit instructions, we design flexible models with millions or billions  
of tunable parameters (called **weights**) and train them using data.

### How It Works
- **Traditional programming**: Write rules explicitly.  
- **Deep learning**: Feed examples into a neural network that tunes itself.  

Example:  
- Linear regression predicts house prices from square footage.  
- Instead of slope + intercept, deep learning models have billions of weights.  

> GPT-3 has ~175 billion parameters organized into ~28,000 matrices.

### Core Ingredients
- **Tensors**: Arrays of real numbers representing input and outputs.  
- **Layers**: Stages that progressively transform tensors.  
- **Backpropagation**: Algorithm that tunes weights using errors.  
- **Matrix multiplications**: The backbone of all computations.  

The success of deep learning comes from models that scale efficiently  
without collapsing under the weight of billions of parameters.

---

## 2. What Are Large Language Models?

An **LLM** is a probabilistic model trained to predict the next token in  
a sequence, given the preceding context.

Formally:
$$
P(\text{next token} \;|\; \text{context})
$$

By repeatedly predicting, sampling, and appending tokens, LLMs generate  
coherent, context-aware text.

---

## 3. GPT: Generative Pretrained Transformer

The term **GPT** stands for:

- **Generative**: Produces text by sampling predictions.  
- **Pretrained**: Learned patterns from massive text corpora.  
- **Transformer**: The architecture enabling context-rich predictions.

### Origins
- Transformers introduced in 2017 (Google) for translation.  
- GPT models extend transformers into general-purpose generators.  

### Predict → Sample → Repeat
- **Predict** a probability distribution over next tokens.  
- **Sample** one token from it.  
- **Append** to context and repeat.  

This loop underlies ChatGPT and other text generators.

---

## 4. Training LLMs

### Pretraining
- Train on trillions of tokens from the internet.  
- Objective: predict the next token.  
- Loss: usually **cross-entropy**.  
- Hardware: large GPU/TPU clusters.  

### Fine-Tuning
- **RLHF (Reinforcement Learning with Human Feedback)**:  
  Refines behavior with human guidance for helpfulness and safety.

---

## 5. Word Embeddings

The first step is turning text into **tokens** and mapping them into vectors.

- Vocabulary (GPT-3): 50,257 tokens (each column will represent a word in the higher dimension).  
- Embedding dimension: 12,288.  
- Embedding matrix ($W_E$): ~617 million weights.
- These weights will be learnt

### Semantic Geometry
- Embeddings live in a high-dimensional space.  
- After training it tends for directions capture meaning:  
  - woman – man ≈ queen – king  ( they will have similr distance or direction, like in both the male word will be above the female)
  - Germany – Japan + sushi ≈ bratwurst  (adding word vectors can result in a vector whcih is a word that is related to the fist two)
- Related words will have their vectores close to each other

### Dot Product Intuition
- Measures similarity between vectors.  
- Positive → aligned (similar).  
- Zero → unrelated.  
- Negative → opposite.

> Dot products appear throughout transformers, especially in attention.

### Beyond Static Meanings
- Initially: embeddings represent single tokens.  
- After processing: embeddings absorb context.  
  - e.g. "king" → "Scottish king in Shakespeare."  

### Context Windows
- GPT-3: 2048 tokens per input.  
- Input represented as a tensor:
  $$
  [\text{2048 tokens} \times \text{12,288 dimensions}]
  $$
- Longer inputs risk losing context.

---


## 6. The Unembedding Step

At the output, the model predicts probabilities for the next token.

- **Unembedding matrix ($W_U$)** maps the final vectors  
  back into vocabulary logits (50,000+ values).  
- Shape: [vocab size × embedding dimension].  
- Adds another ~617 million parameters.  

Thus, embeddings and unembeddings together exceed **1 billion weights**.

---

## 7. From Logits to Probabilities: Softmax

The unembedding step produces **logits** — raw, unnormalized scores.  

**Softmax** converts logits into a valid probability distribution:

1. Exponentiate each logit ($e^{x_i}$).  
2. Divide by sum of all exponentials.  

### Properties
- Ensures values between 0 and 1.  
- Largest logits → highest probabilities.  
- Allows uncertainty and diversity.  

### Temperature Parameter
- Adjusts randomness in generation.  
- High T → more diverse, less predictable.  
- Low T → more deterministic, safer.  
- T = 0 → always pick the highest probability.

Example:
- T = 0 → “Once upon a time…” → predictable fairy tale.  
- T = 1.5 → more original, but risk of nonsense.

---



### 🔑 Key Takeaway
- **BERT (encoder-only)** → Best for **understanding** tasks where meaning must be extracted.
- **GPT (decoder-only)** → Best for **generation** tasks where fluent text must be produced.
- **T5/BART (encoder–decoder)** → Best for **hybrid tasks** like translation and summarization.

---

## 8. Key Takeaways

- **Embeddings** convert tokens into high-dimensional vectors.  
- **Context windows** limit how much the model can consider.  
- **Transformers** use attention + MLP layers to refine meaning.  
- **Unembedding + softmax** turn vectors into probability distributions.  
- **Temperature** controls diversity vs predictability.  
- GPT-3: 175 billion parameters; GPT-4 and beyond are larger.  

---

## 9. Masking

- During training, the model predicts the next token after every subsequence, not just the full context.
- To avoid “cheating,” later tokens must not influence earlier ones.
- Solution: before softmax, set invalid entries to –∞ → they become zero after softmax while keeping normalization intact.
- Always applied in GPT models, even at inference time.

---

## 10. GPT‑3 Parameter Breakdown

GPT‑3 contains a total of **175,181,291,520 parameters**.  
These are spread across different matrices that handle embeddings, attention, feed-forward layers, and the final unembedding step.

Below is the running tally based on what we’ve covered so far.

| Component              | Description                                     | Parameters (approx.) |
|------------------------|-------------------------------------------------|----------------------|
| **Embedding Matrix ($W_E$)** | Maps tokens (50,257 vocab × 12,288 d_embed) into vectors | 617,558,016 |
| **Attention Blocks**   | Self-attention layers (multi-head)              | ~1/3 of parameters |
| **Key** | d_query * d_embed * n_heads * n_layers (128 * 12,288 * 96 * 96) | 14,495,514,624 |
| **Query** | d_query * d_embed * n_heads * n_layers (128 * 12,288 * 96 * 96) | 14,495,514,624 |
| **Value** | d_value * d_embed * n_heads * n_layers (128 * 12,288 * 96 * 96) | 14,495,514,624 |
| **Output** | d_value * d_embed * n_heads * n_layers (128 * 12,288 * 96 * 96) | 14,495,514,624 |
| **Feed-Forward Layers (MLPs)** | Dense transformations per token             | ~2/3 of parameters |
| **Up-projection** | n_neurons * d_embed * n_layers ( 49,152 * 12,288 * 96 )| 57,982,058,496 |
| **Down-projection** | n_neurons * d_embed * n_layers ( 49,152 * 12,288 * 96 )| 57,982,058,496 |
| **Unembedding Matrix ($W_U$)** | Maps final hidden state back to vocabulary logits   | 617,558,016 |
| **Other Matrices**     | Layer norms, biases, and other weights          | To be added |
| **Total**              |                                                | 175,181,291,520 |

> ✅ As we add chapters, this table will expand until all categories of weights are included, summing to the full 175B parameters.
