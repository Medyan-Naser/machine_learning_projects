# Generative AI Architectures and Models

## Architectures Overview

### Recurrent Neural Networks (RNNs)
- Designed for **sequential or time-series data**.  
- Use **loops** to retain memory of previous inputs.  
- Applications:  
    - Language modeling  
    - Machine translation  
    - Speech recognition  
    - Image captioning  
- **Fine-tuning**: Adjust weights/structure for domain-specific tasks.

---

### Transformers
- Use **self-attention** to model long sequences efficiently.  
- Enable **parallel training** for scalability.  
- Example: **GPT** for text generation.  
- **Fine-tuning**: Typically update only the final output layers.  
- Strong in contextual understanding and real-time translation.

---

### Generative Adversarial Networks (GANs)
- Components:  
    - **Generator**: Produces samples.  
    - **Discriminator**: Evaluates real vs. fake.  
- Training is a **competitive process** that improves both models.  
- Applications: **Image and video generation**.

---

### Variational Autoencoders (VAEs)
- **Encoder-decoder** structure with **probabilistic latent space**.  
- Generate diverse outputs with shared characteristics.  
- Applications: **Art, design, and creative tasks**.

---

### Diffusion Models
- Learn to **remove noise** and reconstruct high-quality data.  
- Applications:  
    - High-quality image generation  
    - Restoring old or damaged photographs  

---

## Training Approaches
- **RNNs**: Loop-based memory for sequences.  
- **Transformers**: Self-attention, parallelizable.  
- **GANs**: Generator–Discriminator competition.  
- **VAEs**: Encoder-decoder with probability distributions.  
- **Diffusion Models**: Noise removal guided by data statistics.

---

## Generative AI for NLP

### Applications in NLP
- **Machine Translation**: Context-aware, more accurate conversions.  
- **Chatbots/Virtual Assistants**: Natural, human-like, with personalization.  
- **Sentiment Analysis**: Detects subtle emotional cues in text.  
- **Text Summarization**: Extracts core meaning with precision.  

---

### Large Language Models (LLMs)
- **Definition**: Foundation models trained on massive datasets (websites, books, etc.) using deep learning.  
- **Scale**:  
    - Training data: up to **petabytes**.  
    - **Billions of parameters** fine-tuned to optimize task performance.  
- **Capabilities**:  
    - Predict next words in a sequence.  
    - Generate coherent, contextually relevant, creative content.  
    - Perform tasks with minimal task-specific training.  
- **Examples**:  
    - **GPT**: Decoder-based, excels in text generation and chatbots.  
    - **BERT**: Encoder-only, strong at context understanding (e.g., Q&A, sentiment analysis).  
    - **BART / T5**: Encoder-decoder, versatile across NLP tasks.  

---

### GPT vs ChatGPT
- **GPT**  
  - Uses supervised learning (sometimes RL).  
  - Focused on general text generation tasks.  
- **ChatGPT**  
  - Uses **supervised learning + RLHF** (Reinforcement Learning with Human Feedback).  
  - Optimized for conversational interaction.  

---


## Tokenization in NLP

- **Definition**: The process of splitting text into smaller pieces (tokens) for model input.  
- **Types of Tokenization**:  
  - **Word-based**: Each word is a token. Preserves meaning but increases vocabulary size.  
  - **Character-based**: Splits into characters. Small vocabulary but less semantic meaning.  
  - **Subword-based**: Keeps common words whole, splits rare words into subwords. Balances vocabulary size and meaning.  

- **Popular Tokenizers**:  
  - **WordPiece** (used in BERT) → splits words based on frequency and meaning.  
  - **Unigram / SentencePiece** (used in XLNet) → breaks text into subwords, assigning IDs.  

- **Practical Notes**:  
  - Tools like **NLTK**, **spaCy**, and **torchtext** support tokenization.  
  - Special tokens such as **[BOS]** (beginning of sentence) and **[EOS]** (end of sentence) help models understand sequence boundaries.  
  - Padding tokens ensure uniform input lengths for training.  


---

## Data Loaders in Generative AI

- **Definition**: A tool that efficiently prepares and loads data for training models.  
- **Purpose**:  
    - Handles **batching** (grouping samples together).  
    - Supports **shuffling** to avoid order-based learning.  
    - Enables **on-the-fly pre-processing** to optimize memory.  

- **In PyTorch**:  
    - Built using the **`torch.utils.data.DataLoader`** class.  
    - Works with custom **Dataset** classes defining `__init__`, `__len__`, and `__getitem__`.  
    - Integrates seamlessly into training pipelines.  

- **Batch Processing**:  
    - Ensures consistent input sizes with **padding** (`pad_sequence`).  
    - **`batch_first=True`** → batch size is the first dimension.  
    - Common transformations in batches:  
      - Tokenization  
      - Numericalization (mapping tokens to indices)  
      - Tensor conversion  

- **Collate Function**:  
    - Customizable function applied during batching.  
    - Handles tokenization, padding, and tensor creation without altering the raw dataset.  

---

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

## Summary
- **Word2Vec**: Framework for learning dense word embeddings.
- **CBOW**: Predicts a target word from its context; efficient for frequent words.
- **Skip-Gram**: Predicts context words from a target; better for rare words.
- **Pre-Trained Embeddings**: Use large-scale pre-learned vectors (like GloVe) to improve model accuracy and training speed.

---

## Sequence-to-Sequence Models and Recurrent Neural Networks (RNNs)

### 1. Overview
- **Sequence-to-sequence (seq2seq) models** transform an input sequence into an output sequence of possibly different length.
- Widely used in generative AI applications:
    - **Machine translation**: English → French.
    - **Text summarization**: Condense long text into key points.
    - **Chatbots**: Turn user queries into natural responses.
    - **Code generation**: Convert task descriptions into code.
- **Input and output sequences can differ in length**:
    - **Sequence-to-label**: Many inputs → one label (e.g., document classification).
    - **Label-to-sequence**: One input → many outputs (e.g., image-to-caption generation).

---

### 2. Importance of Sequence Representation
- Context matters:  
    - “The man bites the dog” vs. “The dog bites the man” have the same bag-of-words counts but different meanings.
    - Sequence models (via one-hot encoding or embeddings) preserve word order and context.
- Traditional neural networks assume **IID** (independent, identically distributed) data, which fails for sequences where each element depends on previous ones.
- Conditional probability modeling introduces **memory** to capture dependencies between timesteps.

---

### 3. Preparing Data for Sequence Models
1. **Mark sequence boundaries** with `BOS` (beginning-of-sequence) and `EOS` (end-of-sequence) tokens.
2. **Sort sentences by length** to improve batching efficiency.
3. **Pad shorter sequences** with a special token so all batch elements have the same length.

---

### 4. Basic Recurrent Neural Networks
- **RNNs** process sequences step-by-step, remembering past information through a hidden state.
- **Key components**:
    - **Input** \(x_t\): Data at timestep *t*.
    - **Hidden state** \(h_t\): Memory that captures past information.
    - **Output** \(z_t\): Computed at each timestep from current input and hidden state.
- Hidden states are updated recurrently:  
  \( h_t = f(W_x x_t + W_h h_{t-1}) \) with nonlinearity (often `tanh`).

---

### 5. Unrolling Over Time
- Start with initial hidden state \( h_0 = 0 \).
- For each timestep:
    1. Combine current input and previous hidden state.
    2. Update hidden state.
    3. Generate output.
- Hidden states carry context from previous timesteps into future predictions.

---

### 6. Decoding Strategies
- **Greedy decoding**: Pick the token with the highest probability at each step.
- **Top-k sampling**: Randomly sample from the top *k* candidates for more diverse and fluent text.

---

### 7. RNN Limitations
- Can only retain **short-term dependencies** reliably.
- Suffer from vanishing/exploding gradients in long sequences.

---

### 8. RNN Variants
#### Gated Recurrent Unit (GRU)
- **Two gates**:
    - **Update gate (z)**: Controls how much past information to keep.
    - **Reset gate (r)**: Controls how much past information to forget.
- Simplifies training and improves long-range dependency handling.

#### Long Short-Term Memory (LSTM)
- **Three gates**:
    - **Input gate**: Selectively updates memory.
    - **Forget gate**: Removes irrelevant information.
    - **Output gate**: Decides what part of memory to output.
- Maintains both:
    - **Short-term memory**: Hidden state \( h_t \).
    - **Long-term memory**: Cell state \( c_t \).

---

### 9. Key Takeaways
- Seq2seq models enable flexible input/output sequence lengths for varied tasks.
- RNNs provide memory for sequential data but struggle with long-term dependencies.
- GRUs and LSTMs extend RNNs to capture longer contexts and improve training stability.


---

## Encoder–Decoder RNN Models for Translation

After watching this video, you will be able to **implement an encoder–decoder RNN model**.

### Overview
Encoder–decoder architectures extend sequence-to-sequence models so that the input and output sequences can be of **different lengths**. A popular example is a **translation model**, where the encoder processes an input sentence, and the decoder generates its translation.

### Architecture
- **Encoder**: A series of RNN units (often LSTMs) that process the input sequence token by token.  
  - Each RNN passes its **hidden state** to the next RNN.
  - The **final hidden (and cell) state** becomes the **context** for the decoder.
  - The encoder’s goal: encode the input sequence into a vector representation. It does **not** generate output text.

- **Decoder**: A series of RNN units that generate the output sequence **autoregressively**:
  - Starts with the encoder’s final hidden and cell states.
  - Takes the previous output token as input (or the ground-truth token during teacher forcing).
  - Outputs one token at a time until the `<EOS>` token is generated.

### Encoder Internals
1. **Embedding layer**: Converts each input token into an embedding vector.
2. **LSTM layer**: Processes embeddings to produce hidden and cell states.  
   - `n_layers`: number of recurrent layers.  
   - `hid_dim`: size of hidden and cell states.  
   - `dropout`: regularization to improve generalization.
3. Only the hidden and cell states are passed to the decoder — the output vectors are discarded.

> **Note:** Unlike GRUs, which have only a hidden state, LSTMs also maintain a **cell state**.

### Decoder Internals
1. **Embedding layer**: Converts each predicted token into an embedding vector.
2. **LSTM layer**: Takes the embedding and the previous hidden/cell states to produce the next states.
3. **Linear layer**: Maps LSTM output to the vocabulary size (`output_dim`).
4. **Softmax activation**: Produces a probability distribution over the vocabulary.
5. Operates **autoregressively**, using its own previous predictions or ground-truth tokens (teacher forcing).

**Decoder parameters**:
- `output_dim`: target vocabulary size.
- `emb_dim`: embedding size.
- `hid_dim`: hidden state size.
- `n_layers`: number of LSTM layers.
- `dropout`: dropout probability.

### Building the Seq2Seq Model
- **Class**: Inherits from `nn.Module`.
- **Inputs**: `encoder`, `decoder`, `device`, `trg_vocab`.
- **Forward pass**:
  1. Initialize an output tensor to store predictions.
  2. Pass the source sequence to the encoder → get final hidden and cell states.
  3. Use the first target token (`<BOS>`) to start decoding.
  4. For each time step:
     - Pass the current token + states to the decoder.
     - Save output predictions.
     - Choose next input: ground truth (teacher forcing) or predicted token.
  5. Return the tensor of predictions.

**Teacher Forcing**:
- A training method where the true token from the training set is fed to the decoder instead of its previous prediction.
- Helps models converge faster.

### Recap
- RNNs can create sequence-to-sequence models for tasks like translation.
- Encoder–decoder architectures allow different-length input/output sequences.
- **Encoder**: Processes input, outputs final hidden/cell states.
- **Decoder**: Generates output sequence token-by-token using autoregression.
- LSTM-based architectures track both **short-term (hidden state)** and **long-term (cell state)** dependencies.

---

## Metrics for Evaluating the Quality of Generated Text

After this section, you will be able to:
- Describe **perplexity** and its role in evaluating generative models.
- Use **precision** and **recall** for evaluating generated text.
- Implement evaluation metrics for generated text.

---

### Why Evaluation Matters
Generative AI and Large Language Models (LLMs) produce text, images, and more.  
To measure success, we need metrics that assess **consistency** and **contextual relevance** of generated text.

---

### Perplexity
**Definition:**  
A measure of how "surprised" or uncertain a model is when predicting the next word. Lower perplexity → better model.

**Key concepts:**
- Given a phrase, models assign **probabilities** to sequences based on training data.
- These probabilities mirror the "true" sequence likelihoods (like `y` labels in classification).
- **True probability (P):** Likelihood from actual data.
- **Estimated probability (Q):** Likelihood from the model’s learned distribution.

**Relation to cross-entropy loss:**
- Perplexity is computed from the **average cross-entropy loss**:
  
  \[
  \text{Perplexity} = e^{\text{Average Cross-Entropy Loss}}
  \]
- Cross-entropy measures how far predicted probabilities are from the true probabilities.
- Lower loss → lower perplexity → better performance.

**Example:**
- Model 1 loss → exp(loss) = 35.2  
- Model 2 loss → exp(loss) = 142.6 (worse)

> **Note:** Perplexity is usually computed on the **training/validation set**; it does not capture all qualitative aspects of generated text.

---

### Beyond Perplexity: N-gram Matching
For test set evaluation, metrics like **BLEU** and **ROUGE** compare generated text to reference text using **n-grams**.

**Example:**
Reference: `"The cat is sitting on the rug"`  
Generated: `"The big cat is on the rug"`

Matching unigrams: "The", "cat", "is", "on", "the"  
Matching bigrams: "on the"  

---

### Precision, Recall, and F1 in Machine Translation
- **Precision**: Accuracy of generated text  
  \[
  \text{Precision} = \frac{\text{CountMatch}}{\text{CountGenerated}}
  \]
- **Recall**: Completeness of generated text  
  \[
  \text{Recall} = \frac{\text{CountMatch}}{\text{CountReference}}
  \]
- **F1 Score**: Harmonic mean of precision and recall.

---

### Libraries for Evaluation
- **NLTK**: BLEU, METEOR  
- **PyTorch**: Cross-entropy loss, perplexity computation
- Other libraries: ROUGE, SacreBLEU, etc.

---

# Recap

- **Perplexity** evaluates model confidence; lower is better.  
- **Cross-entropy loss** underlies perplexity calculation.  
- **BLEU/ROUGE** use n-gram matching for qualitative text evaluation.  
- **Precision** measures accuracy; **recall** measures completeness; **F1** balances both.  
- Libraries like **NLTK** and **PyTorch** simplify metric computation.

---

