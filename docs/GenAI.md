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
