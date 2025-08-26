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
