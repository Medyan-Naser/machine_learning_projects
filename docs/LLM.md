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

## 6. Inside a Transformer

Transformers refine embeddings layer by layer.

1. **Tokenization** → Break input into tokens.  
2. **Embedding Layer** → Map tokens into vectors.  
3. **Positional Encoding** → Add order information.  
4. **Self-Attention Block**  
   - Attention updates embeddings based on context, enabling precise meaning.
   - It allows information transfer between distant tokens
   - Formula:
     $$
     \text{Attention}(Q,K,V) = 
     \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
     $$
   - Multi-head attention captures different relationships.  
5. **Feed-Forward Block (MLP)** → Independent nonlinear transformations.  
6. **Stacking Layers** → Repeat attention + MLP many times.  
7. **Final Vector** → Context-rich representation.  

---

## 7. The Unembedding Step

At the output, the model predicts probabilities for the next token.

- **Unembedding matrix ($W_U$)** maps the final vectors  
  back into vocabulary logits (50,000+ values).  
- Shape: [vocab size × embedding dimension].  
- Adds another ~617 million parameters.  

Thus, embeddings and unembeddings together exceed **1 billion weights**.

---

## 8. From Logits to Probabilities: Softmax

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

## 9. Encoder vs Decoder Transformers

- **Encoder–Decoder Models** (e.g., T5, BERT):  
  - Encoder reads input; decoder generates output.  
  - Used in translation, summarization.  

- **Decoder-Only Models** (e.g., GPT):  
  - Focused on **autoregressive text generation**.  
  - Most common for chatbots.

---

## 10. Applications of Transformers

Transformers extend far beyond text:

- **Text → Text**: ChatGPT, summarizers, translators.  
- **Text → Image**: DALL·E, MidJourney.  
- **Image → Text**: Captioning, visual question answering.  
- **Audio → Text**: Whisper.  
- **Text → Audio**: Synthetic voices.  
- **Multimodal**: GPT-4 Vision, combining text + images.  
- **Science**: protein folding (AlphaFold), drug discovery.

---

## 11. Key Takeaways

- **Embeddings** convert tokens into high-dimensional vectors.  
- **Context windows** limit how much the model can consider.  
- **Transformers** use attention + MLP layers to refine meaning.  
- **Unembedding + softmax** turn vectors into probability distributions.  
- **Temperature** controls diversity vs predictability.  
- GPT-3: 175 billion parameters; GPT-4 and beyond are larger.  

---

## 12. Masking

- During training, the model predicts the next token after every subsequence, not just the full context.
- To avoid “cheating,” later tokens must not influence earlier ones.
- Solution: before softmax, set invalid entries to –∞ → they become zero after softmax while keeping normalization intact.
- Always applied in GPT models, even at inference time.
---

## 13. The Attention Mechanism


### Core Idea  
Attention allows the model to determine **which tokens in a sequence should influence each other** when building contextual meaning. Instead of treating each word embedding in isolation, attention selectively passes information between tokens based on their relevance.  

---

### Queries, Keys, and Values  
For each token embedding $e$:  

- **Query ($q$):** What information this token is looking for.  
- **Key ($k$):** What information this token can provide.  
- **Value ($v$):** The actual information passed on if a key matches a query.  

Each is computed through learned weight matrices:  

\[
q = W_q \cdot e,\quad k = W_k \cdot e,\quad v = W_v \cdot e
\]

The **attention score** between two tokens is the dot product $q \cdot k$.  
- Larger scores → higher relevance.  
- Apply **softmax** to normalize scores into probabilities.  
- Use these as weights to form a weighted sum of values:  

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
\]

Here $d_k$ is the dimension of the key/query space, added for numerical stability.  

---

### Single-Head Attention  
- Computes **one attention pattern** across the sequence.  
- Updates embeddings by aggregating relevant information (e.g., adjectives modifying nouns).  
- Limitation: one head can only capture a single type of relationship at a time.  

---

### Multi-Head Attention  
- Runs many independent attention heads in parallel.  
- Each head has its own $W_q$, $W_k$, and $W_v$ matrices → learns different contextual relationships.  
- Example:  
  - One head tracks subject–verb agreement.  
  - Another detects long-range references (e.g., *Harry* + *wizard* → *Harry Potter*).  
- Outputs from all heads are concatenated and transformed by an output matrix.  
- GPT‑3 uses **96 heads per block**.  

---

### Stacking Multiple Layers  
- Attention isn’t applied once, but repeatedly in a stack of layers.  
- Each layer refines embeddings further using the outputs of the previous one.  
- GPT‑3 has **96 layers**, each containing multi-headed attention + feedforward layers.  
- Early layers capture simpler patterns (e.g., local syntax), while deeper layers encode **abstract concepts** like tone, style, and factual relationships.  



---

## 14. The Multi-Layer Perceptron (MLP)  

### Role in Transformers  
In transformers, each token embedding repeatedly flows through two key components:  
- **Attention**, which lets embeddings exchange context.  
- **MLPs**, which independently transform each embedding to enrich meaning and store knowledge.  

While attention gets most of the attention, a **majority of the model’s parameters live inside the MLPs**. Research (e.g., from Google DeepMind) suggests that many of the model’s **factual associations** — like *Michael Jordan → basketball* — are encoded in these MLP layers.  

---

### Structure of an MLP Block  
Each MLP block processes each token embedding independently:  

1. **Up Projection (Expansion)**  
   - Multiply the embedding $E$ by a large learned weight matrix $W_{up}$ and add a bias $B_{up}$.  
   - Expands the vector into a much higher-dimensional space (e.g., 4× the embedding size in GPT‑3).  
   - Each row can be thought of as asking: *Does this embedding contain feature X?*  
   - Produces an intermediate vector of "neuron activations".  

2. **Non-Linearity (ReLU or GELU)**  
   - Apply an elementwise non-linear function.  
   - ReLU maps negative values to 0 and keeps positives unchanged.  
   - Acts like an **AND gate**, only triggering when certain combinations of features are present.  
   - Example: One neuron could activate only when the embedding encodes both “Michael” and “Jordan”.  

3. **Down Projection (Compression)**  
   - Multiply by another learned matrix $W_{down}$ and add bias $B_{down}$.  
   - Projects back to the original embedding dimension.  
   - Columns of $W_{down}$ can be thought of as **directions to add back into the embedding** when the corresponding neuron fires.  
   - Example: The neuron triggered by “Michael Jordan” could add the *basketball* direction.  

4. **Residual Connection**  
   - The output is added back to the original embedding, refining it while preserving context.  

---

### How Facts Can Be Stored  
- Assume there are specific high-dimensional directions for **Michael**, **Jordan**, and **basketball**.  
- The MLP can learn:  
  - Detect when an embedding encodes both Michael + Jordan (via $W_{up}$ and ReLU).  
  - Activate a neuron that adds the basketball direction (via $W_{down}$).  
- Result: The output embedding carries the encoded fact that *Michael Jordan plays basketball*.  


---

### Superposition  
A natural question: does each neuron cleanly represent a feature like *Michael Jordan*?  
Evidence suggests the answer is **no** — and that’s actually beneficial.  

- **Key Idea**: In high dimensions, you can store far more features if you allow them to overlap slightly.  
- This is called **superposition**: features are stored as combinations of neurons rather than isolated ones.  

**Why it works**  
- In an n-dimensional space, you can fit only n perfectly perpendicular directions.  
- But if you allow directions to be *nearly* perpendicular (e.g., 89–91°)
- The number of vectors you can cram into a space that are nearly perpendicular grows exponentially with the number of dimensions.
- The Johnson–Lindenstrauss lemma shows that high-dimensional spaces can pack huge numbers of nearly-orthogonal vectors.  

**Example**  
- In a 100-dimensional space, you can randomly generate 10,000 vectors that are all close to perpendicular.  
- With optimization, they can span many thousands of features while remaining almost independent.  
features**.  
- Features are distributed across combinations of neurons, not isolated units.  
- This explains both why LLMs scale so well and why their internals are hard to interpret.  

Researchers often use **sparse autoencoders** to uncover these hidden superposed features.  

---

### Summary  
- **MLPs** in transformers are the main storage for knowledge and facts.  
- Each block uses:  
  - Up Projection → Non-Linearity → Down Projection → Residual Connection.  
- **Parameter count**: ~116 billion in GPT‑3, making them the largest part of the model.  
- **Superposition** allows packing far more features into the network than its raw dimensionality suggests.  


---

## 15. GPT‑3 Parameter Breakdown

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
