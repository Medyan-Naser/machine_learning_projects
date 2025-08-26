
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
4. **Self-Attention (if Transformer-based encoder, depends on the model)**: Uses **unmasked self-attention**, so each token can access **all tokens in the sequence**, including future words.

> **Note:** Unlike GRUs, which have only a hidden state, LSTMs also maintain a **cell state**.

### Decoder Internals
1. **Embedding layer**: Converts each predicted token into an embedding vector.
2. **LSTM layer**: Takes the embedding and the previous hidden/cell states to produce the next states.
3. **Linear layer**: Maps LSTM output to the vocabulary size (`output_dim`).
4. **Softmax activation**: Produces a probability distribution over the vocabulary.
5. Operates **autoregressively**, using its own previous predictions or ground-truth tokens (teacher forcing).
6. **Limitation**: Maximum context size constrains the number of tokens the decoder can process at once, limiting long-range dependencies.

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