

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

### Recap

- **Perplexity** evaluates model confidence; lower is better.  
- **Cross-entropy loss** underlies perplexity calculation.  
- **BLEU/ROUGE** use n-gram matching for qualitative text evaluation.  
- **Precision** measures accuracy; **recall** measures completeness; **F1** balances both.  
- Libraries like **NLTK** and **PyTorch** simplify metric computation.

---