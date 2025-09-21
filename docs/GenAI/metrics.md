

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

## BLEU score metric for evaluation
While peplexity serves as a general metric to evaluate the performance of language model in predicting the correct next token, BLEU score is helpful in evaluating the quality of the final generated translation.
Validating the results using BLEU score is helpful when there is more than a single valid translation for a sentence as you can include many translation versions in the reference list and compare the generated translation with different versions of translations.

The BLEU (Bilingual Evaluation Understudy) score is a metric commonly used to evaluate the quality of machine-generated translations by comparing them to one or more reference translations. It measures the similarity between the generated translation and the reference translations based on n-gram matching.

The BLEU score is calculated using the following formulas:

1. **Precision**:
    - Precision measures the proportion of n-grams in the generated translation that appear in the reference translations.
    - Precision is calculated for each n-gram order (1 to N) and then combined using a geometric mean.
    - The precision for a particular n-gram order is calculated as:
   
    $$\text{Precision}_n(t) = \frac{\text{CountClip}_n(t)}{\text{Count}_n(t)}$$
    
    where:

      - $\text{CountClip}_n(t)$ is the count of n-grams in the generated translation that appear in any reference translation, clipped by the maximum count of that n-gram in any single reference translation.
      - $\text{Count}_n(t)$ is the count of n-grams in the generated translation.

2. . **Brevity penalty**:
    - The brevity penalty accounts for the fact that shorter translations tend to have higher precision scores.
    - It encourages translations that are closer in length to the reference translations.
    - The brevity penalty is calculated as:
   
    $$\text{BP} = \begin{cases} 1 & \text{if } c > r \\\\\\\\\\\\\\\\\\\\\\ e^{(1 - \frac{r}{c})} & \text{if } c \leq r \end{cases}$$
    
    where:

      - $c$ is the total length of the generated translation.
      - $r$ is the total length of the reference translations.

3. **BLEU score**:
    - The BLEU score is the geometric mean of the precisions, weighted by the brevity penalty.
    - It is calculated as:
    
    $$\text{BLEU} = \text{BP} \cdot \exp(\sum_{n=1}^{N}w_n \log(\text{Precision}_n(t)))$$
    
    where:

      - $N$ is the maximum n-gram order.
      - $w_n$ is the weight assigned to the precision at n-gram order $n$, commonly set as $\frac{1}{N}$ for equal weights.