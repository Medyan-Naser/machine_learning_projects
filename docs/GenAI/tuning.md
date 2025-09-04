# Tuning and Parameter-Efficient Fine-Tuning (PEFT)

## Overview
Fine-tuning is the process of adapting a pretrained model (such as a large language model) to a new, task-specific domain.  
There are two major approaches:

1. **Full Fine-Tuning** – updating all parameters in the network.  
2. **Parameter-Efficient Fine-Tuning (PEFT)** – updating only a small portion of parameters while keeping most of the model frozen.  

PEFT has become the standard approach for working with large pretrained models due to its efficiency, scalability, and ability to preserve pretrained knowledge.

---

## Full Fine-Tuning

- **Definition**: Updates all model parameters, layers, and neurons.  
- **Strengths**: Maximum flexibility and potentially best performance when data and resources are abundant.  
- **Drawbacks**:
    - Requires **massive computational resources** and **GPU memory**.
    - Needs large amounts of **task-specific labeled data**.
    - High risk of **overfitting**.
    - Can lead to **catastrophic forgetting** (loss of general pretrained knowledge).  

Because of these drawbacks, alternative methods (PEFT) were introduced.

---

## Parameter-Efficient Fine-Tuning (PEFT)

PEFT reduces the number of **trainable parameters** while maintaining adaptability to downstream tasks.  
This achieves:  
- Lower compute and memory usage.  
- Faster training and deployment.  
- Stable adaptation without overwriting pretrained knowledge.  

### Categories of PEFT Methods

PEFT methods can be grouped into three main categories:

1. **Selective Fine-Tuning**  
2. **Additive Fine-Tuning**  
3. **Reparameterization-Based Fine-Tuning**  

Some cross-cutting techniques like **soft prompting** also fall into this ecosystem.

---

## 1. Selective Fine-Tuning
- **Idea**: Update only a subset of parameters or layers.  
- **Use Case**: Smaller models or networks where certain layers are most relevant to the new task.  
- **Limitation**: Less effective for transformers because of their highly distributed parameterization.  

---

## 2. Additive Fine-Tuning
Instead of modifying existing weights, new task-specific modules are added.  

### Adapters
- Inserted between **transformer attention blocks**.  
- Structure:
    1. **Down projection** → reduce dimensionality.  
    2. **Non-linear transformation**.  
    3. **Up projection** → restore dimensionality.  
- Only the adapter parameters are trained; the base model remains frozen.  

**Advantages**:
- Modular: adapters can be swapped for different tasks.  
- Efficient: base model can be reused across multiple domains.  

---

## 3. Soft Prompting
Prompts are treated as **trainable parameters** that guide the model’s behavior without altering weights.  

- **Definition**: Learnable embeddings prepended or concatenated to input tokens.  
- **Variants**:
    - **Prompt Tuning** – optimize a small number of tokens.  
    - **Prefix Tuning** – optimize prefix embeddings across layers.  
    - **P-Tuning** – integrates prompts into deeper layers.  
    - **Multitask Prompt Tuning** – share prompts across tasks.  

**Benefit**:  
Fine-tunes a model for new tasks with extremely small memory cost.  

---

## 4. Reparameterization-Based Fine-Tuning
These methods re-express weight matrices using low-rank approximations.  

### LoRA (Low-Rank Adaptation)
- Decomposes weight updates into low-rank matrices.  
- Adds small trainable components while freezing the original weights.  
- Captures the most important directions in parameter space.  

### Variants
- **QLoRA**: Combines LoRA with quantization → reduces memory footprint further.  
- **DoRA**: Dynamically adjusts the rank based on weight magnitude.  

**Key Concept – Rank**:  
Represents the minimum number of vectors needed to span a space.  
- In PEFT, low-rank decompositions reduce dimensionality while preserving expressiveness.  

---

## Comparing Fine-Tuning Methods

| Method                  | Trainable Parameters | Preserves Pretrained Knowledge | Efficiency | Typical Use Case |
|--------------------------|----------------------|--------------------------------|------------|------------------|
| **Full Fine-Tuning**    | 100%                 | ❌ Risk of forgetting           | ❌ Low     | When abundant data & compute are available |
| **Selective Fine-Tuning** | Small subset         | ✅ Partial                      | ⚠️ Moderate| Simple tasks, smaller models |
| **Adapters (Additive)** | Small new modules     | ✅ Yes                         | ✅ High    | Multi-task setups, modular fine-tuning |
| **Soft Prompting**      | Very small (tokens)   | ✅ Yes                         | ✅ Very High | Lightweight tuning, rapid prototyping |
| **LoRA / QLoRA / DoRA** | Low-rank updates      | ✅ Yes                         | ✅ Very High | State-of-the-art LLM tuning |

---

## Recap

- **Full Fine-Tuning**: powerful but expensive and risky.  
- **PEFT**: efficient, scalable, and widely used in modern NLP.  
- **Three main categories**:
    - Selective fine-tuning.  
    - Additive fine-tuning (adapters).  
    - Reparameterization (LoRA, QLoRA, DoRA).  
- **Soft prompting** is a complementary approach that avoids weight updates.  

PEFT has become the **default strategy** for adapting large pretrained models to new domains with minimal cost and maximal efficiency.  
