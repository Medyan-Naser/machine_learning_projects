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

