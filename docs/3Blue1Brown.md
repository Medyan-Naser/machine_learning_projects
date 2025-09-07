# Backpropagation


Backpropagation is the core algorithm that enables neural networks to learn. It computes how each weight and bias should be adjusted to reduce the loss, using gradient descent.

This document covers:

1. Intuitive Understanding of Backpropagation
2. Backpropagation with Calculus (Chain Rule)
3. Practical Training: Stochastic Gradient Descent (SGD)

## **1. Intuitive Understanding of Backpropagation**

**The Goal**

- Learning = minimizing the cost function.
- Each weight and bias influences the final cost differently.
- Backpropagation tells us:
    - Direction: Should a weight go up or down?
    - Magnitude: How much impact does it have on the cost?

**Process**

1. **Forward Pass**

    - Input → layers of weighted sums + biases → activation functions → output.
    - Compute the loss (e.g., squared error or cross-entropy).

2. **Desired Adjustments**

    - For a training example (say an image of “2”):
        - Increase the output neuron for digit “2.”
        - Decrease others.
    - The adjustment size is proportional to the error for each output.

3. **How Neurons Adjust**

    - A neuron's activation is influenced by:
        - Its bias.
        - The weights connecting it to active previous neurons.
    - Strong activations in the previous layer have the biggest influence.
    - Analogy: Neurons that fire together wire together.

4. **Propagating Backwards**

    - Each output neuron suggests changes for the previous layer.
    - Combine suggestions → get nudges for hidden layers.
    - Repeat backwards through the network.

5. **From One Example to the Dataset**

    - One example biases the network toward that class.
    - So we average adjustments across all training examples.
    - This averaged set ≈ negative gradient of the cost function.

## 2. Backpropagation with Calculus

To go beyond intuition, we use calculus and the chain rule.

### Chain Rule in Neural Networks  
Each weight influences the cost indirectly:

$$
w \;\rightarrow\; z \;\rightarrow\; a \;\rightarrow\; C
$$

The **chain rule** computes how a small change in \(w\) affects the cost \(C\):

$$
\frac{\partial C}{\partial w}  
= \frac{\partial C}{\partial a} 
\cdot \frac{\partial a}{\partial z} 
\cdot \frac{\partial z}{\partial w}
$$

---

### Computing Derivatives  

- **Derivative of cost with respect to activation**  

$$
\frac{\partial C}{\partial a} = 2(a - y)
$$  

Larger error → larger influence on cost.  

- **Derivative of activation with respect to weighted input**  

$$
\frac{\partial a}{\partial z} = f'(z)
$$

Where \(f\) is the activation function (e.g., sigmoid or ReLU).  

- **Derivative of weighted input with respect to weight**  

$$
\frac{\partial z}{\partial w} = a^{(L-1)}
$$  

Depends on the activation from the previous layer.  

---

### Interpreting the Derivatives  

- The weight’s influence depends on how active the previous neuron is.  
- Mirrors the idea: **neurons that fire together wire together**.  
- For the bias:  

$$
\frac{\partial z}{\partial b} = 1
$$

---

### Multi-Neuron Layers  

In real networks with many neurons:  

- Activations are indexed by layer and neuron:  

$$
a^{(L)}_j
$$

- Each weight connects neuron \(k\) in layer \(L-1\) to neuron \(j\) in layer \(L\):  

$$
w^{(L)}_{jk}
$$

The chain rule still applies, just with more indices.  
A neuron in layer \(L-1\) can influence multiple outputs, so its sensitivity sums contributions from all paths.  


## 3. Stochastic Gradient Descent (SGD)

**Why Not Use the Full Dataset Each Step?**

Computing the gradient for all examples is too slow.

**Solution: Mini-Batches**

- Shuffle training data.
- Split into mini-batches (e.g., 100 examples).
- Compute gradient for each mini-batch.

**Effect**

- Each step is a noisy approximation of the true gradient.
- Looks like a “drunk person stumbling downhill” → quick but uneven progress.
- Over time, it converges to a local minimum.

**Recap**

- Backpropagation uses the chain rule to compute how each weight and bias affects the cost.
- Intuitively: propagate desired nudges backwards from the output layer.
- Formally: compute partial derivatives using the chain rule.
- SGD makes training feasible by approximating the gradient with mini-batches.
- Together, these steps enable neural networks to learn from large datasets.

## 4. Dynamic Programming and Minimal Energy Methods in AI

### Dynamic Programming (DP)

**Definition**

- DP solves complex problems by breaking them into simpler, overlapping subproblems.
- Relies on **optimal substructure**: the best solution can be built from best solutions of subproblems.
- Stores intermediate results to avoid redundant computation.

**Applications in AI**

- **Reinforcement Learning (RL):**  
    - Value Iteration and Policy Iteration use DP to find optimal policies in Markov Decision Processes (MDPs).  
- **Planning and Search:**  
    - DP algorithms compute shortest paths, optimal sequences of actions, and resource allocation.

**Key Idea**

- Build solutions incrementally from subproblem results.
- Efficiently handles problems where brute-force search would be too expensive.

---

### Minimal Energy Methods

**Definition**

- Starting from the end!
- Represent problems using an **energy or cost function** \(E(x)\).
- Goal: find the configuration \(x^*\) that **minimizes energy**:  
$$
  \[
  x^* = \arg\min_x E(x)
  \]
$$
- In AI, “energy” often corresponds to loss, cost, or negative probability.

**Applications in AI**

- **Computer Vision:** Image segmentation, stereo matching (graph cuts). 
        - This allow for finding a line of pixels with the lowest energy to remove when scalling down an image, but you keep the important parts of the image.
- **Neural Networks:** Training minimizes a loss function (a type of energy).  
- **Probabilistic Models:** Markov Random Fields (MRFs) find the most probable configuration by minimizing energy.

**Key Idea**

- Optimal solutions correspond to minimal energy states.
- Often solved with specialized optimization algorithms, sometimes combined with DP.

---

### Connection Between DP and Minimal Energy

- Both aim to **find optimal solutions** efficiently.  
- Example: **Sequence labeling (OCR, POS tagging)**  
    - Modeled as a chain-structured MRF.  
    - Minimal energy corresponds to the best label sequence.  
    - Solved efficiently with **dynamic programming algorithms**, like the Viterbi algorithm.  

**Summary**

- DP is a method for structured optimization.  
- Minimal energy methods are a problem formulation emphasizing optimal configurations.  
- DP can solve minimal energy problems in AI efficiently when structure allows.
