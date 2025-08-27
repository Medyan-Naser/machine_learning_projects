# Activation Functions in Neural Networks

Activation functions introduce **non-linearity** into neural networks, enabling them to learn complex patterns. Below are the most common types.

---

## 1. Sigmoid (Logistic Function)

**Formula:**  
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

**Info:**
- Output range: (0, 1)
- Historically popular but **rarely used today**
- Problems: vanishing gradients, slow convergence

**Graph:**  
![Sigmoid](../assets/activation_functions/sigmoid.avif)

---

## 2. Tanh (Hyperbolic Tangent)

**Formula:**  
$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

**Info:**
- Output range: (-1, 1)
- Zero-centered (better than sigmoid)
- Still can suffer from vanishing gradients

**Graph:** 
![Tanh](../assets/activation_functions/tanh.avif)

---

## 3. ReLU (Rectified Linear Unit)

**Formula:**  
$$
f(x) = \max(0, x)
$$

**Info:**
- Most popular activation today  
- Inspired by **biological neurons** (fire or not)
- Efficient and helps avoid vanishing gradients
- Risk: "dying ReLU" (neurons stuck at 0)

**Graph:** 
![ReLU](../assets/activation_functions/ReLU.avif)

---

## 4. Leaky ReLU

**Formula:**
$$
f(x) =
\begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$

**Info:**
- Small negative slope ($\alpha$) to avoid dying ReLU
- Common $\alpha = 0.01$

**Graph:**
![Leaky ReLU](../assets/activation_functions/leaky_ReLU.avif)

---

## 5. ELU (Exponential Linear Unit)

**Formula:**  
$$
f(x) =
\begin{cases}
x & x > 0 \\
\alpha(e^x - 1) & x \leq 0
\end{cases}
$$

**Info:**
- Similar to Leaky ReLU but smoother
- Better convergence in some cases

**Graph:**
![ELU](../assets/activation_functions/ELU.avif)

---

## 6. Softmax

**Formula:**  

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i=1,\dots,K
$$

**With Temperature:**  

$$
\sigma_T(z)_i = \frac{e^{z_i / T}}{\sum_{j=1}^{K} e^{z_j / T}}, \quad i=1,\dots,K
$$

**Info:**

- Used in the **output layer** for multi-class classification
- Converts logits to a probability distribution
- **Temperature \(T\):**
    - \(T < 1\): Sharper distribution (more confident, close to always picking the max)
    - \(T > 1\): Flatter distribution (more random, allows exploration)
    - \(T \to 0\): Greedy (always top probability)
    - \(T \to \infty\): Approaches uniform distribution


**Graph:**
![Softmax](../assets/activation_functions/softmax.avif)

---

# ✅ Summary

- **Sigmoid**: Obsolete, suffers from vanishing gradients  
- **Tanh**: Better than sigmoid, but still limited  
- **ReLU**: Most common, fast, biologically motivated  
- **Leaky ReLU / ELU**: Fix dying ReLU issue  
- **Softmax**: For multi-class classification output  

---
