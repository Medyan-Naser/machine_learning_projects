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
<img src="https://upload.wikimedia.org/wikipedia/commons/8/88/Logistic-curve.svg" width="350">

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
<img src="https://upload.wikimedia.org/wikipedia/commons/c/cb/Activation_tanh.svg" width="350">

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
<img src="https://upload.wikimedia.org/wikipedia/commons/6/6c/Rectifier_and_softplus_functions.svg" width="350">

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
<img src="https://upload.wikimedia.org/wikipedia/commons/a/ae/Activation_prelu.svg" width="350">

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
TODO

---

## 6. Softmax

**Formula:**  

$$
\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \quad i=1,\dots,K
$$

**Info:**
- Used in the **output layer** for multi-class classification
- Converts logits to probability distribution

**Graph:**  
TODO

---

# ✅ Summary

- **Sigmoid**: Obsolete, suffers from vanishing gradients  
- **Tanh**: Better than sigmoid, but still limited  
- **ReLU**: Most common, fast, biologically motivated  
- **Leaky ReLU / ELU**: Fix dying ReLU issue  
- **Softmax**: For multi-class classification output  

---
