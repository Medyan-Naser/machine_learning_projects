# Core ML & AI Libraries

Below are the four main Python libraries for Machine Learning and Deep Learning, with an explanation of what they are and when to use them.

---

## 1. Scikit-learn (`sklearn`)

**What is it?**
- A Python library for classical machine learning algorithms.
- Built on top of NumPy, SciPy, and matplotlib.

**Used for:**
- Regression, classification, clustering (e.g., Logistic Regression, Decision Trees, KMeans)
- Preprocessing pipelines
- Feature engineering and model selection

---

## 2. Keras

**What is it?**
- A high-level deep learning API built for fast prototyping.
- Now part of TensorFlow (`tf.keras`).

**Used for:**
- Quickly building and training deep learning models.
- Great for image and text classification tasks using neural networks.
- Beginner-friendly deep learning interface.

---

## 3. TensorFlow

**What is it?**
- An end-to-end ML platform developed by Google.
- Supports low-level and high-level APIs.

**Used for:**
- Training and deploying deep learning models at scale.
- Production ML systems.
- Running models on mobile/edge using TensorFlow Lite.

---

## 4. PyTorch

**What is it?**
- A deep learning framework developed by Facebook.
- Known for dynamic computation graphs and flexibility.

**Used for:**
- Research and development of custom models.
- Complex deep learning architectures (e.g., Transformers, GANs).
- Rapid experimentation and debugging.

---

## 🧠 Library Use by Task

Below is a table showing which library can be used for various ML and AI methods:

| Task / Method                       | Scikit-learn | Keras         | TensorFlow    | PyTorch       |
|------------------------------------|--------------|---------------|---------------|---------------|
| Linear Regression                  | ✅            | ❌            | ❌            | ❌            |
| Logistic Regression                | ✅            | ❌            | ❌            | ❌            |
| Decision Trees / Random Forests    | ✅            | ❌            | ❌            | ❌            |
| K-Means Clustering                 | ✅            | ❌            | ❌            | ❌            |
| Dimensionality Reduction (PCA)     | ✅            | ❌            | ❌            | ❌            |
| Feedforward Neural Networks (MLP)  | ❌            | ✅            | ✅            | ✅            |
| Convolutional Neural Networks (CNN)| ❌            | ✅            | ✅            | ✅            |
| Recurrent Neural Networks (RNN)    | ❌            | ✅            | ✅            | ✅            |
| LSTM / GRU                         | ❌            | ✅            | ✅            | ✅            |
| Transformers (e.g., BERT, GPT)     | ❌            | ❌            | ✅ (HuggingFace) | ✅ (HuggingFace) |
| GANs (Generative Adversarial Nets) | ❌            | ❌            | ✅            | ✅            |
| Autoencoders                       | ❌            | ✅            | ✅            | ✅            |
| Time Series Forecasting            | ✅            | ✅            | ✅            | ✅            |
| Image Classification               | ❌            | ✅            | ✅            | ✅            |
| Text Classification (NLP)          | ❌            | ✅            | ✅            | ✅            |
| Model Deployment                   | ❌            | ✅ (via TF)    | ✅ (TF Serving / Lite) | ✅ (TorchScript, ONNX) |

✅ = Strong Support  
❌ = Not Designed For This

---

You can combine these tools. For example:
- Use **scikit-learn** for preprocessing and use **Keras** or **PyTorch** for deep learning.
- Use **PyTorch** for training and export the model using **ONNX** for inference.
