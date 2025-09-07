# Machine Learning Methods Overview

Machine Learning algorithms can be broadly grouped into categories based on how they learn from data: Supervised Learning, Unsupervised Learning, and Dimensionality Reduction / Feature Extraction. Some additional specialized methods (e.g., ensemble methods, neural networks) are also important.

## 1. Supervised Learning
In supervised learning, the algorithm is trained on labeled data (inputs with known outputs).

### Regression (Predicting Continuous Values)
- Linear Regression: Models a linear relationship between input features and a continuous target variable.
- Polynomial Regression: Extends linear regression with polynomial terms to model non-linear relationships.
- Non-linear Regression: Uses more complex functions (e.g., exponential, logarithmic, or kernel-based) to capture relationships beyond polynomial.
- Regression Trees: Decision trees designed for predicting continuous outcomes instead of categories.
- Support Vector Regression (SVR): Extension of SVM for regression tasks. It tries to fit the best line within a margin of tolerance (epsilon), ignoring errors within that margin and penalizing only larger deviations.

### Classification (Predicting Categories)
Note: all of the following can be used for Regression

- Logistic Regression: A statistical model that predicts the probability of a categorical outcome (binary or multi-class).
- Decision Trees: Splits data based on feature thresholds to classify samples.
- Support Vector Machines (SVM): Finds the best hyperplane to separate classes with maximum margin.
- K-Nearest Neighbors (KNN): Classifies based on the majority label among the k closest training samples.
- Naive Bayes: A probabilistic model based on Bayes’ theorem, assuming independence between features.
- Support Vector Machines (SVM): SVM finds the optimal separating hyperplane that maximizes the margin between classes. Works well in high-dimensional spaces and can use kernels for non-linear separation.

### Ensemble Methods (combines multiple models to produce a stronger and more accurate overall model)
Note: can be used for Classification and Regression

- Random Forests: Combines many decision trees to reduce overfitting and improve accuracy.
- Gradient Boosting (XGBoost, LightGBM, CatBoost): Builds trees sequentially, correcting previous errors for high accuracy.

### Validation
**Purpose:**  
Validation is used to evaluate and tune models during training, ensuring they generalize well to unseen data.  
It helps prevent overfitting and guides hyperparameter selection.

**Difference from Training vs Testing:**

- **Training set:** Used to learn model parameters (weights).
- **Validation set:** Used to tune hyperparameters, compare models, and decide when to stop training (early stopping).
- **Test set:** Used only once at the end to report unbiased performance. The test set must remain untouched until final evaluation.
- If data is limited, use **k-fold cross-validation** to maximize training efficiency.

## 2. Unsupervised Learning
In unsupervised learning, the algorithm learns from unlabeled data, discovering patterns or groups.

### Clustering (Grouping Similar Data)
- K-Means: Partitions data into k clusters based on distance to centroids.
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Finds clusters of arbitrary shape based on density, marking outliers as noise.
- HDBSCAN (Hierarchical DBSCAN): Extension of DBSCAN with better handling of variable density clusters.
- Hierarchical Clustering: Builds a tree (dendrogram) of clusters for multilevel grouping.

### Association Rule Learning (Important)
- Apriori / FP-Growth: Finds frequent itemsets and rules in transactional data (e.g., market basket analysis).

### Temporal Clustering
- Unsupervised clustering that groups data based on **feature similarity and temporal proximity**.
- Useful for **time-series data, logs, or events** where patterns occur in bursts.
- Works by combining feature similarity with **time-based distance** in clustering algorithms (e.g., DBSCAN with time as a dimension).
- Automatically identifies **temporal patterns** and **event clusters** without specifying the number of clusters.

### Anomaly Detection
- Isolation Trees: Randomly split features to isolate individual data points. Anomalies are isolated faster (shorter path lengths) because they lie in sparse regions.
- Isolation Forests: An ensemble of isolation trees. The average path length of a point across many trees indicates its likelihood of being an anomaly. Shorter average path length = higher chance of being an outlier.

### SPELL (Using Longest Common Subsequence)
- Unsupervised method for clustering **sequences of events** based on **temporal similarity**.
- Uses **Longest Common Subsequence (LCS)** to measure similarity between sequences: sequences with longer shared subsequences are considered more similar.
- Constructs a **graph of sequences** connected by similarity and forms clusters from dense regions; sparse sequences are treated as anomalies.
- Useful for **log analysis, process mining, and detecting anomalous temporal patterns**.

## 3. Dimensionality Reduction / Feature Extraction
Used to reduce the number of input features while retaining most information.

- Autoencoder: Neural network that learns compressed representations of data and reconstructs the original input.
- Principal Component Analysis (PCA): Projects data into fewer dimensions that capture the most variance.
- t-SNE (t-Distributed Stochastic Neighbor Embedding): Non-linear technique for visualizing high-dimensional data in 2D/3D.
- UMAP (Uniform Manifold Approximation and Projection): Preserves both local and global structure, faster than t-SNE.
- Linear Discriminant Analysis (LDA): Reduces dimensions while maximizing class separability.

## 4. Other Key Approaches

### Neural Networks & Deep Learning
- Feedforward Neural Networks (FNNs): Layers of neurons mapping inputs to outputs.
- Convolutional Neural Networks (CNNs): Specially designed for images and spatial data.
- Recurrent Neural Networks (RNNs): Designed for sequential data (e.g., text, speech, time series). They maintain a memory of previous inputs via hidden states but suffer from vanishing gradient problems for long sequences.
- Long Short-Term Memory Networks (LSTMs): A type of RNN that uses gates to manage memory, allowing it to capture long-term dependencies and mitigate vanishing gradients.
- Gated Recurrent Units (GRUs):A simplified version of LSTMs with fewer gates. Often perform comparably with fewer parameters and faster training.
- Transformers: State-of-the-art models for NLP and vision tasks.

### Reinforcement Learning (RL)
- Q-Learning, Deep Q-Networks (DQN): Learn optimal action policies by estimating the value (Q-value) of taking an action in a given state, using reward feedback from the environment.
- Policy Gradient Methods: Directly optimize decision-making policies.

### Zero-Shot Learning (ZSL)
Zero-Shot Learning refers to the ability of a model to **make predictions for tasks or classes it has never seen during training**. Instead of relying on labeled examples for every possible category, the model uses **auxiliary information** such as textual descriptions, embeddings, or semantic attributes to generalize.

**Key Points:**
- **No Training Examples for Target Task:** The model predicts outputs for classes or tasks it has never been explicitly trained on.
- **Auxiliary Information:** Uses semantic descriptions or embeddings (e.g., Word2Vec, text prompts) to relate new classes to known ones.
- **Applications:**
  - **NLP / LLMs:** Answer questions, translate, or summarize text without task-specific training.
  - **Computer Vision:** Classify images into categories not seen during training (e.g., CLIP).
- **Difference from Few-Shot Learning:** Zero-shot uses **no examples**, while few-shot uses a small number of labeled examples.
- **How it Works:** The model maps inputs and outputs into a **shared representation space**, compares inputs with unseen class representations, and predicts the most likely match.

Zero-shot learning allows AI models to **generalize beyond their training data**, making them highly flexible for new tasks.
