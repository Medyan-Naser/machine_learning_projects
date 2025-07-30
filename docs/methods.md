# Machine Learning Methods Overview

Machine Learning algorithms can be broadly grouped into categories based on how they learn from data: Supervised Learning, Unsupervised Learning, and Dimensionality Reduction / Feature Extraction. Some additional specialized methods (e.g., ensemble methods, neural networks) are also important.

## 1. Supervised Learning
In supervised learning, the algorithm is trained on labeled data (inputs with known outputs).

### Regression (Predicting Continuous Values)
- Linear Regression: Models a linear relationship between input features and a continuous target variable.
- Polynomial Regression: Extends linear regression with polynomial terms to model non-linear relationships.
- Non-linear Regression: Uses more complex functions (e.g., exponential, logarithmic, or kernel-based) to capture relationships beyond polynomial.
- Regression Trees: Decision trees designed for predicting continuous outcomes instead of categories.

### Classification (Predicting Categories)
Note: all of the following can be used for Regression

- Logistic Regression: A statistical model that predicts the probability of a categorical outcome (binary or multi-class).
- Decision Trees: Splits data based on feature thresholds to classify samples.
- Support Vector Machines (SVM): Finds the best hyperplane to separate classes with maximum margin.
- K-Nearest Neighbors (KNN): Classifies based on the majority label among the k closest training samples.
- Naive Bayes: A probabilistic model based on Bayes’ theorem, assuming independence between features.

### Ensemble Methods (combines multiple models to produce a stronger and more accurate overall model)
Note: can be used for Classification and Regression

- Random Forests: Combines many decision trees to reduce overfitting and improve accuracy.
- Gradient Boosting (XGBoost, LightGBM, CatBoost): Builds trees sequentially, correcting previous errors for high accuracy.

## 2. Unsupervised Learning
In unsupervised learning, the algorithm learns from unlabeled data, discovering patterns or groups.

### Clustering (Grouping Similar Data)
- K-Means: Partitions data into k clusters based on distance to centroids.
- DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Finds clusters of arbitrary shape based on density, marking outliers as noise.
- HDBSCAN (Hierarchical DBSCAN): Extension of DBSCAN with better handling of variable density clusters.
- Hierarchical Clustering: Builds a tree (dendrogram) of clusters for multilevel grouping.

### Association Rule Learning (Important)
- Apriori / FP-Growth: Finds frequent itemsets and rules in transactional data (e.g., market basket analysis).

## 3. Dimensionality Reduction / Feature Extraction
Used to reduce the number of input features while retaining most information.

- Principal Component Analysis (PCA): Projects data into fewer dimensions that capture the most variance.
- t-SNE (t-Distributed Stochastic Neighbor Embedding): Non-linear technique for visualizing high-dimensional data in 2D/3D.
- UMAP (Uniform Manifold Approximation and Projection): Preserves both local and global structure, faster than t-SNE.
- Linear Discriminant Analysis (LDA): Reduces dimensions while maximizing class separability.

## 4. Other Key Approaches

### Neural Networks & Deep Learning
- Feedforward Neural Networks (FNNs): Layers of neurons mapping inputs to outputs.
- Convolutional Neural Networks (CNNs): Specially designed for images and spatial data.
- Recurrent Neural Networks (RNNs) & LSTMs: Handle sequential data such as text or time series.
- Transformers: State-of-the-art models for NLP and vision tasks.

### Reinforcement Learning (RL)
- Q-Learning, Deep Q-Networks: Learn optimal policies through reward signals.
- Policy Gradient Methods: Directly optimize decision-making policies.