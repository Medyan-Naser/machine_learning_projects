# Gradient Descent

Gradient is vectore of relatvie derivitave.
the magnitude of gradient descent of all the componenets tells you which changes matter more. some connections (weights) matter more.

Gradient Descent is the core algorithm behind most optimizers.  
It updates model parameters (weights and biases) by moving them in the direction that reduces the loss (cost) function.

**Idea**: Find the parameter values that minimize the loss.
**Update Rule**:  
  θ = θ - α ∇L(θ)  
where:  
  - θ = parameters (weights)  
  - α = learning rate (step size)  
  - ∇L(θ) = gradient of the loss with respect to parameters  

**Variants**:
  - **Batch Gradient Descent**: Uses the entire dataset each update (slow, rarely used).  
  - **Stochastic Gradient Descent (SGD)**: Uses one sample at a time (fast but noisy).  
  - **Mini-Batch Gradient Descent**: Uses small batches (most common).  

> **Note**: All optimizers below (SGD, Adam, RMSprop, etc.) are improvements or modifications of basic gradient descent to make training faster, more stable, or better at generalizing.

---

# Optimizers Overview

Optimizers are algorithms that update model weights to minimize loss during training. Different optimizers use different strategies to adjust learning rates, momentum, and gradient scaling.

## Common Optimizers in Keras:

## SGD (Stochastic Gradient Descent)

- Updates weights using gradients of each batch.
- Can use momentum and Nesterov acceleration for faster convergence.
- Often used in computer vision tasks.

## Adam (Adaptive Moment Estimation)

- Combines momentum and adaptive learning rates.
- Works well in most cases and is the default choice for many models.
- Fast and efficient.

## RMSprop

- Adapts learning rate for each parameter based on recent gradient magnitudes.
- Good for recurrent neural networks (RNNs).

## Adagrad
- Adjusts learning rate per parameter, larger updates for infrequent features.
- Effective for sparse data (e.g., text).
- Learning rate decays quickly.

## Notes on Usage
1. Start with Adam for general tasks.
2. Use SGD with momentum when aiming for better generalization.
3. Consider RMSprop for sequence models.
4. AdamW (in TensorFlow Addons/PyTorch) is common in NLP for improved weight decay handling.

---

# Loss
Measures how well the model fits the data (to minimize).

## Regression

- `mean_squared_error` (MSE): Penalizes large errors more, standard for regression.
- `mean_absolute_error` (MAE): Measures average absolute difference.
- huber loss: Combines MSE and MAE, less sensitive to outliers.

## Binary Classification

- `binary_crossentropy`: Used when predicting 2 classes (sigmoid output).
- hinge: Common in SVM-like setups.

## Multiclass Classification

- `categorical_crossentropy`: For one-hot encoded labels (softmax output).
- `sparse_categorical_crossentropy`: For integer labels instead of one-hot.

## Other Specialized
- `kullback_leibler_divergence`: Measures difference between probability distributions.
- `cosine_proximity`: Maximizes similarity between predicted and true vectors.

---

# Metrics
Track model performance during training/testing (for monitoring, not optimization).

## Regression

- mean_squared_error
- mean_absolute_error
- RootMeanSquaredError (custom or from tf.keras.metrics)

## Classification

- accuracy: Percentage of correct predictions.
- precision: Of predicted positives, how many are correct.
- recall: Of actual positives, how many are detected.
- AUC: Area under ROC curve, useful for imbalanced data.

## Other

- top_k_categorical_accuracy(k): Accuracy where the true label is in the top k predictions.