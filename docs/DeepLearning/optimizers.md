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

## Momentum

Momentum is a technique that helps accelerate gradient descent by accumulating a velocity vector in parameter space, which smooths out updates and helps avoid getting stuck in local minima or noisy gradients.

- Instead of updating parameters solely based on the current gradient, momentum adds a fraction of the previous update to the current one.
- This creates inertia, allowing the optimizer to maintain direction across small fluctuations.
- It improves convergence speed and stability, especially in ravines or when gradients vary greatly.

**Update Rule with Momentum:**

$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha v_t
$$

where:  
- \(v_t\) is the velocity at iteration \(t\)  
- \(\beta\) is the momentum coefficient (commonly around 0.9)  
- \(\alpha\) is the learning rate  
- \(\nabla L(\theta_t)\) is the gradient at iteration \(t\)

Momentum helps training by:  
- Smoothing noisy updates from stochastic gradients  
- Accelerating convergence along consistent gradient directions  
- Reducing oscillations in directions with high curvature

## Batch Normalization

Batch Normalization (BatchNorm) is a technique that normalizes the inputs of each layer to stabilize and speed up training.  
It reduces internal covariate shift (changes in input distributions across layers) and allows for higher learning rates.

**Process:**  
1. For each mini-batch, compute the mean and variance of the inputs.  
2. Normalize the inputs:  
   $$
   \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
   $$
3. Scale and shift using learnable parameters $\gamma$ and $\beta$:  
   $$
   y = \gamma \hat{x} + \beta
   $$

**Benefits:**  
- Speeds up convergence by keeping activations in a stable range.  
- Reduces sensitivity to initialization.  
- Provides slight regularization, lowering the risk of overfitting.  
- Often allows training with fewer epochs and higher learning rates.


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

# Weight Initialization

Weight initialization is the process of setting the starting values of model parameters before training begins. Proper initialization helps stabilize training and prevents issues like vanishing or exploding gradients.

## Default Initialization

- Most deep learning frameworks use built-in default initializations for layers.
- For example, linear and convolutional layers often initialize weights using uniform or normal distributions scaled based on layer size.
- Default initializations work well in many cases but might not be optimal for all architectures or activations.

## Popular Initialization Methods

### Xavier (Glorot) Initialization

- Designed for layers with sigmoid or tanh activations.
- Samples weights from a uniform distribution within:

  $$
  U\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)
  $$

  where \(n_{in}\) and \(n_{out}\) are the input and output units of the layer.
- Keeps variance of activations and gradients balanced across layers.

### He (Kaiming) Initialization

He Initialization scales weights to maintain stable signal variance in networks using ReLU activations, which zero out about half of the neurons and affect variance flow.

- Designed for ReLU and similar activations.
- Samples weights from a normal distribution with variance:

  $$
  \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)
  $$

- Helps maintain stable signal flow when many neurons are zeroed by ReLU.

## Why Initialization Matters

- Poor initialization can cause slow convergence or training failure.
- Good initialization helps training start smoothly and reach better results faster.
- Initialization is a key complement to optimizers and learning rate schedules.


---

# Loss
Measures how well the model fits the data (to minimize).

Most common loss functions are derived from the principle of **Maximum Likelihood Estimation (MLE)**.  
MLE chooses parameters that maximize the likelihood of observing the training data.  
- In regression, minimizing Mean Squared Error corresponds to MLE under a Gaussian error assumption.  
- In classification, cross-entropy losses come from MLE under a categorical distribution assumption.  
*(Note: MLE defines the objective. Optimization algorithms like gradient descent are used to find the parameters, starting from an initial guess.)*


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