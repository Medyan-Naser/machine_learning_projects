# Movies Rating Prediction

This project aims to predict movie ratings using deep learning techniques. The dataset used contains user ratings for various movies, and the goal is to predict how a user would rate a movie they haven't watched yet. Two different models are used in this project: **Autoencoder** and **Boltzmann Machine**. Below, we will explain the theory behind each model, their respective architectures, and the training processes.

## 1. Autoencoder Model

### Overview
An **Autoencoder** is a type of neural network used to learn efficient representations of data, typically for the purpose of dimensionality reduction or feature learning. In this project, the Autoencoder is used for collaborative filtering, where the network is trained to reconstruct missing movie ratings based on user preferences.

### Architecture
1. **Input Layer**: 
   - Represents movie ratings for each user, with dimensions `(1, nb_movies)`.

2. **Encoder**:
   - **Layer 1 (`fc1`)**: Compresses input from `nb_movies` to `20` dimensions, using a fully connected layer with a **Sigmoid** activation.
   - **Layer 2 (`fc2`)**: Further reduces the representation from `20` to `10` dimensions.

3. **Bottleneck Layer**:
   - **Layer 3 (`fc3`)**: Expands the representation back from `10` to `20` dimensions, capturing the compressed form of the data.

4. **Decoder**:
   - **Layer 4 (`fc4`)**: Reconstructs the data back to the original `nb_movies` size, aiming to replicate the user's movie ratings.

5. **Activation Function**:
   - **Sigmoid** is used after each layer to introduce non-linearity, allowing the model to capture complex patterns.

The architecture is a **stacked Autoencoder**, which means multiple layers of encoders and decoders are stacked to extract increasingly complex features from the input.

### Training
The model is trained using the **Mean Squared Error (MSE)** loss function, which measures the difference between the predicted and actual ratings. The optimization process minimizes this error, gradually improving the model’s predictions. The **RMSprop** optimizer is used for gradient descent.

The training process runs for **200 epochs**, during which the model progressively learns to predict the ratings based on the patterns discovered in the data.

### Evaluation
After training, the model's performance is evaluated using a test set. The root mean squared error (RMSE) is calculated for the predictions made by the Autoencoder on the test set, providing a measure of how accurately the model predicts unseen ratings.


## 2. Boltzmann Machine Model

### Overview
A **Boltzmann Machine** (BM) is a type of stochastic neural network used for unsupervised learning. It is particularly effective for learning the distribution of binary data. In this project, a **Restricted Boltzmann Machine (RBM)**, a simplified version of a Boltzmann Machine, is used to learn the underlying structure of users' movie preferences.

### Architecture
1. **Visible Layer**:
   - Represents the movie ratings with a size of `(1, nb_movies)`.

2. **Hidden Layer**:
   - **Layer 1 (`nh`)**: Contains `100` units, learning abstract features from the data.

3. **Weights**:
   - A weight matrix `[nh, nv]` connects the visible and hidden layers, defining the strength of their relationships.

4. **Biases**:
   - **Biases (`a`, `b`)**: Each layer has biases to adjust for offsets in the data.

5. **Sampling**:
   - **Gibbs sampling** is used to update the visible and hidden layers probabilistically, capturing latent features in the data.


### Training
The model is trained using **contrastive divergence** and **Gibbs sampling** to update the weights. The training process aims to minimize the difference between the reconstructed and original ratings. This is achieved by adjusting the weights and biases through gradient descent.

The training process runs for **10 epochs**, and the model iteratively improves its ability to predict missing ratings by refining the weights of the connections between the visible and hidden layers.

### Evaluation
After training, the model is evaluated on a test set using a similar approach to the Autoencoder model. The test loss, which is the mean absolute error between the predicted and actual ratings, is calculated. This gives an indication of how well the Boltzmann Machine generalizes to unseen data.


## Conclusion

In this project, both Autoencoders and Boltzmann Machines were used to predict movie ratings based on user preferences. The Autoencoder was chosen for its ability to learn compact representations of user-item interactions, while the Boltzmann Machine was used for its strength in modeling the distribution of user preferences and generating predictions. By comparing the performance of these two models, we gain insights into the strengths and limitations of each approach for collaborative filtering tasks.

Each model was trained on a movie ratings dataset, and their performance was evaluated based on their ability to predict unseen ratings. The results of this project demonstrate the effectiveness of deep learning models in recommendation systems and collaborative filtering tasks.

