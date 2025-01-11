# Convolutional Neural Network for Cat vs. Dog Classification

This project demonstrates the application of Convolutional Neural Networks (CNNs) for image classification. The model, built with TensorFlow and Keras, is trained on a labeled dataset consisting of images of two distinct categories: cats and dogs. By leveraging deep learning techniques, the goal is to develop a model capable of recognizing and classifying images based on their content, focusing on efficiently distinguishing between these two common classes. The project highlights the power of CNNs in solving visual recognition tasks through automated feature extraction and pattern learning.

## Objective

The objective of this project is to:
- Train a CNN to classify images as either "Cat" or "Dog."
- Evaluate the model’s performance on a test set of images.
- Visualize the model's accuracy and loss over the epochs of training.

## Data Preprocessing

Before training the model, the following preprocessing steps were applied:
1. **Rescaling**: All pixel values are scaled by 1/255 to normalize the input images.
2. **Augmentation**: To improve the model's generalization, various augmentation techniques such as shear range, zoom, and horizontal flip were applied to the training set.
3. **Batch Size**: The images were processed in batches of 32 to optimize memory usage and computation time.

## CNN Architecture

The Convolutional Neural Network is structured with the following layers:

1. **Convolutional Layer** (Filters: 32, Kernel Size: 3x3, Activation: ReLU, Input Shape: (64, 64, 3)):
   - **Purpose**: This layer extracts low-level features such as edges and textures. The 3x3 kernel with 32 filters captures various feature maps, while ReLU introduces non-linearity to enable the network to learn complex patterns.

2. **MaxPooling Layer** (Pool Size: 2x2, Strides: 2):
   - **Purpose**: MaxPooling reduces the spatial dimensions of the feature maps, lowering the computational load and making the model more robust to small translations by focusing on the most important features.

3. **Second Convolutional Layer** (Filters: 32, Kernel Size: 3x3, Activation: ReLU):
   - **Purpose**: This layer allows the model to learn more abstract features and deeper interactions between the initial feature maps. ReLU helps prevent vanishing gradients and speeds up training.

4. **MaxPooling Layer** (Pool Size: 2x2, Strides: 2):
   - **Purpose**: The second MaxPooling further reduces the spatial resolution of the feature maps, ensuring that only the most significant information is retained, aiding in better generalization.

5. **Flatten Layer**:
   - **Purpose**: This layer reshapes the 2D feature maps into a 1D vector, allowing the output from the convolutional and pooling layers to be passed into fully connected layers for classification.

6. **Fully Connected Layer** (Units: 128, Activation: ReLU):
   - **Purpose**: The fully connected layer learns complex patterns by combining the features extracted by the previous layers. ReLU ensures non-linearity and improves training speed by mitigating the vanishing gradient problem.

7. **Output Layer** (Units: 1, Activation: Sigmoid):
   - **Purpose**: The output layer produces a probability value between 0 and 1, suitable for binary classification tasks like distinguishing between "Cat" and "Dog." The sigmoid function ensures a clear decision boundary at 0.5.

Each layer is designed to progressively extract, refine, and combine features from the input image, enabling the model to make accurate predictions for binary classification tasks.

## Model Training

The model was compiled using the following configuration:
- **Optimizer**: Adam (Adaptive Moment Estimation)
- **Loss Function**: Binary Cross-Entropy (since this is a binary classification problem)
- **Metrics**: Accuracy

The model was trained for 20 epochs, using the training set for training and the test set for validation. The training progress was monitored by plotting both the loss and accuracy curves.

## Results

### Loss and Accuracy Graphs

The model's performance is visualized by plotting the training and validation accuracy, as well as the training and validation loss. The following plots show the trends over the 20 epochs:

![cnn_accuracy_and_loss](https://github.com/user-attachments/assets/5c36cbe7-e503-44f1-952d-877d5b0a0e71)


### Test Image Predictions

The model was tested on a set of images from the test set to evaluate its performance on unseen data.


<div align="center">
    <img src="https://github.com/user-attachments/assets/646352c3-d489-4ed5-ab6b-f97a96444a58" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/b6267efb-5232-4da8-b411-2b998097a409" alt="SVM" width="45%" style="margin: 0 2%;">
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/0f9d1454-aec0-4d0a-a0d9-724986b1c80b" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/ce422f1d-b0d6-412b-b12c-e053fd8681f1" alt="SVM" width="45%" style="margin: 0 2%;">
</div>

<div align="center">
    <img src="https://github.com/user-attachments/assets/67331739-7e15-4407-899f-762d535eefbd" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/f55190f7-0131-4b13-a240-ab7ba049631b" alt="SVM" width="45%" style="margin: 0 2%;">
</div>
<div align="center">
    <img src="https://github.com/user-attachments/assets/02de0caa-a7ac-4ba0-8a69-8671a726da70" alt="Logistic Regression" width="45%" style="margin: 0 2%;">
    <img src="https://github.com/user-attachments/assets/e66997df-4564-497f-9dd4-7aba9ae1e5c7" alt="SVM" width="45%" style="margin: 0 2%;">
</div>
