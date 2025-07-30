# Convolutional Neural Networks (CNNs) Overview

CNNs are deep learning models specialized for image data. They automatically learn spatial features (edges, shapes, patterns) through layers like convolutions, pooling, and dense connections.

## 🔹 Core Components

### Convolution Layer (`Conv2D`)

- Applies filters/kernels to input images to detect features (edges, textures, shapes).
- Each filter slides across the image (convolution operation) creating feature maps.
- Parameters:
    - `filters`: number of feature maps to learn.
    - `kernel_size`: size of filter (e.g., (3,3)).
    - `strides`: step size of filter movement.
    - `padding`:
        - `same` → keeps output size same as input.
        - `valid` → no padding, output shrinks.

### Activation Functions

- Typically ReLU after each Conv layer.
- Adds non-linearity, allowing the network to learn complex patterns.

### Pooling Layer

- MaxPooling2D: reduces spatial dimensions by taking the maximum value in each region.
- Purpose:
    - Reduces computation.
    - Makes features more robust to translation.
- Example: `MaxPooling2D(pool_size=(2,2))`.

### Flatten Layer

- Converts the 2D feature maps into a 1D vector.
- Prepares data for fully connected (Dense) layers.

### Fully Connected (Dense) Layers
- After flattening, Dense layers perform classification or regression.
- Final layer usually uses:
    - Softmax → for multiclass classification.
    - Sigmoid → for binary classification.
    - Linear → for regression.

## 🔹 Example CNN Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

## 🔹 Key Notes
- Conv2D → extracts local features.
- MaxPooling2D → reduces size & overfitting risk.
- Flatten → prepares features for Dense layers.
- Dense layers → perform final prediction.
- CNNs excel at image classification, object detection, and computer vision tasks.