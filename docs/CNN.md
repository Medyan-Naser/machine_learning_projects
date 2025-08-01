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

## Data Augmentation

Data Augmentation is a technique to artificially increase the size and diversity of your training dataset by applying random transformations to the input data (usually images).

The goal:

- Prevent overfitting by making the model see more varied examples.
- Improve generalization to unseen data.
- Reduce dependency on large datasets.

**Common Image Augmentations**

- Flip: Horizontal/vertical flip of an image.
- Rotation: Rotate images by small random degrees.
- Zoom / Crop: Zoom in/out, crop random parts.
- Shift: Translate image horizontally/vertically.
- Brightness / Contrast: Change lighting conditions.
- Shear: Skew image along an axis.
- Noise: Add random noise for robustness.

## Diffusion Models

Diffusion models are generative deep learning models that create new data (such as images or audio) by reversing a step-by-step noise process.

Add noise to your data set, and then train your model by giving the noisey images as input and normal images as expected output.

**Key Points**

- State-of-the-art for high-quality image generation (e.g., Stable Diffusion, DALL·E).
- Can be combined with CNNs or Transformers as the backbone for the denoising steps.
- Unlike CNNs (used mostly for classification/detection), diffusion models are designed for data generation.
- Image synthesis and editing.
- Text-to-image generation.
- Audio and video generation.
- Data augmentation.

## Generative Adversarial Networks (GANs)

GANs are generative models that learn to create new, realistic data (like images) by training two neural networks in competition:

- **Generator**: Creates fake data from random noise.
- **Discriminator**: Tries to distinguish between real and generated data.
- Both are trained together until the generator produces data the discriminator cannot easily tell apart from real data.

**Key Points**

- Adversarial training drives the generator to produce highly realistic outputs.
- GANs often use CNNs for image tasks.
- Image synthesis (e.g., creating faces, artwork).
- Image-to-image translation (e.g., sketches → photos, day → night).
- Super-resolution (enhancing image quality).
- Data augmentation for limited datasets.

**Example GAN Structure**

```python
import tensorflow as tf
from tensorflow.keras import layers

# Generator
generator = tf.keras.Sequential([
    layers.Dense(128, activation="relu", input_shape=(100,)),
    layers.Dense(784, activation="sigmoid"),
    layers.Reshape((28,28,1))
])

# Discriminator
discriminator = tf.keras.Sequential([
    layers.Flatten(input_shape=(28,28,1)),
    layers.Dense(128, activation=tf.nn.leaky_relu),
    layers.Dense(1, activation="sigmoid")
])

# Compile discriminator
discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Combined model (Generator + Discriminator, with discriminator frozen)
z = tf.keras.Input(shape=(100,))
fake_img = generator(z)
discriminator.trainable = False
validity = discriminator(fake_img)
gan = tf.keras.Model(z, validity)
gan.compile(optimizer="adam", loss="binary_crossentropy")

# Training loop (simplified)
for epoch in range(epochs):
    # 1. Train discriminator
    real_imgs = get_real_images(batch_size)
    noise = tf.random.normal((batch_size, 100))
    fake_imgs = generator.predict(noise)
    X = tf.concat([real_imgs, fake_imgs], axis=0)
    y = [1]*batch_size + [0]*batch_size
    discriminator.train_on_batch(X, y)

    # 2. Train generator (wants discriminator to label fakes as real)
    noise = tf.random.normal((batch_size, 100))
    gan.train_on_batch(noise, [1]*batch_size)
```