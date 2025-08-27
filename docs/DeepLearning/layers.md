# Common Keras Layers Overview

Keras provides many layer types you can add to neural networks. Each serves a specific purpose in learning, regularization, or optimization.

## Core Layers

- Dense
    - Fully connected layer.
    - Each neuron connects to every input.
    - Used at the end for classification or regression.
- Conv2D / Conv1D / Conv3D
    - Convolution layers for 1D, 2D, or 3D data.
    - Extract spatial or temporal features.
    - Key for CNNs.
- MaxPooling / AveragePooling
    - Downsamples feature maps by taking max (or average) from each region.
    - Reduces size and overfitting.
- Flatten
        - Converts multidimensional feature maps into a 1D vector.
    - Used before Dense layers.
- Embedding
    - Converts integers (like word indices) into dense vectors.
    - Essential for NLP models.

## Regularization Layers

- Dropout
    - Randomly drops a fraction of neurons during training.
    - Prevents overfitting by forcing redundancy.
    - Example: Dropout(0.5) drops 50% of neurons.
    - Dropout is less common in convolutional layers due to the spatial dependencies in image data.
- BatchNormalization
    - Normalizes activations across a batch.
    - Speeds up training, improves stability.
    - Often placed after Conv/Dense layers.
- SpatialDropout2D
    - Drops entire feature maps instead of individual neurons.
    - Useful in CNNs to maintain spatial correlation.

## Advanced / Special Layers

- LSTM / GRU
    - Recurrent layers that capture sequence dependencies.
    - Used in time series and NLP.
- ConvLSTM2D
    - Combines CNNs and LSTMs.
    - Good for video or spatiotemporal tasks.
- GlobalAveragePooling / GlobalMaxPooling
    - Reduces feature maps to a single value per channel.
    - Keeps spatial invariance without Flatten.
- Reshape
    - Changes tensor shape without changing data.
    - Used to fit outputs into the next layer’s expected input.
- Input
    - Defines the model’s input shape.
    - First layer in a Sequential or Functional API model.

## Customization Layers

- Lambda
    - Wraps custom functions as a layer.
    - For operations not available in built-in layers.
- ActivityRegularization
    - Adds a regularization penalty to the layer’s output.
    - Controls magnitude of activations.