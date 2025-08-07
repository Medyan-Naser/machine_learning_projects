# What is a Tensor and What is Special About It?

## 🔹 What is a Tensor?

At its core, a **tensor is a generalization of scalars, vectors, and matrices** to higher dimensions.  
You can think of it as a container for numbers, organized by **rank (number of dimensions)**:

- **Rank 0**: Scalar → just a number (e.g., `5`).
- **Rank 1**: Vector → a 1D array of numbers (e.g., `[1, 2, 3]`).
- **Rank 2**: Matrix → a 2D array of numbers (e.g., a grid of values).
- **Rank 3+**: Higher-dimensional arrays → e.g., a cube of values, or beyond.

For example:
- An image (height × width × color channels) is often represented as a **rank-3 tensor**.
- A batch of images for training (batch × height × width × channels) is a **rank-4 tensor**.

Formally:

> A **tensor is a multi-dimensional array that obeys certain transformation rules when coordinates change**.

---

## 🔹 What Makes Tensors Special?

1. **Coordinate-System Independence**  
   
    Unlike plain arrays, tensors represent data in a way that doesn’t depend on the choice of coordinates.  
    Example: In physics, the stress tensor or Einstein’s curvature tensor describes physical properties that remain meaningful regardless of the coordinate system.

2. **Mathematical Structure**  

    Tensors can be added, multiplied, and transformed using linear algebra operations. They follow specific transformation rules that preserve meaning across dimensions.

3. **Compact Representation of Complex Data**  

    They let us pack a lot of information into a single object.  
    For example, in deep learning, a tensor might represent the weights of a layer across all input and output neurons.

4. **Efficient Computation**  

    GPUs and specialized hardware are optimized for tensor operations (matrix multiplications, convolutions, etc.).
    This is why libraries like **TensorFlow** and **PyTorch** are literally named after tensors.

5. **Connection Between Math and Reality**  

    In **physics**: Tensors describe stress, strain, and spacetime curvature.  
    In **machine learning**: Tensors describe the flow of data through neural networks.  
    In **engineering**: They describe relationships like how forces translate into deformations.

---

## Derivatives in PyTorch

PyTorch uses **autograd** to compute derivatives automatically.

- Use requires_grad=True for tensors you need derivatives of.
- Call .backward() on the output to compute gradients.

**Single Variable**

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3
```

**Multiple Variables (Partial Derivatives)**

```python
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

z = x**2 + y**3
z.backward()
print(x.grad)  # ∂z/∂x = 2x
print(y.grad)  # ∂z/∂y = 3y^2
```

---
# Image Transforms in PyTorch

`torchvision.transforms` provides tools to preprocess and augment images before feeding them into models.

## Example: Basic Transform Pipeline

```python
import torchvision.transforms as transforms
from PIL import Image

# Define a transform pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Converts image to tensor [C, H, W] in [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values
])

# Load and transform an image
img = Image.open("example.jpg")
img_tensor = transform(img)

print(img_tensor.shape)
```

**Common Transforms**

- transforms.Resize(size) → Resize image
- transforms.CenterCrop(size) → Crop from center
- transforms.RandomHorizontalFlip() → Data augmentation
- transforms.ToTensor() → Convert PIL.Image to tensor
- transforms.Normalize(mean, std) → Normalize channels
- Combine multiple transforms using transforms.Compose.

---

# Training Loop in PyTorch

The **training loop** is the core process that allows a model in PyTorch to learn from data. It repeatedly updates model parameters to minimize a loss function, using gradients computed via backpropagation.

---

## 🔑 Structure of a Training Loop

A typical training loop involves the following steps:

1. **Forward Pass**

Feed the input data into the model to produce predictions (`y_hat`).
Here, `model` can be a built-in PyTorch layer (e.g., `nn.Linear`) or a custom model (e.g. `nn.Module`) you define.

```python
y_hat = model(X)
```

2. **Compute Loss**

Measure how far the predictions are from the target values using a loss function (e.g., Mean Squared Error).

```python
criterion = nn.MSELoss()
loss = criterion(y_hat, Y)
```

3. **Backward Pass (Backpropagation)**

Call `loss.backward()` to compute gradients of the loss with respect to each learnable parameter.
PyTorch builds a computational graph during the forward pass and uses it here to calculate derivatives automatically.

```python
loss.backward()
```

4. **Update Parameters with Optimizer**

Adjust parameters using the gradients to reduce the loss.  
This can be done manually or via an optimizer.

Example (manual update):  
```python
with torch.no_grad():
    w -= lr * w.grad
```

Example:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# clear grad
optimizer.zero_grad()

# Update the model's weights
optimizer.step()
```

5. **Zero Gradients**

Clear gradients before the next iteration to prevent accumulation or can be done before .backward to insure its cleared.

```python
w.grad.zero_()
```

---

## 🔄 Conceptual Flow


- **Forward pass** evaluates the model with current parameters.  
- **Loss** measures prediction error.  
- **Backward pass** compute gradients to see how each parameter contributed to error. 
- **Optimizer** step adjust parameters in the direction that reduces the loss
- **Zero gradients** reset for the next loop
- Repeating this process iteratively moves the model closer to the best solution.

```python
# Write your code here

num_epochs = 1000

# Training loop
for epoch in range(num_epochs):
    model.train() 
    # Forward pass
    outputs = model(X_train_tensor)
    # Compute loss
    loss = criterion(outputs, y_train_tensor)
    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()  
with torch.no_grad():
    # Predictions on training and testing data
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

    # Apply threshold of 0.5
    train_pred_labels = (train_preds >= 0.5).float()
    test_pred_labels = (test_preds >= 0.5).float()

    # Calculate accuracy
    train_accuracy = (train_pred_labels == y_train_tensor).sum().item() / y_train_tensor.size(0)
    test_accuracy = (test_pred_labels == y_test_tensor).sum().item() / y_test_tensor.size(0)

# Print accuracy results
print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
```


---

## ⚡ Key Ideas
- **Gradients drive learning**: they indicate how to change parameters to reduce error.
- **Learning rate** controls the step size in updates — too high can overshoot, too low can slow learning.
- **Zeroing gradients** is essential, as PyTorch accumulates gradients by default.
- This pattern scales from simple linear regression to deep neural networks.

---

✅ In essence, the training loop is an optimization cycle where predictions improve step by step through feedback from the loss function.


---

# Pytorch

A dynamic computation graph means that the network's structure can change on the fly during execution, allowing for more intuitive and flexible model development. 
This feature is helpful in advanced NLP applications where the neural network architecture needs to adapt dynamically to varying inputs.

## Dynamic computation graphs (Autograd)

PyTorch's Autograd system allows dynamic changes to the network during training, enhancing flexibility and easing the development process. This adaptability is particularly beneficial for research and experimentation.