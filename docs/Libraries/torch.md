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

PyTorch uses a **dynamic computational graph**, also called a **define-by-run** graph. A computational graph is a representation where:

- **Nodes** represent operations (e.g., addition, multiplication, matrix multiplication).  
- **Edges** represent the flow of data (tensors) between operations.

In PyTorch, this graph is **created dynamically during each forward pass**. Every operation performed on a tensor automatically adds nodes to the graph, and the graph keeps a **history of operations** required for computing derivatives.

### Static vs Dynamic Graph

- **Static Graph (e.g., TensorFlow 1.x, Theano)**:  
  - The graph is defined **before** running any data.  
  - Changing the computation requires rebuilding the graph.  
  - Optimized for repeated execution, but less flexible.  

- **Dynamic Graph (PyTorch)**:  
  - Built **on-the-fly** during execution.  
  - Supports Python control flow (`if`, `for`, `while`) naturally.  
  - Easier to debug and allows variable-length inputs and dynamically changing architectures.

### Differentiation and Backpropagation

PyTorch’s dynamic graph works closely with **autograd**:

- Each operation in the graph records its inputs and operation type.  
- When `.backward()` is called on a tensor, PyTorch traverses the graph in reverse to compute **derivatives (gradients)**.  
- Gradients are calculated **only when needed**, and each forward pass creates a fresh graph.  

This allows flexible gradient computation for complex and varying models without predefining the entire graph.

### Why It’s Special

- **Flexible and Pythonic**: Write models with standard Python code.  
- **Supports dynamic architectures**: Useful for variable-length sequences, recursive networks, and research experimentation.  
- **Automatic differentiation**: Keeps the history of computations to compute derivatives efficiently and on demand.  
- **Easy debugging**: Inspect tensors and operations at any point using Python tools.

Overall, PyTorch’s dynamic computational graph combines **flexibility, transparency, and automatic differentiation**, making it ideal for both research and production use.


## Training model in PyTorch
Now that the model is defined, you define a train function the seq2seq model. Let's go through the code and understand its components:

1. `train(model, iterator, optimizer, criterion, clip)` takes five arguments:
    - `model` is the model that will be trained.
    - `iterator` is an iterable object that provides the training data in batches.
    - `optimizer` is the optimization algorithm used to update the model's parameters.
    - `criterion` is the loss function that measures the model's performance.
    - `clip` is a value used to clip the gradients to prevent them from becoming too large during backpropagation.

2. The function starts by setting the model to training mode with `model.train()`. This is necessary to enable certain layers (e.g., dropout) that behave differently during training and evaluation.

3. It initializes a variable `epoch_loss` to keep track of the accumulated loss during the epoch.

4. The function iterates over the training data provided by the `iterator`. Each iteration retrieves a batch of input sequences (`src`) and target sequences (`trg`).

5. The input sequences (`src`) and target sequences (`trg`) are moved to the appropriate device (e.g., GPU) using `src = src.to(device)` and `trg = trg.to(device)`.

6. The gradients of the model's parameters are cleared using `optimizer.zero_grad()` to prepare for the new batch.

7. The model is then called with `output = model(src, trg)` to obtain the model's predictions for the target sequences.

8. The `output` tensor has dimensions `[trg len, batch size, output dim]`. To calculate the loss, the tensor is reshaped to `[trg len - 1, batch size, output dim]` to remove the initial `<bos>` token, which is not used for calculating the loss.

9. The target sequences (`trg`) are also reshaped to `[trg len - 1]` by removing the initial `<bos>` token and making it a contiguous tensor. This matches the shape of the reshaped `output` tensor.

10. The loss between the reshaped `output` and `trg` tensors is calculated using the specified `criterion`.

11. The gradients of the loss with respect to the model's parameters are computed using `loss.backward()`.

12. The gradients are then clipped to a maximum value specified by `clip` using `torch.nn.utils.clip_grad_norm_(model.parameters(), clip)`. This prevents the gradients from becoming too large, which can cause issues during optimization.

13. The optimizer's `step()` method is called to update the model's parameters using the computed gradients.

14. The current batch loss (`loss.item()`) is added to the `epoch_loss` variable.

15. After all the batches have been processed, the function returns the average loss per batch for the entire epoch, calculated as `epoch_loss / len(list(iterator))`.


## Evaluating model in PyTorch
You also need to define a function to evaluate the model. Let's go through the code and understand its components:

1. `evaluate(model, iterator, criterion)` takes three arguments:
    - `model` is the neural network model that will be evaluated.
    - `iterator` is an iterable object that provides the evaluation data in batches.
    - `criterion` is the loss function that measures the model's performance.
* Note that evaluate function do not perform any optimization on the model.

2. The function starts by setting the model to evaluation mode with `model.eval()`.

3. It initializes a variable `epoch_loss` to keep track of the accumulated loss during the evaluation.

4. The function enters a `with torch.no_grad()` block, which ensures that no gradients are computed during the evaluation. This saves memory and speeds up the evaluation process since gradients are not needed for parameter updates.

5. The function iterates over the evaluation data provided by the `iterator`. Each iteration retrieves a batch of input sequences (`src`) and target sequences (`trg`).

6. The input sequences (`src`) and target sequences (`trg`) are moved to the appropriate device (e.g., GPU) using `src = src.to(device)` and `trg = trg.to(device)`.

7. The model is then called with `output = model(src, trg, 0)` to obtain the model's predictions for the target sequences. The third argument `0` is passed to indicate that teacher forcing is turned off during evaluation.  During evaluation, teacher forcing is typically turned off to evaluate the model's ability to generate sequences based on its own predictions.

8. The `output` tensor has dimensions `[trg len, batch size, output dim]`. To calculate the loss, the tensor is reshaped to `[trg len - 1, batch size, output dim]` to remove the initial `<bos>` (beginning of sequence) token, which is not used for calculating the loss.

9. The target sequences (`trg`) are also reshaped to `[trg len - 1]` by removing the initial `<bos>` token and making it a contiguous tensor. This matches the shape of the reshaped `output` tensor.

10. The loss between the reshaped `output` and `trg` tensors is calculated using the specified `criterion`.

11. The current batch loss (`loss.item()`) is added to the `epoch_loss` variable.

12. After all the batches have been processed, the function returns the average loss per batch for the entire evaluation, calculated as `epoch_loss / len(list(iterator))`.