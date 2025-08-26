# Optimization Techniques for Efficient Transformer Training

Training transformers is costly, but several techniques improve efficiency:

## Gradient Accumulation
- Accumulates gradients over mini-batches before updating.
- Enables larger effective batch sizes with less memory.

## Mixed-Precision Training
- Uses FP16 for some operations to cut memory and speed up training.
- AMP (PyTorch/TensorFlow) automates precision choice.

## Distributed Training
- **Data Parallelism:** Split data across devices, sync gradients.
- **Model Parallelism:** Split model across devices when too large.

## Efficient Optimizers
- **AdamW:** Adaptive learning + weight decay for better generalization.
- **LAMB:** Scales well with very large batches.

---