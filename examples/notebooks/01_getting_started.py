#!/usr/bin/env python3
"""
Getting Started with torch-rs
=====================================

This notebook introduces the basic concepts of torch-rs and shows
how to build your first neural network.

Topics covered:
- Tensor operations
- Building neural networks
- Training a simple model
- Comparing with PyTorch
"""

import numpy as np
try:
    import torch_rs as trs
except ImportError:
    print("torch-rs not installed. Install with: pip install torch-rs")
    exit(1)

print("🚀 Welcome to torch-rs!")
print("==============================\n")

# %% [markdown]
# ## 1. Basic Tensor Operations
# 
# Let's start with basic tensor creation and operations, similar to PyTorch.

print("📊 Creating Tensors")
print("-" * 30)

# Create tensors
a = trs.zeros([3, 4])
b = trs.ones([3, 4])
c = trs.randn([3, 4])

print(f"Zeros tensor shape: {a.shape}")
print(f"Ones tensor shape: {b.shape}")
print(f"Random tensor shape: {c.shape}")
print(f"Random tensor device: {c.device}")
print()

# %% [markdown]
# ## 2. Tensor Arithmetic
# 
# torch-rs supports all the standard tensor operations you'd expect.

print("🔢 Tensor Arithmetic")
print("-" * 30)

# Basic operations
d = a + b
e = c * b
f = c.matmul(b.T)  # Matrix multiplication

print(f"Addition result shape: {d.shape}")
print(f"Element-wise multiplication shape: {e.shape}")
print(f"Matrix multiplication shape: {f.shape}")
print()

# %% [markdown]
# ## 3. NumPy Interoperability
# 
# Seamlessly convert between NumPy arrays and torch-rs tensors.

print("🔄 NumPy Integration")
print("-" * 30)

# NumPy to Phoenix
np_array = np.random.randn(2, 3).astype(np.float32)
phoenix_tensor = trs.from_numpy(np_array)
print(f"Converted from NumPy: {phoenix_tensor.shape}")

# Phoenix to NumPy
back_to_numpy = trs.to_numpy(phoenix_tensor)
print(f"Converted back to NumPy: {back_to_numpy.shape}")
print(f"Data preserved: {np.allclose(np_array, back_to_numpy)}")
print()

# %% [markdown]
# ## 4. Building Neural Networks
# 
# torch-rs provides PyTorch-like APIs for building neural networks.

print("🧠 Neural Network Construction")
print("-" * 30)

# Create a simple feedforward network
model = trs.Sequential()
model.add(trs.Linear(784, 128))  # Input layer
# model.add(trs.nn.ReLU())  # Activation (will be added in future versions)
model.add(trs.Linear(128, 64))   # Hidden layer
model.add(trs.Linear(64, 10))    # Output layer

print("Model created with Sequential API")
print()

# %% [markdown]
# ## 5. Forward Pass
# 
# Let's run data through our network.

print("⚡ Forward Pass")
print("-" * 30)

# Create dummy input (batch of MNIST-like images)
input_batch = trs.randn([32, 784])  # 32 samples, 784 features (28x28 flattened)
print(f"Input shape: {input_batch.shape}")

# Forward pass
output = model.forward(input_batch)
print(f"Output shape: {output.shape}")
print(f"Output represents logits for {output.shape[1]} classes")
print()

# Apply softmax to get probabilities
probs = output.softmax(-1)
print(f"Probabilities shape: {probs.shape}")
print()

# %% [markdown]
# ## 6. Simple Training Loop
# 
# Let's implement a basic training loop using torch-rs.

print("🏋️ Training Loop")
print("-" * 30)

# Create optimizer
optimizer = trs.Adam(model.parameters(), lr=0.001)

# Simulate training data
def generate_batch(batch_size=32):
    """Generate synthetic training data"""
    X = trs.randn([batch_size, 784])
    y = trs.Tensor([np.random.randint(0, 10) for _ in range(batch_size)], [batch_size])
    return X, y

# Simple loss function (MSE for demonstration)
def mse_loss(pred, target):
    """Simple MSE loss implementation"""
    # Convert target to one-hot
    target_np = trs.to_numpy(target).astype(int)
    one_hot = np.eye(10)[target_np]
    target_one_hot = trs.from_numpy(one_hot)
    
    # MSE loss
    diff = pred - target_one_hot
    return trs.from_numpy(np.array([np.mean(trs.to_numpy(diff) ** 2)]))

# Training loop
num_epochs = 3
print(f"Training for {num_epochs} epochs...")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 10
    
    for batch_idx in range(num_batches):
        # Get batch
        inputs, targets = generate_batch()
        
        # Forward pass
        outputs = model.forward(inputs)
        loss = mse_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        # Track loss
        loss_value = trs.to_numpy(loss)[0]
        epoch_loss += loss_value
        
        if batch_idx % 5 == 0:
            print(f"  Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss_value:.4f}")
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

print("Training completed!\n")

# %% [markdown]
# ## 7. Model Evaluation
# 
# Let's evaluate our trained model.

print("📊 Model Evaluation")
print("-" * 30)

# Generate test data
test_inputs, test_targets = generate_batch(64)

# Evaluation mode (no gradients)
with torch_rs.no_grad():  # Would be implemented
    test_outputs = model.forward(test_inputs)
    test_probs = test_outputs.softmax(-1)
    
    # Get predictions
    predictions = np.argmax(trs.to_numpy(test_probs), axis=1)
    targets_np = trs.to_numpy(test_targets).astype(int)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == targets_np)
    print(f"Test Accuracy: {accuracy:.2%}")
print()

# %% [markdown]
# ## 8. Comparison with PyTorch
# 
# Let's compare the APIs side by side.

print("⚖️  PyTorch vs Phoenix Comparison")
print("-" * 30)

print("PyTorch API:")
print("""
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.Linear(64, 10)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training step
output = model(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()
optimizer.zero_grad()
""")

print("\nPhoenix API:")
print("""
import torch_rs as trs

model = trs.Sequential()
model.add(trs.Linear(784, 128))
# model.add(trs.nn.ReLU())  # Coming soon
model.add(trs.Linear(128, 64))
model.add(trs.Linear(64, 10))

optimizer = trs.Adam(model.parameters(), lr=0.001)
# criterion would be implemented in future versions

# Training step
output = model.forward(input)
loss = compute_loss(output, target)  # Custom loss for now
loss.backward()
optimizer.step()
optimizer.zero_grad()
""")

print("\n✨ Key Benefits of torch-rs:")
print("- 🦀 Memory safety with Rust")
print("- ⚡ Native performance")
print("- 🔗 Seamless Python integration")
print("- 🎯 PyTorch-compatible APIs")
print("- 🛡️  No runtime overhead")

# %% [markdown]
# ## 9. Device Management
# 
# torch-rs supports both CPU and GPU computation.

print("\n🖥️  Device Management")
print("-" * 30)

# Check current device
print(f"Current tensor device: {c.device}")

# Move to different devices (if available)
try:
    gpu_tensor = c.to("cuda")
    print(f"Moved to GPU: {gpu_tensor.device}")
    
    cpu_tensor = gpu_tensor.to("cpu")
    print(f"Moved back to CPU: {cpu_tensor.device}")
except Exception as e:
    print(f"GPU not available: {e}")

print("\n🎉 Congratulations!")
print("You've completed the torch-rs getting started tutorial!")
print("\nNext steps:")
print("- Try the advanced examples")
print("- Explore the vision models")
print("- Check out the Lightning-style trainer")
print("- Read the complete guide: TORCH_RS_GUIDE.md")