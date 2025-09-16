#!/usr/bin/env python3
"""Basic usage example of torch-rs Python bindings"""

import torch_rs as trs
import numpy as np


def tensor_operations():
    """Demonstrate basic tensor operations"""
    print("Basic Tensor Operations")
    print("="*50)
    
    # Create tensors
    a = trs.randn([3, 4])
    b = trs.ones([3, 4])
    
    print(f"Tensor a shape: {a.shape}")
    print(f"Tensor b shape: {b.shape}")
    
    # Arithmetic operations
    c = a + b
    d = a * b
    
    print(f"Addition result shape: {c.shape}")
    print(f"Multiplication result shape: {d.shape}")
    
    # Activations
    relu_out = a.relu()
    sigmoid_out = a.sigmoid()
    
    print(f"ReLU output shape: {relu_out.shape}")
    print(f"Sigmoid output shape: {sigmoid_out.shape}")
    
    # Matrix multiplication
    e = trs.randn([3, 4])
    f = trs.randn([4, 5])
    g = e.matmul(f)
    print(f"Matmul result shape: {g.shape}")
    print()


def numpy_interop():
    """Demonstrate numpy interoperability"""
    print("NumPy Interoperability")
    print("="*50)
    
    # NumPy to torch-rs
    np_array = np.random.randn(2, 3).astype(np.float32)
    tensor = trs.from_numpy(np_array)
    print(f"Converted tensor shape: {tensor.shape}")
    
    # torch-rs to NumPy
    tensor2 = trs.randn([4, 5])
    np_array2 = trs.to_numpy(tensor2)
    print(f"Converted numpy array shape: {np_array2.shape}")
    print()


def neural_network():
    """Demonstrate neural network creation"""
    print("Neural Network Example")
    print("="*50)
    
    # Create a simple network
    model = trs.Sequential()
    model.add(trs.Linear(10, 20))
    # model.add(trs.nn.ReLU())  # Would be added when implemented
    model.add(trs.Linear(20, 10))
    model.add(trs.Linear(10, 1))
    
    # Forward pass
    input_tensor = trs.randn([32, 10])  # Batch of 32, 10 features
    output = model.forward(input_tensor)
    print(f"Model output shape: {output.shape}")
    
    # Create optimizer
    # In practice, we'd extract parameters from the model
    dummy_params = [trs.randn([10, 20]), trs.randn([20, 10]), trs.randn([10, 1])]
    optimizer = trs.Adam(dummy_params, lr=0.001)
    
    # Simulate training step
    loss = output  # In practice, compute actual loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print("Training step completed")
    print()


def device_management():
    """Demonstrate device management"""
    print("Device Management")
    print("="*50)
    
    # Create tensor on CPU
    cpu_tensor = trs.randn([3, 3])
    print(f"CPU tensor device: {cpu_tensor.device}")
    
    # Move to GPU if available
    try:
        cuda_tensor = cpu_tensor.to("cuda")
        print(f"CUDA tensor device: {cuda_tensor.device}")
    except Exception as e:
        print(f"CUDA not available: {e}")
    
    # Move back to CPU
    cpu_tensor2 = cpu_tensor.to("cpu")
    print(f"Moved back to CPU: {cpu_tensor2.device}")
    print()


if __name__ == "__main__":
    print("torch-rs Python Bindings Example")
    print("="*50)
    print()
    
    tensor_operations()
    numpy_interop()
    neural_network()
    device_management()
    
    print("All examples completed successfully!")