#!/usr/bin/env python3
"""Training example using torch-rs Python bindings"""

import torch_rs as trs
import numpy as np
from typing import Tuple
import time


class SimpleDataset:
    """Simple dataset for demonstration"""
    
    def __init__(self, num_samples: int = 1000, input_dim: int = 10):
        # Generate synthetic data
        np.random.seed(42)
        self.X = np.random.randn(num_samples, input_dim).astype(np.float32)
        # Simple linear relationship with noise
        self.weights = np.random.randn(input_dim, 1).astype(np.float32)
        self.y = (self.X @ self.weights + np.random.randn(num_samples, 1).astype(np.float32) * 0.1)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[trs.Tensor, trs.Tensor]:
        x_tensor = trs.from_numpy(self.X[idx:idx+1])
        y_tensor = trs.from_numpy(self.y[idx:idx+1])
        return x_tensor, y_tensor


class SimpleModel:
    """Simple feedforward model"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.model = trs.Sequential()
        self.model.add(trs.Linear(input_dim, hidden_dim))
        # ReLU would go here when implemented
        self.model.add(trs.Linear(hidden_dim, hidden_dim // 2))
        self.model.add(trs.Linear(hidden_dim // 2, output_dim))
        
        # Collect parameters (in practice, would be extracted from model)
        self.parameters = [
            trs.randn([input_dim, hidden_dim]),
            trs.randn([hidden_dim, hidden_dim // 2]),
            trs.randn([hidden_dim // 2, output_dim])
        ]
    
    def forward(self, x: trs.Tensor) -> trs.Tensor:
        return self.model.forward(x)
    
    def get_parameters(self):
        return self.parameters


def compute_mse_loss(pred: trs.Tensor, target: trs.Tensor) -> float:
    """Compute MSE loss (simplified version)"""
    # In practice, this would be implemented in Rust
    pred_np = trs.to_numpy(pred)
    target_np = trs.to_numpy(target)
    return np.mean((pred_np - target_np) ** 2)


def train_epoch(model: SimpleModel, dataloader: trs.DataLoader, 
                optimizer: trs.Adam, epoch: int) -> float:
    """Train for one epoch"""
    total_loss = 0.0
    num_batches = 0
    
    for batch_data, batch_target in dataloader:
        # Forward pass
        predictions = model.forward(batch_data)
        
        # Compute loss (simplified)
        loss_value = compute_mse_loss(predictions, batch_target)
        total_loss += loss_value
        
        # Backward pass (simplified - would use autograd)
        predictions.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
        
        num_batches += 1
        
        if num_batches % 10 == 0:
            print(f"  Batch {num_batches}: Loss = {loss_value:.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model: SimpleModel, dataloader: trs.DataLoader) -> float:
    """Evaluate model on dataset"""
    total_loss = 0.0
    num_batches = 0
    
    for batch_data, batch_target in dataloader:
        predictions = model.forward(batch_data)
        loss_value = compute_mse_loss(predictions, batch_target)
        total_loss += loss_value
        num_batches += 1
    
    return total_loss / num_batches


def main():
    """Main training loop"""
    print("torch-rs Training Example")
    print("="*50)
    
    # Hyperparameters
    input_dim = 10
    hidden_dim = 64
    output_dim = 1
    batch_size = 32
    num_epochs = 5
    learning_rate = 0.001
    
    # Create dataset and dataloader
    print("Creating dataset...")
    train_dataset = SimpleDataset(num_samples=1000, input_dim=input_dim)
    val_dataset = SimpleDataset(num_samples=200, input_dim=input_dim)
    
    train_loader = trs.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = trs.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model and optimizer
    print("Initializing model...")
    model = SimpleModel(input_dim, hidden_dim, output_dim)
    optimizer = trs.Adam(model.get_parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*50)
    
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        
        # Validation
        val_loss = evaluate(model, val_loader)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss!")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })
    
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Print training history
    print("\nTraining History:")
    print("Epoch | Train Loss | Val Loss")
    print("-" * 35)
    for entry in training_history:
        print(f"{entry['epoch']:5d} | {entry['train_loss']:10.4f} | {entry['val_loss']:8.4f}")


if __name__ == "__main__":
    main()