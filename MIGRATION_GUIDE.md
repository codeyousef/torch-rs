# Migration Guide: PyTorch to torch-rs

This guide helps PyTorch users migrate to torch-rs, covering API differences, best practices, and common patterns.

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Core Concepts](#core-concepts)
3. [Tensor Operations](#tensor-operations)
4. [Neural Network Layers](#neural-network-layers)
5. [Training Loop](#training-loop)
6. [Optimizers](#optimizers)
7. [Data Loading](#data-loading)
8. [Model Management](#model-management)
9. [Advanced Features](#advanced-features)
10. [Performance Considerations](#performance-considerations)
11. [Common Pitfalls](#common-pitfalls)
12. [Examples](#examples)

## Quick Reference

### Package Imports

**PyTorch:**
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
```

**torch-rs Python:**
```python
import torch_rs as trs
from torch_rs import nn, optim, functional as F
import torch_rs_vision
```

**torch-rs Rust:**
```rust
use tch::{nn, Device, Tensor};
use tch::nn::*;
use tch::optim::*;
use tch_vision::models;
```

### API Mapping

| PyTorch | torch-rs Python | torch-rs Rust |
|---------|----------------|-------------|
| `torch.tensor()` | `trs.tensor()` | `Tensor::of_slice()` |
| `torch.zeros()` | `trs.zeros()` | `Tensor::zeros()` |
| `torch.randn()` | `trs.randn()` | `Tensor::randn()` |
| `nn.Linear()` | `trs.Linear()` | `Linear::new()` |
| `nn.Conv2d()` | `trs.Conv2d()` | `Conv2d::new()` |
| `nn.LSTM()` | `trs.LSTM()` | `LSTM::new()` |
| `optim.Adam()` | `trs.Adam()` | `Adam::new()` |
| `F.relu()` | `tensor.relu()` | `tensor.relu()` |
| `model.parameters()` | `model.parameters()` | `model.parameters()` |

## Core Concepts

### Memory Management

**PyTorch:** Automatic garbage collection
```python
x = torch.randn(1000, 1000)  # GC handles cleanup
del x  # Optional
```

**torch-rs:** Rust's ownership system
```rust
let x = Tensor::randn(&[1000, 1000], (Kind::Float, Device::Cpu));
// x is automatically dropped when out of scope
// No manual memory management needed
```

### Error Handling

**PyTorch:** Exceptions
```python
try:
    result = model(input)
except RuntimeError as e:
    print(f"Error: {e}")
```

**torch-rs:** Result types
```rust
match model.forward(&input) {
    Ok(result) => println!("Success: {:?}", result.size()),
    Err(e) => println!("Error: {:?}", e),
}
```

### Device Management

**PyTorch:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
tensor = tensor.to(device)
```

**torch-rs:**
```rust
let device = Device::cuda_if_available();
model.to_device(device)?;
let tensor = tensor.to_device(device);
```

## Tensor Operations

### Creation

**PyTorch:**
```python
# Various creation methods
zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)
randn = torch.randn(3, 4)
from_list = torch.tensor([[1, 2], [3, 4]])
from_numpy = torch.from_numpy(np_array)
```

**torch-rs Rust:**
```rust
// Equivalent operations
let zeros = Tensor::zeros(&[3, 4], (Kind::Float, Device::Cpu));
let ones = Tensor::ones(&[3, 4], (Kind::Float, Device::Cpu));
let randn = Tensor::randn(&[3, 4], (Kind::Float, Device::Cpu));
let from_slice = Tensor::of_slice(&[1, 2, 3, 4]).reshape(&[2, 2]);
// NumPy interop available via Python bindings
```

### Operations

**PyTorch:**
```python
# Arithmetic
c = a + b
d = torch.matmul(a, b)
e = torch.nn.functional.relu(a)

# Shape manipulation
reshaped = a.view(2, -1)
transposed = a.transpose(0, 1)
```

**torch-rs:**
```rust
// Arithmetic
let c = &a + &b;
let d = a.matmul(&b);
let e = a.relu();

// Shape manipulation
let reshaped = a.view(&[2, -1]);
let transposed = a.transpose(0, 1);
```

### Indexing and Slicing

**PyTorch:**
```python
# Indexing
subset = tensor[0:2, :, 1:3]
item = tensor[0, 0].item()

# Boolean indexing
mask = tensor > 0.5
filtered = tensor[mask]
```

**torch-rs:**
```rust
// Indexing
let subset = tensor.narrow(0, 0, 2).narrow(2, 1, 2);
let item: f64 = tensor.double_value(&[0, 0]);

// Boolean indexing
let mask = tensor.gt(0.5);
let filtered = tensor.masked_select(&mask);
```

## Neural Network Layers

### Basic Layers

**PyTorch:**
```python
import torch.nn as nn

# Linear layer
linear = nn.Linear(784, 128, bias=True)

# Convolution
conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

# Batch normalization
bn = nn.BatchNorm2d(64)

# Dropout
dropout = nn.Dropout(0.5)
```

**torch-rs:**
```rust
use tch::nn::*;

// Linear layer
let linear = Linear::builder(784, 128)
    .bias(true)
    .build();

// Convolution
let conv = Conv2d::builder(3, 64, 3)
    .stride(1)
    .padding(1)
    .build();

// Batch normalization
let bn = BatchNorm2d::new(64);

// Dropout
let dropout = Dropout::new(0.5);
```

### Sequential Models

**PyTorch:**
```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 10)
)
```

**torch-rs:**
```rust
let model = Sequential::new()
    .add(Linear::new(784, 256))
    .add_fn(|x| x.relu())
    .add(Dropout::new(0.5))
    .add(Linear::new(256, 10));
```

### Custom Modules

**PyTorch:**
```python
class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

**torch-rs:**
```rust
#[derive(PhoenixModule)]  // Auto-implements parameter discovery
struct CustomModel {
    fc1: Linear,
    fc2: Linear,
    dropout: Dropout,
}

impl CustomModel {
    fn new(input_dim: i64, hidden_dim: i64, output_dim: i64) -> Self {
        Self {
            fc1: Linear::new(input_dim, hidden_dim),
            fc2: Linear::new(hidden_dim, output_dim),
            dropout: Dropout::new(0.5),
        }
    }
    
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.fc1.forward(x).relu();
        let x = self.dropout.forward_t(&x, self.training);
        self.fc2.forward(&x)
    }
}
```

## Training Loop

### Basic Training

**PyTorch:**
```python
model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

**torch-rs:**
```rust
let mut model = CustomModel::new(784, 256, 10);
let mut optimizer = Adam::with_lr(model.parameters_mut(), 0.001)?;

for epoch in 0..num_epochs {
    for (batch_idx, (data, target)) in train_loader.enumerate() {
        optimizer.zero_grad();
        let output = model.forward(&data);
        let loss = output.cross_entropy_loss(&target);
        loss.backward();
        optimizer.step()?;
    }
}
```

### Lightning-Style Training

**PyTorch Lightning:**
```python
import pytorch_lightning as pl

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MyModel()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, train_dataloader)
```

**Phoenix Trainer:**
```rust
use tch::nn::trainer::*;

impl LightningModule for CustomModel {
    fn configure_optimizers(&self) -> (Box<dyn PhoenixOptimizer>, Option<Box<dyn LRScheduler>>) {
        let optimizer = Box::new(Adam::with_lr(self.parameters(), 1e-3).unwrap());
        (optimizer, None)
    }
    
    fn training_step(&mut self, batch: &Batch, batch_idx: i64) -> TrainingStepOutput {
        let output = self.forward(&batch.inputs);
        let loss = output.cross_entropy_loss(&batch.targets);
        TrainingStepOutput {
            loss,
            log: HashMap::new(),
        }
    }
}

let config = TrainerConfig::default();
let mut trainer = Trainer::new(config);
trainer.fit(&mut model, &mut train_loader, Some(&mut val_loader))?;
```

## Optimizers

### Optimizer Configuration

**PyTorch:**
```python
# Adam
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# SGD with momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

**torch-rs:**
```rust
// Adam
let optimizer = Adam::new(model.parameters_mut(), AdamConfig {
    lr: 1e-3,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.01,
    amsgrad: false,
})?;

// SGD with momentum
let optimizer = SGD::builder(model.parameters_mut())
    .lr(0.1)
    .momentum(0.9)
    .weight_decay(1e-4)
    .nesterov(true)
    .build()?;
```

### Learning Rate Scheduling

**PyTorch:**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# In training loop
for epoch in range(num_epochs):
    train_epoch()
    scheduler.step()
```

**torch-rs:**
```rust
let mut optimizer = Adam::with_lr(model.parameters_mut(), 1e-3)?;
let mut scheduler = StepLR::new(&mut optimizer, 30, 0.1);

// In training loop
for epoch in 0..num_epochs {
    train_epoch();
    scheduler.step();
}
```

## Data Loading

### DataLoader Equivalent

**PyTorch:**
```python
from torch.utils.data import DataLoader, Dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataloader = DataLoader(
    CustomDataset(train_data, train_labels),
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

**torch-rs Python:**
```python
class CustomDataset:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return trs.from_numpy(self.data[idx]), trs.from_numpy(self.labels[idx])

dataloader = trs.DataLoader(
    CustomDataset(train_data, train_labels),
    batch_size=32,
    shuffle=True
)
```

## Model Management

### Saving and Loading

**PyTorch:**
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

**torch-rs:**
```rust
// Save model
let state_dict = model.state_dict();
// Serialization would be implemented

// Load model
let mut model = CustomModel::new(784, 256, 10);
model.load_state_dict(&state_dict)?;
model.set_training(false);
```

### Pre-trained Models

**PyTorch:**
```python
import torchvision.models as models

# Load pre-trained ResNet
model = models.resnet50(pretrained=True)

# Modify for different number of classes
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

**torch-rs:**
```rust
use tch_vision::models;

// Load pre-trained ResNet
let mut model = models::resnet50(&vs.root(), 1000);
model.download_pretrained("resnet50").await?;

// Modify classifier
// Would require model modification API
```

## Advanced Features

### Mixed Precision Training

**PyTorch:**
```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**torch-rs:**
```rust
// Mixed precision support in trainer config
let config = TrainerConfig {
    precision: Precision::F16,
    ..Default::default()
};

let trainer = Trainer::new(config);
```

### Gradient Clipping

**PyTorch:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**torch-rs:**
```rust
// Built into trainer
let config = TrainerConfig {
    gradient_clip_val: Some(1.0),
    ..Default::default()
};
```

## Performance Considerations

### Memory Usage

**PyTorch Best Practices:**
- Use `torch.no_grad()` for inference
- Clear gradients with `optimizer.zero_grad()`
- Use `del` for large tensors
- Enable memory efficient attention

**Phoenix Advantages:**
- Automatic memory management via Rust
- No garbage collection overhead
- Compile-time memory safety
- Zero-cost abstractions

### Speed Optimizations

**PyTorch:**
```python
# JIT compilation
model = torch.jit.script(model)

# Enable cudNN benchmarking
torch.backends.cudnn.benchmark = True

# Use DataLoader workers
dataloader = DataLoader(dataset, num_workers=4)
```

**torch-rs:**
```rust
// Built-in optimizations
// - Zero-cost abstractions
// - Compile-time optimizations
// - Efficient memory layout
// - Native performance
```

## Common Pitfalls

### 1. Tensor Ownership in Rust

**Problem:** Borrowing issues with tensors
```rust
// This won't compile
let a = Tensor::randn(&[3, 3], (Kind::Float, Device::Cpu));
let b = a.view(&[9]);  // Borrowing a
let c = a + 1.0;       // Error: a already borrowed
```

**Solution:** Use references or clone appropriately
```rust
let a = Tensor::randn(&[3, 3], (Kind::Float, Device::Cpu));
let b = a.view(&[9]);
let c = &a + 1.0;      // Use reference
// or
let c = a.shallow_clone() + 1.0;  // Shallow clone
```

### 2. Device Mismatches

**Problem:** Tensors on different devices
```rust
let a = Tensor::randn(&[3, 3], (Kind::Float, Device::Cpu));
let b = Tensor::randn(&[3, 3], (Kind::Float, Device::Cuda(0)));
let c = &a + &b;  // Error: device mismatch
```

**Solution:** Ensure consistent device placement
```rust
let device = Device::cuda_if_available();
let a = Tensor::randn(&[3, 3], (Kind::Float, device));
let b = Tensor::randn(&[3, 3], (Kind::Float, device));
let c = &a + &b;  // OK
```

### 3. Gradient Computation

**Problem:** Forgetting to enable gradients
```rust
let x = Tensor::randn(&[3, 3], (Kind::Float, Device::Cpu));
// x.requires_grad() is false by default
let y = &x * &x;
y.backward();  // No gradients computed
```

**Solution:** Enable gradients explicitly
```rust
let x = Tensor::randn(&[3, 3], (Kind::Float, Device::Cpu))
    .set_requires_grad(true);
let y = &x * &x;
y.backward();  // Gradients computed
```

## Complete Migration Example

Here's a complete example showing a PyTorch model migrated to torch-rs:

### PyTorch Version

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Training
model = MLP(784, 256, 10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 784)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### torch-rs Rust Version

```rust
use tch::{nn, nn::*, Device, Tensor};
use tch::optim::*;

#[derive(PhoenixModule)]
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new(input_size: i64, hidden_size: i64, num_classes: i64) -> Self {
        Self {
            fc1: Linear::new(input_size, hidden_size),
            fc2: Linear::new(hidden_size, num_classes),
        }
    }
    
    fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.fc1.forward(x).relu();
        self.fc2.forward(&out)
    }
}

// Training
let mut model = MLP::new(784, 256, 10);
let mut optimizer = Adam::with_lr(model.parameters_mut(), 1e-3)?;

for epoch in 0..num_epochs {
    for (images, labels) in train_loader {
        let images = images.reshape(&[-1, 784]);
        
        optimizer.zero_grad();
        let outputs = model.forward(&images);
        let loss = outputs.cross_entropy_loss(&labels);
        loss.backward();
        optimizer.step()?;
    }
}
```

## Conclusion

Migrating from PyTorch to Phoenix offers several advantages:

- 🦀 **Memory Safety**: No segfaults or memory leaks
- ⚡ **Performance**: Native speed without Python overhead
- 🛡️ **Type Safety**: Compile-time error checking
- 🎯 **Compatibility**: Familiar PyTorch-like APIs
- 🔗 **Interop**: Seamless Python integration when needed

While there's a learning curve for Rust-specific concepts like ownership and borrowing, the torch-rs system provides familiar abstractions that make the transition smoother for PyTorch users.

For more detailed information, see:
- [torch-rs Guide](TORCH_RS_GUIDE.md)
- [API Documentation](https://docs.rs/tch)
- [Examples](examples/)
- [Python Bindings](python/README.md)