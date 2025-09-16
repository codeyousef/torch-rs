# torch-rs: Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Neural Network Layers](#neural-network-layers)
5. [Optimizers](#optimizers)
6. [Learning Rate Schedulers](#learning-rate-schedulers)
7. [Pre-trained Models](#pre-trained-models)
8. [Training with Lightning](#training-with-lightning)
9. [Python Integration](#python-integration)
10. [Migration from PyTorch](#migration-from-pytorch)
11. [Performance Guide](#performance-guide)
12. [API Reference](#api-reference)

## Introduction

torch-rs brings PyTorch-level functionality to Rust with zero-cost abstractions and memory safety. This guide covers all features implemented in the torch-rs module system.

### Key Features
- 🚀 **PyTorch API Compatibility**: Familiar APIs for PyTorch users
- 🔧 **Zero-cost Abstractions**: No runtime overhead for unused features
- 🛡️ **Memory Safety**: Rust's ownership system prevents memory bugs
- 🐍 **Python Bindings**: Seamless integration with Python via PyO3
- ⚡ **High Performance**: Native Rust performance with CUDA support

## Installation

### Rust Project

Add to your `Cargo.toml`:

```toml
[dependencies]
tch = { version = "0.21.0", features = ["phoenix"] }
tch-vision = "0.21.0"
```

### Python Bindings

```bash
cd python/
pip install setuptools-rust
python setup.py install
```

## Core Concepts

### TorchModule Trait

The foundation of torch-rs is the `TorchModule` trait, which provides automatic parameter discovery:

```rust
use tch::nn::{TorchModule, TorchModuleError};
use tch::{Device, Tensor};
use std::collections::HashMap;

pub trait TorchModule {
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn num_parameters(&self) -> usize;
    fn to_device(&mut self, device: Device) -> Result<(), TorchModuleError>;
    fn set_training(&mut self, training: bool);
    fn is_training(&self) -> bool;
    fn state_dict(&self) -> HashMap<String, Tensor>;
    fn load_state_dict(&mut self, state_dict: &HashMap<String, Tensor>) 
        -> Result<(), TorchModuleError>;
}
```

### Builder Pattern

All layers use the builder pattern for ergonomic construction:

```rust
let model = Sequential::new()
    .add(Linear::builder(784, 128).bias(true).build())
    .add(ReLU::new())
    .add(Dropout::new(0.5))
    .add(Linear::new(128, 10));
```

## Neural Network Layers

### Linear Layers

```rust
use tch::nn::Linear;

// Basic linear layer
let linear = Linear::new(input_dim, output_dim);

// With builder pattern
let linear = Linear::builder(784, 128)
    .bias(false)
    .build();
```

### Convolutional Layers

```rust
use tch::nn::Conv2d;

// 2D Convolution
let conv = Conv2d::builder(3, 64, 3)
    .stride(1)
    .padding(1)
    .dilation(1)
    .groups(1)
    .bias(true)
    .build();
```

### Recurrent Layers

```rust
use tch::nn::layers::{LSTM, GRU};

// LSTM
let lstm = LSTM::builder(input_size, hidden_size)
    .num_layers(2)
    .dropout(0.2)
    .bidirectional(true)
    .batch_first(true)
    .build();

// GRU
let gru = GRU::builder(input_size, hidden_size)
    .num_layers(2)
    .dropout(0.2)
    .build();
```

### Attention Mechanisms

```rust
use tch::nn::layers::MultiheadAttention;

let attention = MultiheadAttention::builder(embed_dim, num_heads)
    .dropout(0.1)
    .batch_first(true)
    .build();

// Forward with mask
let (output, weights) = attention.forward_with_mask(
    &query, &key, &value,
    key_padding_mask,
    need_weights,
    attn_mask
);
```

### Transformer Layers

```rust
use tch::nn::layers::TransformerEncoderLayer;

let encoder = TransformerEncoderLayer::builder(d_model, nhead)
    .dim_feedforward(2048)
    .dropout(0.1)
    .activation(Activation::GELU)
    .layer_norm_eps(1e-5)
    .batch_first(true)
    .norm_first(false)
    .build();
```

### Normalization Layers

```rust
use tch::nn::{BatchNorm1d, BatchNorm2d, LayerNorm};

// Batch normalization
let bn1d = BatchNorm1d::new(num_features);
let bn2d = BatchNorm2d::builder(num_features)
    .eps(1e-5)
    .momentum(0.1)
    .affine(true)
    .track_running_stats(true)
    .build();

// Layer normalization
let ln = LayerNorm::new(&[normalized_shape], 1e-5);
```

### Dropout

```rust
use tch::nn::Dropout;

let dropout = Dropout::new(0.5);
let output = dropout.forward_t(&input, training);
```

## Optimizers

### SGD

```rust
use tch::optim::SGD;

let optimizer = SGD::builder(model.parameters())
    .lr(0.1)
    .momentum(0.9)
    .weight_decay(1e-4)
    .nesterov(true)
    .build()?;
```

### Adam and AdamW

```rust
use tch::optim::{Adam, AdamW};

// Adam
let adam = Adam::builder(model.parameters())
    .lr(1e-3)
    .betas((0.9, 0.999))
    .eps(1e-8)
    .weight_decay(0.0)
    .build()?;

// AdamW (decoupled weight decay)
let adamw = AdamW::new(model.parameters(), AdamWConfig {
    lr: 1e-3,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.01,  // Higher default than Adam
    amsgrad: false,
})?;
```

### RMSprop

```rust
use tch::optim::RMSprop;

let rmsprop = RMSprop::new(model.parameters(), RMSpropConfig {
    lr: 1e-2,
    alpha: 0.99,
    eps: 1e-8,
    weight_decay: 0.0,
    momentum: 0.0,
    centered: false,
})?;
```

### Adagrad

```rust
use tch::optim::Adagrad;

let adagrad = Adagrad::new(model.parameters(), AdagradConfig {
    lr: 1e-2,
    lr_decay: 0.0,
    weight_decay: 0.0,
    initial_accumulator_value: 0.0,
    eps: 1e-10,
})?;
```

## Learning Rate Schedulers

### StepLR

```rust
use tch::optim::{StepLR, LRScheduler};

let mut scheduler = StepLR::new(&mut optimizer, step_size, gamma);

// In training loop
for epoch in 0..num_epochs {
    train_epoch();
    scheduler.step();
}
```

### CosineAnnealingLR

```rust
use tch::optim::CosineAnnealingLR;

let scheduler = CosineAnnealingLR::new(&mut optimizer, t_max, eta_min);
```

### ReduceLROnPlateau

```rust
use tch::optim::ReduceLROnPlateau;

let mut scheduler = ReduceLROnPlateau::new(&mut optimizer)
    .mode(PlateauMode::Min)
    .factor(0.1)
    .patience(10)
    .min_lr(1e-6);

// Step with metric
scheduler.step_with_metric(val_loss);
```

### OneCycleLR

```rust
use tch::optim::OneCycleLR;

let scheduler = OneCycleLR::new(&mut optimizer, max_lr_vec, total_steps)
    .pct_start(0.3)
    .anneal_strategy(AnnealStrategy::Cos)
    .cycle_momentum(true);
```

## Pre-trained Models

### ResNet

```rust
use tch_vision::models::{resnet18, resnet50};
use tch::nn::VarStore;

let vs = VarStore::new(Device::cuda_if_available());

// Create model
let mut model = resnet50(&vs.root(), 1000);

// Load pre-trained weights
model.download_pretrained("resnet50").await?;
```

### Vision Transformer (ViT)

```rust
use tch_vision::models::vit_base_patch16_224;

let model = vit_base_patch16_224(&vs.root(), num_classes);
```

### VGG

```rust
use tch_vision::models::vgg16;

let model = vgg16(&vs.root(), num_classes, true);  // with batch norm
```

## Training with Lightning

### Define a Lightning Module

```rust
use tch::nn::trainer::{LightningModule, Batch, TrainingStepOutput};

struct MyModel {
    network: Sequential,
    // ... other fields
}

impl LightningModule for MyModel {
    fn configure_optimizers(&self) -> (Box<dyn PhoenixOptimizer>, Option<Box<dyn LRScheduler>>) {
        let optimizer = Box::new(Adam::with_lr(self.parameters(), 1e-3).unwrap());
        let scheduler = Box::new(StepLR::new(&mut *optimizer, 30, 0.1));
        (optimizer, Some(scheduler))
    }
    
    fn training_step(&mut self, batch: &Batch, batch_idx: i64) -> TrainingStepOutput {
        let output = self.network.forward(&batch.inputs);
        let loss = output.cross_entropy_loss(&batch.targets);
        
        let mut log = HashMap::new();
        log.insert("train_loss".to_string(), loss.double_value(&[]));
        
        TrainingStepOutput { loss, log }
    }
    
    fn validation_step(&mut self, batch: &Batch, batch_idx: i64) -> ValidationStepOutput {
        // Similar to training_step but without gradients
    }
}
```

### Train with Trainer

```rust
use tch::nn::trainer::{Trainer, TrainerConfig};

let config = TrainerConfig {
    max_epochs: 100,
    gradient_clip_val: Some(1.0),
    accumulate_grad_batches: 4,
    enable_progress_bar: true,
    enable_checkpointing: true,
    early_stopping_patience: Some(10),
    devices: vec![Device::Cuda(0)],
    ..Default::default()
};

let mut trainer = Trainer::new(config);
trainer.fit(&mut model, &mut train_loader, Some(&mut val_loader))?;
```

## Python Integration

### Basic Usage

```python
import torch_rs as trs
import numpy as np

# Tensor operations
a = trs.randn([3, 4])
b = trs.ones([3, 4])
c = a + b
d = a.matmul(b.T)

# NumPy interop
np_array = np.random.randn(2, 3).astype(np.float32)
tensor = trs.from_numpy(np_array)
back = trs.to_numpy(tensor)

# Neural networks
model = trs.Sequential()
model.add(trs.Linear(784, 128))
model.add(trs.Linear(128, 10))

output = model.forward(input_tensor)
```

### Training in Python

```python
# Create optimizer
optimizer = trs.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    for batch_data, batch_target in dataloader:
        # Forward pass
        output = model.forward(batch_data)
        loss = compute_loss(output, batch_target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Migration from PyTorch

### API Mapping

| PyTorch | torch-rs Phoenix |
|---------|------------------|
| `torch.nn.Linear` | `tch::nn::phoenix::Linear` |
| `torch.nn.Conv2d` | `tch::nn::phoenix::Conv2d` |
| `torch.nn.LSTM` | `tch::nn::layers::LSTM` |
| `torch.optim.Adam` | `tch::optim::Adam` |
| `torch.nn.functional.relu` | `tensor.relu()` |
| `model.parameters()` | `model.parameters()` (via PhoenixModule) |
| `model.to(device)` | `model.to_device(device)?` |
| `model.state_dict()` | `model.state_dict()` |

### Code Example Migration

**PyTorch:**
```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
```

**torch-rs Phoenix:**
```rust
use tch::nn::{PhoenixModule, Linear, Sequential};

struct Model {
    network: Sequential,
}

impl Model {
    fn new() -> Self {
        let network = Sequential::new()
            .add(Linear::new(784, 128))
            .add_fn(|x| x.relu())
            .add(Linear::new(128, 10));
        Self { network }
    }
}

impl TorchModule for Model {
    // Automatically derived with #[derive(TorchModule)] macro
}
```

## Performance Guide

### Best Practices

1. **Use Builder Pattern**: Compile-time optimization
   ```rust
   // Good
   let layer = Linear::builder(784, 128).bias(false).build();
   
   // Less optimal
   let mut layer = Linear::new(784, 128);
   layer.set_bias(false);
   ```

2. **Batch Operations**: Process multiple samples together
   ```rust
   // Process batch
   let batch_output = model.forward(&batch_input);
   ```

3. **Device Placement**: Move model to GPU once
   ```rust
   model.to_device(Device::Cuda(0))?;
   ```

4. **Gradient Accumulation**: For large batch sizes
   ```rust
   if step % accumulate_steps == 0 {
       optimizer.step()?;
       optimizer.zero_grad();
   }
   ```

### Memory Optimization

- Use `shallow_clone()` for tensor views
- Call `zero_grad()` to free gradient memory
- Use smaller batch sizes with gradient accumulation
- Enable mixed precision training (F16/BF16)

## API Reference

Complete API documentation is available at:
- Rust docs: Run `cargo doc --open`
- Python docs: See `python/docs/`

## Troubleshooting

### Common Issues

1. **CUDA not available**: Ensure PyTorch is installed with CUDA support
2. **Out of memory**: Reduce batch size or use gradient accumulation
3. **Dimension mismatch**: Check tensor shapes with `.size()`
4. **Gradient not flowing**: Ensure `requires_grad` is set

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - See [LICENSE](LICENSE) file.