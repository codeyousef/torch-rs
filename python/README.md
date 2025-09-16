# torch-rs Python Bindings

Python bindings for torch-rs, providing PyTorch-like functionality implemented in Rust.

## Installation

### From Source

```bash
# Install build dependencies
pip install setuptools-rust

# Build and install
cd python/
python setup.py install
```

### Development Mode

```bash
# Install in development mode for easier iteration
pip install -e .
```

## Quick Start

```python
import torch_rs as trs
import numpy as np

# Create tensors
a = trs.randn([3, 4])
b = trs.ones([3, 4])

# Operations
c = a + b
d = a.matmul(b.T)  # Matrix multiplication

# Activations
relu_out = a.relu()
sigmoid_out = a.sigmoid()

# NumPy interoperability
np_array = np.random.randn(2, 3).astype(np.float32)
tensor = trs.from_numpy(np_array)
back_to_numpy = trs.to_numpy(tensor)

# Neural networks
model = trs.Sequential()
model.add(trs.Linear(10, 20))
model.add(trs.Linear(20, 1))

# Forward pass
input_tensor = trs.randn([32, 10])
output = model.forward(input_tensor)

# Optimization
optimizer = trs.Adam(model.parameters(), lr=0.001)
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## Features

### Tensor Operations
- Creation: `zeros`, `ones`, `randn`, `arange`
- Arithmetic: `+`, `-`, `*`, `/`, `matmul`
- Activations: `relu`, `sigmoid`, `softmax`
- Device management: CPU/CUDA support
- Automatic differentiation

### Neural Network Layers
- `Linear`: Fully connected layer
- `Sequential`: Container for sequential layers
- More layers coming soon: Conv2d, BatchNorm, Dropout, etc.

### Optimizers
- `Adam`: Adaptive moment estimation
- Coming soon: SGD, AdamW, RMSprop

### Data Utilities
- `DataLoader`: Batched data iteration
- NumPy interoperability

## Examples

See the `examples/` directory for complete examples:
- `basic_usage.py`: Basic tensor operations and neural networks
- `train_model.py`: Training a simple model

## Architecture

The Python bindings use PyO3 to wrap the Rust implementation, providing:
- Zero-copy tensor sharing where possible
- Automatic memory management
- Thread-safe operations
- GPU acceleration through CUDA

## Performance

Being implemented in Rust, torch-rs provides:
- Memory safety without garbage collection
- Efficient parallelization
- Minimal Python overhead
- Native performance for tensor operations

## Development

### Building from Source

```bash
# Debug build
maturin develop

# Release build
maturin develop --release

# Run tests
python -m pytest tests/
```

### Adding New Features

1. Implement the feature in Rust (`src/lib.rs`)
2. Add Python wrapper in `torch_rs/__init__.py`
3. Write tests in `tests/`
4. Update documentation

## Limitations

This is an alpha release. Current limitations include:
- Limited layer types (more coming soon)
- Basic autograd support
- Limited optimizer selection
- Some operations not yet implemented

## Contributing

Contributions are welcome! Please see the main torch-rs repository for contribution guidelines.

## License

MIT License - See LICENSE file in the root directory.