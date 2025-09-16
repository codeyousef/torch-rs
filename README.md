# torch-rs

**A PyTorch-compatible deep learning library for Rust with Python interoperability**

> **Note**: This is a fork of [tch-rs](https://github.com/LaurentMazare/tch-rs) that extends the original crate with comprehensive PyTorch-like APIs, pre-trained models, and seamless Python integration.

While the original `tch-rs` provides thin wrappers around PyTorch's C++ API, **torch-rs** builds upon this foundation to offer:

- 🚀 **Complete PyTorch API compatibility** - Familiar APIs for seamless migration from PyTorch
- 🦀 **Native Rust performance** - Zero-cost abstractions with memory safety guarantees
- 🐍 **Python interoperability** - Seamless integration via PyO3 bindings
- 🧠 **Pre-trained models** - ResNet, VGG, Vision Transformers with automatic weight downloading
- ⚡ **Production-ready training** - Lightning-style trainer with advanced features
- 📊 **Comprehensive ecosystem** - Optimizers, schedulers, metrics, and data transforms

The [documentation](https://docs.rs/tch/) can be found on docs.rs.

[![Build Status](https://github.com/LaurentMazare/tch-rs/workflows/Continuous%20integration/badge.svg)](https://github.com/LaurentMazare/tch-rs/actions)
[![Latest version](https://img.shields.io/crates/v/tch.svg)](https://crates.io/crates/tch)
[![Documentation](https://docs.rs/tch/badge.svg)](https://docs.rs/tch)
[![Dependency Status](https://deps.rs/repo/github/LaurentMazare/tch-rs/status.svg)](https://deps.rs/repo/github/LaurentMazare/tch-rs)
![License](https://img.shields.io/crates/l/tch.svg)

## 🌟 Key Features

- **🎯 PyTorch Compatibility**: Drop-in replacement APIs for PyTorch users
- **🛡️ Memory Safety**: Rust's ownership system prevents memory leaks and segfaults
- **⚡ Zero-Cost Abstractions**: High-level APIs with no runtime performance penalty
- **🔧 Production Ready**: Lightning-style trainer, automatic mixed precision, gradient accumulation
- **🤖 Pre-trained Models**: ResNet, VGG, ViT with one-line model loading
- **🐍 Python Bindings**: Use torch-rs from Python with numpy interoperability
- **📈 Comprehensive**: Full ecosystem of optimizers, schedulers, metrics, and transforms


**Acknowledgments:**
- Original `tch-rs` by [Laurent Mazare](https://github.com/LaurentMazare/tch-rs)
- C API code generation from [ocaml-torch](https://github.com/LaurentMazare/ocaml-torch)
- Inspired by [PyTorch](https://pytorch.org) and [PyTorch Lightning](https://lightning.ai)

## 🚀 Quick Start

### Installation

**Rust:**
```toml
[dependencies]
tch = { version = "0.21.0", features = ["phoenix"] }  # Enable torch-rs features
tch-vision = "0.21.0"  # For computer vision models and transforms
```

**Python:**
```bash
cd python/
pip install setuptools-rust
python setup.py install
```

### Basic Usage

**Rust:**
```rust
use tch::nn::*;
use tch::{Device, Tensor};

fn main() {
    // Create tensors
    let a = Tensor::randn(&[3, 4], (tch::Kind::Float, Device::Cpu));
    let b = Tensor::ones(&[3, 4], (tch::Kind::Float, Device::Cpu));

    // Operations
    let c = &a + &b;
    let d = a.matmul(&b.transpose(0, 1));

    // Build neural networks
    let model = Sequential::new()
        .add(Linear::new(784, 128))
        .add_fn(|x| x.relu())
        .add(Linear::new(128, 10));

    println!("Model created with {} parameters", model.num_parameters());
}
```

**Python:**
```python
import torch_rs as trs
import numpy as np

# Create tensors
a = trs.randn([3, 4])
b = trs.ones([3, 4])

# Operations
c = a + b
d = a.matmul(b.T)

# NumPy interop
np_array = np.random.randn(2, 3).astype(np.float32)
tensor = trs.from_numpy(np_array)

# Neural networks
model = trs.Sequential()
model.add(trs.Linear(784, 128))
model.add(trs.Linear(128, 10))
```

## 🧠 Pre-trained Models

```rust
use tch_vision::models;

// Load ResNet50 with ImageNet weights
let mut model = models::resnet50(&vs.root(), 1000);
model.download_pretrained("resnet50").await?;

// Vision Transformer
let vit = models::vit_base_patch16_224(&vs.root(), 1000);

// VGG with batch normalization
let vgg = models::vgg16(&vs.root(), 1000, true);
```

## 🏋️ Training with Lightning-Style Trainer

```rust
use tch::nn::trainer::{Trainer, TrainerConfig, LightningModule};

// Configure training
let config = TrainerConfig {
    max_epochs: 10,
    gradient_clip_val: Some(1.0),
    enable_progress_bar: true,
    early_stopping_patience: Some(5),
    devices: vec![Device::cuda_if_available()],
    ..Default::default()
};

// Train model
let mut trainer = Trainer::new(config);
trainer.fit(&mut model, &mut train_loader, Some(&mut val_loader))?;
```

## 📚 Documentation

- **[Complete Guide](TORCH_RS_GUIDE.md)** - Comprehensive usage documentation
- **[Migration Guide](MIGRATION_GUIDE.md)** - Migrate from PyTorch to torch-rs
- **[Performance Guide](PERFORMANCE_GUIDE.md)** - Optimization best practices
- **[Python Examples](python/examples/)** - Jupyter-style tutorials
- **[API Reference](https://docs.rs/tch)** - Detailed API documentation

## 🔧 System Requirements

torch-rs requires the C++ PyTorch library (libtorch) in version *v2.8.0* to be available on
your system. You can either:

- Use the system-wide libtorch installation (default).
- Install libtorch manually and let the build script know about it via the `LIBTORCH` environment variable.
- Use a Python PyTorch install, to do this set `LIBTORCH_USE_PYTORCH=1`.
- When a system-wide libtorch can't be found and `LIBTORCH` is not set, the
  build script can download a pre-built binary version of libtorch by using
  the `download-libtorch` feature. By default a CPU version is used. The
  `TORCH_CUDA_VERSION` environment variable can be set to `cu117` in order to
  get a pre-built binary using CUDA 11.7.

### System-wide Libtorch

On linux platforms, the build script will look for a system-wide libtorch
library in `/usr/lib/libtorch.so`.

### Python PyTorch Install

If the `LIBTORCH_USE_PYTORCH` environment variable is set, the active python
interpreter is called to retrieve information about the torch python package.
This version is then linked against.

### Libtorch Manual Install

- Get `libtorch` from the
[PyTorch website download section](https://pytorch.org/get-started/locally/) and extract
the content of the zip file.
- For Linux and macOS users, add the following to your `.bashrc` or equivalent, where `/path/to/libtorch`
is the path to the directory that was created when unzipping the file.
```bash
export LIBTORCH=/path/to/libtorch
```
The header files location can also be specified separately from the shared library via
the following:
```bash
# LIBTORCH_INCLUDE must contain `include` directory.
export LIBTORCH_INCLUDE=/path/to/libtorch/
# LIBTORCH_LIB must contain `lib` directory.
export LIBTORCH_LIB=/path/to/libtorch/
```
- For Windows users, assuming that `X:\path\to\libtorch` is the unzipped libtorch directory.
    - Navigate to Control Panel -> View advanced system settings -> Environment variables.
    - Create the `LIBTORCH` variable and set it to `X:\path\to\libtorch`.
    - Append `X:\path\to\libtorch\lib` to the `Path` variable.

  If you prefer to temporarily set environment variables, in PowerShell you can run
```powershell
$Env:LIBTORCH = "X:\path\to\libtorch"
$Env:Path += ";X:\path\to\libtorch\lib"
```
- You should now be able to run some examples, e.g. `cargo run --example basics`.

### Windows Specific Notes

As per [the pytorch docs](https://pytorch.org/cppdocs/installing.html) the Windows debug and release builds are not ABI-compatible. This could lead to some segfaults if the incorrect version of libtorch is used.

It is recommended to use the MSVC Rust toolchain (e.g. by installing `stable-x86_64-pc-windows-msvc` via rustup) rather than a MinGW based one as PyTorch has compatibilities issues with MinGW.

### Static Linking

When setting environment variable `LIBTORCH_STATIC=1`, `libtorch` is statically
linked rather than using the dynamic libraries. The pre-compiled artifacts don't
seem to include `libtorch.a` by default so this would have to be compiled
manually, e.g. via the following:

```bash
git clone -b v2.8.0 --recurse-submodule https://github.com/pytorch/pytorch.git pytorch-static --depth 1
cd pytorch-static
USE_CUDA=OFF BUILD_SHARED_LIBS=OFF python setup.py build
# export LIBTORCH to point at the build directory in pytorch-static.
```

## Examples

### Basic Tensor Operations

This crate provides a tensor type which wraps PyTorch tensors. Here is a minimal
example of how to perform some tensor operations.

```rust
use tch::Tensor;

fn main() {
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
}
```

### Training a Model via Gradient Descent

PyTorch provides automatic differentiation for most tensor operations
it supports. This is commonly used to train models using gradient
descent. The optimization is performed over variables which are created
via a `nn::VarStore` by defining their shapes and initializations.

In the example below `my_module` uses two variables `x1` and `x2`
which initial values are 0. The forward pass applied to tensor `xs`
returns `xs * x1 + exp(xs) * x2`.

Once the model has been generated, a `nn::Sgd` optimizer is created.
Then on each step of the training loop:

- The forward pass is applied to a mini-batch of data.
- A loss is computed as the mean square error between the model output and the mini-batch ground truth.
- Finally an optimization step is performed: gradients are computed and variables from the `VarStore` are modified accordingly.


```rust
use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Tensor};

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

fn gradient_descent() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..50 {
        // Dummy mini-batches made of zeros.
        let xs = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let ys = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let loss = (my_module.forward(&xs) - ys).pow_tensor_scalar(2).sum(kind::Kind::Float);
        opt.backward_step(&loss);
    }
}
```

### Writing a Simple Neural Network

The `nn` api can be used to create neural network architectures, e.g. the following code defines
a simple model with one hidden layer and trains it on the MNIST dataset using the Adam optimizer.

```rust
use anyhow::Result;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device};

const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;
const LABELS: i64 = 10;

fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            IMAGE_DIM,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

pub fn run() -> Result<()> {
    let m = tch::vision::mnist::load_dir("data")?;
    let vs = nn::VarStore::new(Device::Cpu);
    let net = net(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
    for epoch in 1..200 {
        let loss = net
            .forward(&m.train_images)
            .cross_entropy_for_logits(&m.train_labels);
        opt.backward_step(&loss);
        let test_accuracy = net
            .forward(&m.test_images)
            .accuracy_for_logits(&m.test_labels);
        println!(
            "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
            epoch,
            f64::from(&loss),
            100. * f64::from(&test_accuracy),
        );
    }
    Ok(())
}
```

More details on the training loop can be found in the
[detailed tutorial](https://github.com/LaurentMazare/tch-rs/tree/master/examples/mnist).

### Enhanced PyTorch-Style API with torch-rs

This fork provides enhanced PyTorch-compatible APIs for more intuitive model building:

```rust
use tch::{nn::{TorchModule, Sequential}, Device};

// Build models using the enhanced Sequential API
let model = Sequential::new()
    .add(tch::nn::Linear::new(784, 128))
    .add(tch::nn::ReLU::new())
    .add(tch::nn::Linear::new(128, 10));

// Automatic parameter discovery
let parameters = model.parameters();
let optimizer = tch::optim::Adam::new(parameters, 0.001);

// PyTorch-style training loop
let output = model.forward(&input);
let loss = output.cross_entropy_for_logits(&targets);
loss.backward();
optimizer.step();
optimizer.zero_grad();
```

### Using some Pre-Trained Model

The [pretrained-models example](https://github.com/LaurentMazare/tch-rs/tree/master/examples/pretrained-models/main.rs)
illustrates how to use some pre-trained computer vision model on an image.

**Original tch-rs approach:** Manual weight downloading and loading.
**Enhanced torch-rs approach:** Automatic model downloading with built-in model zoo.

```rust
// Original tch-rs approach
let vs = tch::nn::VarStore::new(tch::Device::Cpu);
let resnet18 = tch::vision::resnet::resnet18(vs.root(), 1000);
vs.load("resnet18.ot")?; // Manual weight file

// Enhanced torch-rs approach (with phoenix feature)
use tch::models::resnet18;
let model = resnet18(Some(1000)); // num_classes
model.download_pretrained("resnet18")?; // Automatic download
```

The example can then be run via the following command:
```bash
cargo run --example pretrained-models -- resnet18.ot tiger.jpg
```
This should print the top 5 imagenet categories for the image. The code for this example is pretty simple.

```rust
    // First the image is loaded and resized to 224x224.
    let image = imagenet::load_image_and_resize(image_file)?;

    // A variable store is created to hold the model parameters.
    let vs = tch::nn::VarStore::new(tch::Device::Cpu);

    // Then the model is built on this variable store, and the weights are loaded.
    let resnet18 = tch::vision::resnet::resnet18(vs.root(), imagenet::CLASS_COUNT);
    vs.load(weight_file)?;

    // Apply the forward pass of the model to get the logits and convert them
    // to probabilities via a softmax.
    let output = resnet18
        .forward_t(&image.unsqueeze(0), /*train=*/ false)
        .softmax(-1);

    // Finally print the top 5 categories and their associated probabilities.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
```
### Importing Pre-Trained Weights from PyTorch Using SafeTensors

`safetensors` is a new simple format by HuggingFace for storing tensors. It does not rely on Python's `pickle` module, and therefore the tensors are not bound to the specific classes and the exact directory structure used when the model is saved. It is also zero-copy, which means that reading the file will require no more memory than the original file.

For more information on `safetensors`, please check out https://github.com/huggingface/safetensors

#### Installing `safetensors`

You can install `safetensors` via the pip manager:

```
pip install safetensors
```

#### Exporting weights in PyTorch

```python
import torchvision
from safetensors import torch as stt

model = torchvision.models.resnet18(pretrained=True)
stt.save_file(model.state_dict(), 'resnet18.safetensors')
```

*Note: the filename of the export must be named with  a `.safetensors` suffix for it to be properly decoded by `tch`.*

#### Importing weights in `tch`

```rust
use anyhow::Result;
use tch::{
	Device,
	Kind,
	nn::VarStore,
	vision::{
		imagenet,
		resnet::resnet18,
	}
};

fn main() -> Result<()> {
	// Create the model and load the pre-trained weights
	let mut vs = VarStore::new(Device::cuda_if_available());
	let model = resnet18(&vs.root(), 1000);
	vs.load("resnet18.safetensors")?;
	
	// Load the image file and resize it to the usual imagenet dimension of 224x224.
	let image = imagenet::load_image_and_resize224("dog.jpg")?
		.to_device(vs.device());

	// Apply the forward pass of the model to get the logits
	let output = image
		.unsqueeze(0)
		.apply_t(&model, false)
		.softmax(-1, Kind::Float);
	
	// Print the top 5 categories for this image.
    for (probability, class) in imagenet::top(&output, 5).iter() {
        println!("{:50} {:5.2}%", class, 100.0 * probability)
    }
    
    Ok(())
}
```

Further examples include:
* A simplified version of
  [char-rnn](https://github.com/LaurentMazare/tch-rs/blob/master/examples/char-rnn)
  illustrating character level language modeling using Recurrent Neural Networks.
* [Neural style transfer](https://github.com/LaurentMazare/tch-rs/blob/master/examples/neural-style-transfer)
  uses a pre-trained VGG-16 model to compose an image in the style of another image.
* Some [ResNet examples on CIFAR-10](https://github.com/LaurentMazare/tch-rs/tree/master/examples/cifar).
* A [tutorial](https://github.com/LaurentMazare/tch-rs/tree/master/examples/jit)
  showing how to deploy/run some Python trained models using
  [TorchScript JIT](https://pytorch.org/docs/stable/jit.html).
* Some [Reinforcement Learning](https://github.com/LaurentMazare/tch-rs/blob/master/examples/reinforcement-learning)
  examples using the [OpenAI Gym](https://github.com/openai/gym) environment. This includes a policy gradient
  example as well as an A2C implementation that can run on Atari games.
* A [Transfer Learning Tutorial](https://github.com/LaurentMazare/tch-rs/blob/master/examples/transfer-learning)
  shows how to finetune a pre-trained ResNet model on a very small dataset.
* A [simplified version of GPT](https://github.com/LaurentMazare/tch-rs/blob/master/examples/min-gpt)
  similar to minGPT.
* A [Stable Diffusion](https://github.com/LaurentMazare/diffusers-rs)
  implementation following the lines of hugginface's diffusers library.

### Enhanced torch-rs Examples

This fork includes additional examples showcasing the enhanced PyTorch-compatible features:

* **Lightning-Style Training**: Examples using the `LightningModule` trait for organized training loops
* **Computer Vision Pipeline**: Complete CV workflows with transforms, data loading, and pre-trained models
* **Python Integration**: Jupyter notebooks demonstrating seamless Python-Rust interoperability
* **Model Zoo Usage**: Examples using the built-in model downloader for ResNet, VGG, and Vision Transformers
* **Advanced Optimizers**: Demonstrations of AdamW, learning rate schedulers, and mixed precision training

See the `examples/notebooks/` directory for interactive tutorials.

External material:
* A [tutorial](http://vegapit.com/article/how-to-use-torch-in-rust-with-tch-rs) showing how to use Torch to compute option prices and greeks.
* [tchrs-opencv-webcam-inference](https://github.com/metobom/tchrs-opencv-webcam-inference) uses `tch-rs` and `opencv` to run inference
  on a webcam feed for some Python trained model based on mobilenet v3.

## FAQ

### What are the best practices for Python to Rust model translations?

See some details in [this thread](https://github.com/LaurentMazare/tch-rs/issues/549#issuecomment-1296840898).

### How to get this to work on a M1/M2 mac?

Check this [issue](https://github.com/LaurentMazare/tch-rs/issues/488).

### Compilation is slow, torch-sys seems to be rebuilt every time cargo gets run.
See this [issue](https://github.com/LaurentMazare/tch-rs/issues/596), this could
be caused by rust-analyzer not knowing about the proper environment variables
like `LIBTORCH` and `LD_LIBRARY_PATH`.

### Using Rust/torch-rs code from Python.
This fork provides enhanced Python integration via PyO3. With the `python-compat` feature enabled,
you can seamlessly use torch-rs models and tensors from Python with NumPy interoperability.

```python
import torch_rs as trs
import numpy as np

# Create models in Python using Rust backend
model = trs.Sequential()
model.add(trs.Linear(784, 128))
model.add(trs.Linear(128, 10))

# Convert between NumPy and torch-rs tensors
np_array = np.random.randn(32, 784)
tensor = trs.from_numpy(np_array)
result = model.forward(tensor)
output = trs.to_numpy(result)
```

The original [tch-ext](https://github.com/LaurentMazare/tch-ext) example still applies for basic PyO3 integration.

### Error loading shared libraries. 

If you get an error about not finding some shared libraries when running the generated binaries
(e.g. 
` error while loading shared libraries: libtorch_cpu.so: cannot open shared object file: No such file or directory`).
You can try adding the following to your `.bashrc` where `/path/to/libtorch` is the path to your
libtorch install.
```
# For Linux
export LD_LIBRARY_PATH=/path/to/libtorch/lib:$LD_LIBRARY_PATH
# For macOS
export DYLD_LIBRARY_PATH=/path/to/libtorch/lib:$DYLD_LIBRARY_PATH
```

## Acknowledgments

This project is a fork of the excellent [tch-rs](https://github.com/LaurentMazare/tch-rs) crate by Laurent Mazare. The original tch-rs provides the foundational PyTorch bindings that make this enhanced version possible.

### What's New in torch-rs

- **PyTorch-Compatible APIs**: Enhanced Sequential, Module, and training abstractions
- **Pre-trained Model Zoo**: Automatic downloading and loading of popular CV models
- **Lightning-Style Training**: Organized training loops with built-in metrics and logging
- **Python Integration**: Seamless NumPy interoperability and Python bindings
- **Advanced Optimizers**: AdamW, learning rate scheduling, and mixed precision support
- **Complete Documentation**: Comprehensive guides, tutorials, and API reference

## License
`torch-rs` is distributed under the terms of both the MIT license
and the Apache license (version 2.0), at your option, maintaining compatibility
with the original tch-rs license.

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT) for more
details.
