# Performance Optimization Guide for torch-rs

This comprehensive guide covers performance optimization strategies for torch-rs, from basic tips to advanced techniques.

## Table of Contents

1. [Overview](#overview)
2. [Memory Optimization](#memory-optimization)
3. [Computational Optimization](#computational-optimization)
4. [Model Architecture](#model-architecture)
5. [Training Optimization](#training-optimization)
6. [Inference Optimization](#inference-optimization)
7. [Hardware Optimization](#hardware-optimization)
8. [Profiling and Debugging](#profiling-and-debugging)
9. [Benchmarking](#benchmarking)
10. [Best Practices](#best-practices)

## Overview

torch-rs provides several performance advantages over traditional Python-based deep learning frameworks:

- 🦀 **Zero-cost abstractions**: Compile-time optimizations with no runtime overhead
- 🛡️ **Memory safety**: No garbage collection pauses or memory leaks
- ⚡ **Native performance**: Direct compilation to machine code
- 🔗 **Efficient interop**: Minimal overhead Python bindings
- 📊 **Predictable performance**: No JIT warm-up or GC spikes

## Memory Optimization

### 1. Tensor Memory Management

#### Avoid Unnecessary Copies

**Bad:**
```rust
let a = Tensor::randn(&[1000, 1000], (Kind::Float, Device::Cpu));
let b = a.clone();  // Full copy
let c = b.clone();  // Another full copy
```

**Good:**
```rust
let a = Tensor::randn(&[1000, 1000], (Kind::Float, Device::Cpu));
let b = a.shallow_clone();  // Shared storage
let c = &a;  // Reference
```

#### Use Views Instead of Copies

**Bad:**
```rust
let reshaped = tensor.reshape(&[new_shape]).detach();  // Copy
```

**Good:**
```rust
let reshaped = tensor.view(&[new_shape]);  // View (no copy)
```

#### In-place Operations

**Bad:**
```rust
let result = &tensor + 1.0;  // Creates new tensor
```

**Good:**
```rust
let _ = tensor.add_(1.0);  // In-place operation
```

### 2. Gradient Memory

#### Clear Gradients Regularly

```rust
// Clear gradients after each optimizer step
optimizer.step()?;
optimizer.zero_grad();  // Free gradient memory
```

#### Use Gradient Accumulation for Large Batches

```rust
let accumulation_steps = 4;
let effective_batch_size = batch_size * accumulation_steps;

for (step, batch) in data_loader.enumerate() {
    let output = model.forward(&batch.input);
    let loss = compute_loss(&output, &batch.target) / accumulation_steps as f64;
    loss.backward();
    
    if (step + 1) % accumulation_steps == 0 {
        optimizer.step()?;
        optimizer.zero_grad();
    }
}
```

### 3. Memory Pool Management

#### Pre-allocate Tensors

```rust
// Pre-allocate working tensors
struct ModelState {
    temp_buffer: Tensor,
    intermediate_result: Tensor,
}

impl ModelState {
    fn new(max_batch_size: i64, hidden_dim: i64) -> Self {
        Self {
            temp_buffer: Tensor::zeros(&[max_batch_size, hidden_dim], 
                                     (Kind::Float, Device::Cpu)),
            intermediate_result: Tensor::zeros(&[max_batch_size, hidden_dim], 
                                             (Kind::Float, Device::Cpu)),
        }
    }
    
    fn process_batch(&mut self, input: &Tensor) -> Tensor {
        // Reuse pre-allocated tensors
        self.temp_buffer.copy_(input);
        // ... processing
        self.intermediate_result.shallow_clone()
    }
}
```

### 4. Memory Layout Optimization

#### Contiguous Memory Access

```rust
// Ensure tensors are contiguous for better cache performance
let tensor = tensor.contiguous();

// Process data in memory order
for batch in data.chunks(batch_size) {
    // Process batch
}
```

## Computational Optimization

### 1. Vectorization and Broadcasting

#### Use Broadcasting Effectively

**Bad:**
```rust
// Element-wise operations in loop
for i in 0..tensor.size()[0] {
    let row = tensor.get(i);
    let result = &row + &bias;  // Inefficient
}
```

**Good:**
```rust
// Vectorized operation with broadcasting
let result = &tensor + &bias;  // Efficient broadcast
```

### 2. Batch Operations

#### Maximize Batch Sizes

```rust
// Find optimal batch size
fn find_optimal_batch_size(model: &impl torch-rsModule, 
                          input_shape: &[i64]) -> i64 {
    let mut batch_size = 1;
    let base_input = Tensor::randn(input_shape, (Kind::Float, Device::Cpu));
    
    loop {
        let test_batch_size = batch_size * 2;
        let mut batch_shape = input_shape.to_vec();
        batch_shape[0] = test_batch_size;
        
        let test_input = Tensor::randn(&batch_shape, (Kind::Float, Device::Cpu));
        
        // Test if this batch size fits in memory
        match std::panic::catch_unwind(|| {
            let _ = model.forward(&test_input);
        }) {
            Ok(_) => batch_size = test_batch_size,
            Err(_) => break,
        }
        
        if batch_size > 1024 { break; }  // Reasonable upper limit
    }
    
    batch_size
}
```

### 3. Efficient Layer Implementations

#### Fused Operations

```rust
// Custom fused layer
struct FusedLinearReLU {
    linear: Linear,
}

impl FusedLinearReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Fused linear + ReLU in single operation
        self.linear.forward(x).relu()
    }
}
```

#### Optimized Attention

```rust
// Efficient attention implementation
impl MultiheadAttention {
    fn forward_optimized(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
        let (batch_size, seq_len, embed_dim) = q.size3().unwrap();
        
        // Use efficient attention implementation
        if seq_len > 1024 {
            // Use sparse attention for long sequences
            self.sparse_attention(q, k, v)
        } else {
            // Use standard attention for short sequences
            self.standard_attention(q, k, v)
        }
    }
}
```

## Model Architecture

### 1. Efficient Model Design

#### Depth vs Width Trade-offs

```rust
// Efficient model architectures

// Wide and shallow (good for inference)
struct WideShallowModel {
    layers: Vec<Linear>,
}

// Narrow and deep (good for accuracy)
struct NarrowDeepModel {
    layers: Vec<Sequential>,
}

// Residual connections for deep models
struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    shortcut: Option<Conv2d>,
}

impl ResidualBlock {
    fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.conv1.forward(x).relu();
        let out = self.conv2.forward(&out);
        
        let shortcut = match &self.shortcut {
            Some(conv) => conv.forward(x),
            None => x.shallow_clone(),
        };
        
        (&out + &shortcut).relu()
    }
}
```

### 2. Parameter Sharing

```rust
// Share parameters across layers
struct TiedEmbedding {
    embedding: Tensor,  // Shared weight matrix
}

impl TiedEmbedding {
    fn embed(&self, input: &Tensor) -> Tensor {
        // Use as embedding
        input.embedding(&self.embedding)
    }
    
    fn project(&self, hidden: &Tensor) -> Tensor {
        // Use as output projection (transpose)
        hidden.matmul(&self.embedding.transpose(0, 1))
    }
}
```

### 3. Model Compression

#### Quantization

```rust
// Quantized linear layer
struct QuantizedLinear {
    weight: Tensor,  // INT8 weights
    scale: f32,
    zero_point: i32,
}

impl QuantizedLinear {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Quantized computation
        let x_quantized = self.quantize_input(x);
        let output_quantized = x_quantized.matmul(&self.weight);
        self.dequantize_output(output_quantized)
    }
}
```

## Training Optimization

### 1. Mixed Precision Training

```rust
// Automatic mixed precision
struct AMPTrainer {
    model: Box<dyn torch-rsModule>,
    optimizer: Box<dyn torch-rsOptimizer>,
    scaler: GradScaler,
}

impl AMPTrainer {
    fn training_step(&mut self, batch: &Batch) -> f64 {
        self.optimizer.zero_grad();
        
        // Forward pass in FP16
        let output = self.model.forward(&batch.input.to_kind(Kind::Half));
        let loss = compute_loss(&output, &batch.target);
        
        // Backward pass with gradient scaling
        let scaled_loss = self.scaler.scale(&loss);
        scaled_loss.backward();
        
        // Optimizer step with unscaling
        self.scaler.step(&mut self.optimizer);
        self.scaler.update();
        
        loss.double_value(&[])
    }
}
```

### 2. Gradient Clipping

```rust
// Efficient gradient clipping
fn clip_gradients(model: &impl torch-rsModule, max_norm: f64) {
    let params = model.parameters();
    let mut total_norm_sq = 0.0;
    
    // Compute total gradient norm
    for param in &params {
        if let Some(grad) = param.grad() {
            let grad_norm = grad.norm().double_value(&[]);
            total_norm_sq += grad_norm * grad_norm;
        }
    }
    
    let total_norm = total_norm_sq.sqrt();
    
    // Clip if necessary
    if total_norm > max_norm {
        let scale = max_norm / total_norm;
        for param in params {
            if let Some(grad) = param.grad() {
                let _ = grad.mul_(scale);
            }
        }
    }
}
```

### 3. Learning Rate Scheduling

```rust
// Warm-up + cosine annealing scheduler
struct WarmupCosineScheduler {
    base_lr: f64,
    warmup_steps: i64,
    total_steps: i64,
    current_step: i64,
}

impl WarmupCosineScheduler {
    fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            // Cosine annealing
            let progress = (self.current_step - self.warmup_steps) as f64 / 
                          (self.total_steps - self.warmup_steps) as f64;
            self.base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }
}
```

## Inference Optimization

### 1. Model Optimization for Inference

```rust
// Optimized inference model
struct InferenceModel {
    model: Box<dyn torch-rsModule>,
    input_cache: Tensor,
    output_cache: Tensor,
}

impl InferenceModel {
    fn new(model: Box<dyn torch-rsModule>, max_batch_size: i64) -> Self {
        let mut inference_model = Self {
            model,
            input_cache: Tensor::zeros(&[max_batch_size, 0], (Kind::Float, Device::Cpu)),
            output_cache: Tensor::zeros(&[max_batch_size, 0], (Kind::Float, Device::Cpu)),
        };
        
        // Set model to evaluation mode
        inference_model.model.set_training(false);
        inference_model
    }
    
    fn predict(&mut self, input: &Tensor) -> Tensor {
        // Reuse cached tensors when possible
        if input.size()[0] <= self.input_cache.size()[0] {
            // Use cached tensor
            let batch_size = input.size()[0];
            let input_slice = self.input_cache.narrow(0, 0, batch_size);
            input_slice.copy_(input);
            
            self.model.forward(&input_slice)
        } else {
            // Direct computation for larger batches
            self.model.forward(input)
        }
    }
}
```

### 2. Batch Processing

```rust
// Efficient batch processing
struct BatchProcessor {
    model: InferenceModel,
    batch_size: usize,
    buffer: Vec<Tensor>,
}

impl BatchProcessor {
    fn process_stream<I>(&mut self, inputs: I) -> Vec<Tensor>
    where
        I: Iterator<Item = Tensor>,
    {
        let mut results = Vec::new();
        
        for input in inputs {
            self.buffer.push(input);
            
            if self.buffer.len() == self.batch_size {
                let batch = Tensor::stack(&self.buffer, 0);
                let output = self.model.predict(&batch);
                
                // Split batch results
                for i in 0..self.batch_size {
                    results.push(output.get(i as i64));
                }
                
                self.buffer.clear();
            }
        }
        
        // Process remaining inputs
        if !self.buffer.is_empty() {
            let batch = Tensor::stack(&self.buffer, 0);
            let output = self.model.predict(&batch);
            
            for i in 0..self.buffer.len() {
                results.push(output.get(i as i64));
            }
        }
        
        results
    }
}
```

## Hardware Optimization

### 1. GPU Optimization

```rust
// Optimal GPU usage
struct GPUOptimizedModel {
    model: Box<dyn torch-rsModule>,
    device: Device,
    stream: CudaStream,
}

impl GPUOptimizedModel {
    fn new(mut model: Box<dyn torch-rsModule>, gpu_id: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let device = Device::Cuda(gpu_id);
        model.to_device(device)?;
        
        Ok(Self {
            model,
            device,
            stream: CudaStream::new()?,
        })
    }
    
    fn forward_async(&self, input: &Tensor) -> Tensor {
        // Use CUDA streams for async execution
        tch::Cuda::set_stream(self.stream.id());
        let result = self.model.forward(input);
        tch::Cuda::synchronize_stream(self.stream.id());
        result
    }
}
```

### 2. Multi-GPU Training

```rust
// Data parallel training
struct DataParallelTrainer {
    models: Vec<Box<dyn torch-rsModule>>,
    devices: Vec<Device>,
    optimizers: Vec<Box<dyn torch-rsOptimizer>>,
}

impl DataParallelTrainer {
    fn training_step(&mut self, batch: &Batch) -> Vec<f64> {
        let batch_size = batch.input.size()[0];
        let per_gpu_batch_size = batch_size / self.devices.len() as i64;
        
        let mut losses = Vec::new();
        
        // Distribute batch across GPUs
        for (i, (model, optimizer)) in self.models.iter_mut()
            .zip(self.optimizers.iter_mut()).enumerate() {
            
            let start = i as i64 * per_gpu_batch_size;
            let end = if i == self.devices.len() - 1 {
                batch_size  // Last GPU gets remainder
            } else {
                start + per_gpu_batch_size
            };
            
            let sub_batch_input = batch.input.narrow(0, start, end - start)
                .to_device(self.devices[i]);
            let sub_batch_target = batch.target.narrow(0, start, end - start)
                .to_device(self.devices[i]);
            
            // Forward and backward pass
            optimizer.zero_grad();
            let output = model.forward(&sub_batch_input);
            let loss = compute_loss(&output, &sub_batch_target);
            loss.backward();
            optimizer.step();
            
            losses.push(loss.double_value(&[]));
        }
        
        // Synchronize gradients across GPUs
        self.all_reduce_gradients();
        
        losses
    }
    
    fn all_reduce_gradients(&mut self) {
        // Implement gradient synchronization
        // This would use NCCL or similar for efficient communication
    }
}
```

### 3. CPU Optimization

```rust
// Multi-threaded CPU training
use rayon::prelude::*;

struct CPUOptimizedTrainer {
    model: Arc<Mutex<Box<dyn torch-rsModule>>>,
    thread_pool: rayon::ThreadPool,
}

impl CPUOptimizedTrainer {
    fn parallel_batch_processing(&self, batches: Vec<Batch>) -> Vec<Tensor> {
        batches.into_par_iter().map(|batch| {
            let model = self.model.lock().unwrap();
            model.forward(&batch.input)
        }).collect()
    }
}
```

## Profiling and Debugging

### 1. Performance Profiling

```rust
// Built-in profiling
use std::time::Instant;

struct Profiler {
    timers: std::collections::HashMap<String, Vec<f64>>,
}

impl Profiler {
    fn time<F, R>(&mut self, name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64();
        
        self.timers.entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
        
        result
    }
    
    fn report(&self) {
        for (name, times) in &self.timers {
            let avg = times.iter().sum::<f64>() / times.len() as f64;
            let min = times.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = times.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            println!("{}: avg={:.4}ms, min={:.4}ms, max={:.4}ms", 
                    name, avg * 1000.0, min * 1000.0, max * 1000.0);
        }
    }
}
```

### 2. Memory Profiling

```rust
// Memory usage tracking
struct MemoryProfiler {
    peak_usage: usize,
    current_usage: usize,
}

impl MemoryProfiler {
    fn track_allocation(&mut self, size: usize) {
        self.current_usage += size;
        self.peak_usage = self.peak_usage.max(self.current_usage);
    }
    
    fn track_deallocation(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
    }
    
    fn report(&self) {
        println!("Peak memory usage: {:.2} MB", self.peak_usage as f64 / 1024.0 / 1024.0);
        println!("Current usage: {:.2} MB", self.current_usage as f64 / 1024.0 / 1024.0);
    }
}
```

## Benchmarking

### Comprehensive Benchmarks

```rust
// Benchmark suite
fn benchmark_model_performance() {
    let model = create_test_model();
    let input = Tensor::randn(&[32, 784], (Kind::Float, Device::Cpu));
    
    // Warm-up
    for _ in 0..10 {
        let _ = model.forward(&input);
    }
    
    // Benchmark forward pass
    let start = Instant::now();
    for _ in 0..100 {
        let _ = model.forward(&input);
    }
    let forward_time = start.elapsed().as_secs_f64() / 100.0;
    
    // Benchmark backward pass
    let start = Instant::now();
    for _ in 0..100 {
        let output = model.forward(&input);
        let loss = output.sum(Kind::Float);
        loss.backward();
    }
    let backward_time = start.elapsed().as_secs_f64() / 100.0;
    
    println!("Forward pass: {:.4}ms", forward_time * 1000.0);
    println!("Backward pass: {:.4}ms", backward_time * 1000.0);
}
```

## Best Practices Summary

### 🚀 Performance Checklist

- [ ] **Memory Management**
  - [ ] Use `shallow_clone()` instead of `clone()` when possible
  - [ ] Prefer views over copies (`view()` vs `reshape()`)
  - [ ] Use in-place operations (`add_()` vs `+`)
  - [ ] Clear gradients regularly (`zero_grad()`)
  - [ ] Pre-allocate working tensors

- [ ] **Computation**
  - [ ] Maximize batch sizes within memory constraints
  - [ ] Use broadcasting effectively
  - [ ] Prefer vectorized operations over loops
  - [ ] Consider fused operations for common patterns

- [ ] **Model Architecture**
  - [ ] Use appropriate model size for dataset
  - [ ] Consider residual connections for deep models
  - [ ] Implement parameter sharing where applicable
  - [ ] Use efficient attention mechanisms for long sequences

- [ ] **Training**
  - [ ] Enable mixed precision training (FP16/BF16)
  - [ ] Use gradient accumulation for large effective batch sizes
  - [ ] Implement gradient clipping for stability
  - [ ] Use appropriate learning rate scheduling

- [ ] **Hardware**
  - [ ] Utilize GPU acceleration (`to_device(Device::Cuda(0))`)
  - [ ] Consider multi-GPU training for large models
  - [ ] Optimize CPU usage with parallel processing
  - [ ] Use appropriate device memory management

- [ ] **Profiling**
  - [ ] Profile memory usage regularly
  - [ ] Benchmark critical paths
  - [ ] Monitor GPU utilization
  - [ ] Track gradient flow and numerical stability

### ⚡ Quick Wins

1. **Enable GPU**: Move model and data to GPU
2. **Increase Batch Size**: Use largest batch that fits in memory
3. **Mixed Precision**: Enable FP16 training
4. **In-place Operations**: Use `_` suffix operations
5. **Gradient Accumulation**: For larger effective batches
6. **Model Compilation**: Use release builds (`--release`)
7. **Memory Pre-allocation**: Reuse tensors in loops
8. **Efficient Data Loading**: Minimize data movement

By following these guidelines, you can achieve optimal performance with torch-rs torch-rs while maintaining code clarity and correctness.