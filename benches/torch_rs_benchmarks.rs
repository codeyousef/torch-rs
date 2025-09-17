//! Benchmarking suite for torch-rs
//!
//! Performance comparison between torch-rs and PyTorch

#![feature(test)]
extern crate test;

use tch::nn::layers::*;
use tch::nn::*;
use tch::optim::*;
use tch::{nn::*, Device, Kind, Tensor};
use test::Bencher;

// Tensor operation benchmarks
#[bench]
fn bench_tensor_matmul_small(b: &mut Bencher) {
    let a = Tensor::randn(&[128, 256], (Kind::Float, Device::Cpu));
    let b = Tensor::randn(&[256, 512], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = a.matmul(&b);
    });
}

#[bench]
fn bench_tensor_matmul_large(b: &mut Bencher) {
    let a = Tensor::randn(&[1024, 2048], (Kind::Float, Device::Cpu));
    let b = Tensor::randn(&[2048, 4096], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = a.matmul(&b);
    });
}

#[bench]
fn bench_tensor_conv2d(b: &mut Bencher) {
    let input = Tensor::randn(&[32, 3, 224, 224], (Kind::Float, Device::Cpu));
    let weight = Tensor::randn(&[64, 3, 7, 7], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = input.conv2d(&weight, None, &[2, 2], &[3, 3], &[1, 1], 1);
    });
}

// Layer benchmarks
#[bench]
fn bench_linear_forward(b: &mut Bencher) {
    let layer = Linear::new(1024, 2048);
    let input = Tensor::randn(&[128, 1024], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = layer.forward(&input);
    });
}

#[bench]
fn bench_conv2d_forward(b: &mut Bencher) {
    let layer = Conv2d::builder(3, 64, 3).stride(1).padding(1).build();
    let input = Tensor::randn(&[8, 3, 224, 224], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = layer.forward(&input);
    });
}

#[bench]
fn bench_lstm_forward(b: &mut Bencher) {
    let lstm = LSTM::builder(512, 256).num_layers(2).batch_first(true).build();
    let input = Tensor::randn(&[32, 100, 512], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = lstm.forward(&input);
    });
}

#[bench]
fn bench_multihead_attention(b: &mut Bencher) {
    let attention = MultiheadAttention::builder(512, 8).build();
    let input = Tensor::randn(&[32, 100, 512], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = attention.forward(&input);
    });
}

#[bench]
fn bench_transformer_encoder(b: &mut Bencher) {
    let encoder = TransformerEncoderLayer::builder(512, 8).dim_feedforward(2048).build();
    let input = Tensor::randn(&[100, 32, 512], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = encoder.forward(&input);
    });
}

// Optimizer benchmarks
#[bench]
fn bench_adam_step(b: &mut Bencher) {
    let params: Vec<_> = (0..100)
        .map(|_| Tensor::randn(&[128, 128], (Kind::Float, Device::Cpu)).set_requires_grad(true))
        .collect();

    let param_ptrs: Vec<_> = params.iter().map(|p| p as *const Tensor as *mut Tensor).collect();
    let mut optimizer = Adam::with_lr(param_ptrs.into_iter(), 1e-3).unwrap();

    // Set dummy gradients
    for param in &params {
        let grad = Tensor::randn(&[128, 128], (Kind::Float, Device::Cpu));
        param.set_grad(&grad);
    }

    b.iter(|| {
        let _ = optimizer.step();
    });
}

#[bench]
fn bench_sgd_step(b: &mut Bencher) {
    let params: Vec<_> = (0..100)
        .map(|_| Tensor::randn(&[128, 128], (Kind::Float, Device::Cpu)).set_requires_grad(true))
        .collect();

    let param_ptrs: Vec<_> = params.iter().map(|p| p as *const Tensor as *mut Tensor).collect();
    let mut optimizer = SGD::builder(param_ptrs.into_iter()).lr(0.1).momentum(0.9).build().unwrap();

    // Set dummy gradients
    for param in &params {
        let grad = Tensor::randn(&[128, 128], (Kind::Float, Device::Cpu));
        param.set_grad(&grad);
    }

    b.iter(|| {
        let _ = optimizer.step();
    });
}

// Sequential model benchmark
#[bench]
fn bench_sequential_forward(b: &mut Bencher) {
    let model = Sequential::new()
        .add(Linear::new(784, 512))
        .add_fn(|x| x.relu())
        .add(Dropout::new(0.5))
        .add(Linear::new(512, 256))
        .add_fn(|x| x.relu())
        .add(Linear::new(256, 10));

    let input = Tensor::randn(&[64, 784], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = model.forward(&input);
    });
}

// Backward pass benchmark
#[bench]
fn bench_backward_pass(b: &mut Bencher) {
    let model =
        Sequential::new().add(Linear::new(784, 256)).add_fn(|x| x.relu()).add(Linear::new(256, 10));

    let input = Tensor::randn(&[32, 784], (Kind::Float, Device::Cpu)).set_requires_grad(true);
    let target = Tensor::randint(10, &[32], (Kind::Int64, Device::Cpu));

    b.iter(|| {
        let output = model.forward(&input);
        let loss = output.cross_entropy_loss(&target);
        loss.backward();
    });
}

// Memory allocation benchmark
#[bench]
fn bench_tensor_allocation_small(b: &mut Bencher) {
    b.iter(|| {
        let _ = Tensor::zeros(&[128, 128], (Kind::Float, Device::Cpu));
    });
}

#[bench]
fn bench_tensor_allocation_large(b: &mut Bencher) {
    b.iter(|| {
        let _ = Tensor::zeros(&[2048, 2048], (Kind::Float, Device::Cpu));
    });
}

// Batch normalization benchmark
#[bench]
fn bench_batchnorm2d_forward(b: &mut Bencher) {
    let bn = BatchNorm2d::new(64);
    let input = Tensor::randn(&[32, 64, 56, 56], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = bn.forward_t(&input, true);
    });
}

// GPU benchmarks (if available)
#[cfg(feature = "cuda")]
mod gpu_benchmarks {
    use super::*;

    #[bench]
    fn bench_gpu_matmul(b: &mut Bencher) {
        let device = Device::Cuda(0);
        let a = Tensor::randn(&[1024, 2048], (Kind::Float, device));
        let b = Tensor::randn(&[2048, 4096], (Kind::Float, device));

        b.iter(|| {
            let _ = a.matmul(&b);
            tch::Cuda::synchronize(0);
        });
    }

    #[bench]
    fn bench_gpu_conv2d(b: &mut Bencher) {
        let device = Device::Cuda(0);
        let input = Tensor::randn(&[32, 3, 224, 224], (Kind::Float, device));
        let weight = Tensor::randn(&[64, 3, 7, 7], (Kind::Float, device));

        b.iter(|| {
            let _ = input.conv2d(&weight, None, &[2, 2], &[3, 3], &[1, 1], 1);
            tch::Cuda::synchronize(0);
        });
    }

    #[bench]
    fn bench_gpu_transformer(b: &mut Bencher) {
        let device = Device::Cuda(0);
        let mut encoder = TransformerEncoderLayer::builder(512, 8).dim_feedforward(2048).build();
        encoder.to_device(device).unwrap();

        let input = Tensor::randn(&[100, 32, 512], (Kind::Float, device));

        b.iter(|| {
            let _ = encoder.forward(&input);
            tch::Cuda::synchronize(0);
        });
    }
}

// Comparison with baseline
#[bench]
fn bench_torch_rs_vs_baseline_linear(b: &mut Bencher) {
    // torch-rs version
    let torch_rs_linear = Linear::new(1024, 2048);
    let input = Tensor::randn(&[128, 1024], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = torch_rs_linear.forward(&input);
    });
}

#[bench]
fn bench_baseline_linear(b: &mut Bencher) {
    // Standard tch version
    let vs = VarStore::new(Device::Cpu);
    let linear = nn::linear(&vs.root(), 1024, 2048, Default::default());
    let input = Tensor::randn(&[128, 1024], (Kind::Float, Device::Cpu));

    b.iter(|| {
        let _ = input.apply(&linear);
    });
}
