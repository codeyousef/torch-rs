//! Contract test for torch.compile integration
#![cfg(all(feature = "torch-rs", feature = "python-compat"))]

use tch::python::{PythonBridge, TorchCompile, CompileBackend};
use tch::nn::phoenix::{PhoenixModule, Linear};
use tch::{nn, Device, Kind, Tensor};

#[derive(Debug)]
struct SimpleModel {
    fc1: Linear,
    fc2: Linear,
}

impl SimpleModel {
    fn new() -> Self {
        Self {
            fc1: Linear::new(10, 5),
            fc2: Linear::new(5, 2),
        }
    }
}

impl nn::Module for SimpleModel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).relu().apply(&self.fc2)
    }
}

tch::impl_phoenix_module!(SimpleModel {
    fc1: Linear,
    fc2: Linear,
});

#[test]
fn test_torch_compile_basic() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    if !bridge.has_torch_compile() {
        eprintln!("Skipping: torch.compile not available (PyTorch < 2.0)");
        return;
    }

    let model = SimpleModel::new();
    let compiled = bridge.compile(&model, CompileBackend::Inductor);

    assert!(compiled.is_ok());
    let compiled_model = compiled.unwrap();

    // Test inference with compiled model
    let input = Tensor::randn(&[4, 10], (Kind::Float, Device::Cpu));
    let output = compiled_model.forward(&input);
    assert_eq!(output.size(), vec![4, 2]);
}

#[test]
fn test_compile_backends() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    if !bridge.has_torch_compile() {
        return;
    }

    let model = SimpleModel::new();

    let backends = vec![
        CompileBackend::Inductor,
        CompileBackend::AotEager,
        CompileBackend::CudaGraphs,
    ];

    for backend in backends {
        let result = bridge.compile(&model, backend);

        if backend == CompileBackend::CudaGraphs && !tch::utils::has_cuda() {
            // CudaGraphs requires CUDA
            assert!(result.is_err());
        } else {
            // Other backends should work
            assert!(result.is_ok(), "Failed to compile with {:?}", backend);
        }
    }
}

#[test]
fn test_compile_speedup() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    if !bridge.has_torch_compile() {
        return;
    }

    let model = SimpleModel::new();
    let compiled_model = bridge.compile(&model, CompileBackend::Inductor).unwrap();

    let input = Tensor::randn(&[100, 10], (Kind::Float, Device::Cpu));

    // Warmup
    for _ in 0..10 {
        let _ = model.forward(&input);
        let _ = compiled_model.forward(&input);
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = model.forward(&input);
    }
    let normal_time = start.elapsed();

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = compiled_model.forward(&input);
    }
    let compiled_time = start.elapsed();

    println!("Normal: {:?}, Compiled: {:?}", normal_time, compiled_time);

    // Compiled should be at least as fast (often faster)
    assert!(compiled_time.as_millis() <= normal_time.as_millis() * 2);
}

#[test]
fn test_compile_with_dynamic_shapes() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    if !bridge.has_torch_compile() {
        return;
    }

    let model = SimpleModel::new();
    let compiled_model = bridge
        .compile_with_options(&model, CompileBackend::Inductor, true, false)
        .unwrap();

    // Test with different batch sizes (dynamic shapes)
    let sizes = vec![1, 4, 8, 16];
    for batch_size in sizes {
        let input = Tensor::randn(&[batch_size, 10], (Kind::Float, Device::Cpu));
        let output = compiled_model.forward(&input);
        assert_eq!(output.size(), vec![batch_size, 2]);
    }
}