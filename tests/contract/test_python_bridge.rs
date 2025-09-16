//! Contract test for Python tensor conversion
#![cfg(all(feature = "torch-rs", feature = "python-compat"))]

use tch::python::{PythonBridge, TensorConversion};
use tch::{Device, Kind, Tensor};

#[test]
fn test_python_bridge_initialization() {
    let bridge = PythonBridge::new();
    if bridge.is_err() {
        // Python might not be available in CI
        eprintln!("Skipping Python bridge test: Python runtime not available");
        return;
    }

    let bridge = bridge.unwrap();
    assert!(bridge.is_available());
    assert!(bridge.python_version().starts_with("3."));
}

#[test]
fn test_tensor_to_python_conversion() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    let rust_tensor = Tensor::randn(&[2, 3, 4], (Kind::Float, Device::Cpu));
    let py_tensor = bridge.to_python(&rust_tensor);

    assert!(py_tensor.is_ok());
    let py_tensor = py_tensor.unwrap();

    // Verify metadata matches
    assert_eq!(py_tensor.shape(), vec![2, 3, 4]);
    assert_eq!(py_tensor.dtype(), "float32");
    assert_eq!(py_tensor.device(), "cpu");
}

#[test]
fn test_tensor_from_python_conversion() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    // Create a Python tensor first
    let rust_tensor = Tensor::ones(&[3, 3], (Kind::Float, Device::Cpu));
    let py_tensor = bridge.to_python(&rust_tensor).unwrap();

    // Convert back to Rust
    let converted = bridge.from_python(py_tensor);
    assert!(converted.is_ok());

    let converted_tensor = converted.unwrap();
    assert_eq!(converted_tensor.size(), vec![3, 3]);

    // Check values match
    let diff = (&rust_tensor - &converted_tensor).abs().sum(Kind::Float);
    assert!(f64::try_from(diff).unwrap() < 1e-6);
}

#[test]
fn test_zero_copy_conversion() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    // Large tensor to verify zero-copy (no memory spike)
    let large_tensor = Tensor::randn(&[1000, 1000], (Kind::Float, Device::Cpu));
    let data_ptr = large_tensor.data_ptr();

    let py_tensor = bridge.to_python(&large_tensor).unwrap();
    assert!(py_tensor.is_zero_copy());

    // Data pointer should be the same (zero-copy)
    assert_eq!(py_tensor.data_ptr(), data_ptr);
}

#[test]
fn test_gradient_preservation() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    let mut tensor = Tensor::randn(&[2, 2], (Kind::Float, Device::Cpu));
    tensor = tensor.set_requires_grad(true);

    let py_tensor = bridge.to_python(&tensor).unwrap();
    assert!(py_tensor.requires_grad());

    // Convert back
    let converted = bridge.from_python(py_tensor).unwrap();
    assert!(converted.requires_grad());
}

#[test]
fn test_device_compatibility() {
    let bridge = match PythonBridge::new() {
        Ok(b) => b,
        Err(_) => {
            eprintln!("Skipping: Python runtime not available");
            return;
        }
    };

    // CPU tensor
    let cpu_tensor = Tensor::randn(&[2, 2], (Kind::Float, Device::Cpu));
    let py_cpu = bridge.to_python(&cpu_tensor).unwrap();
    assert_eq!(py_cpu.device(), "cpu");

    // CUDA tensor (if available)
    if tch::utils::has_cuda() {
        let cuda_tensor = Tensor::randn(&[2, 2], (Kind::Float, Device::Cuda(0)));
        let py_cuda = bridge.to_python(&cuda_tensor).unwrap();
        assert!(py_cuda.device().starts_with("cuda"));
    }
}