//! Python bindings for torch-rs using PyO3
//!
//! Provides seamless integration between Rust's torch-rs and Python

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::wrap_pyfunction;
use numpy::{PyArray, PyArrayDyn, PyReadonlyArrayDyn};
use ndarray::{ArrayD, IxDyn};
use tch::{Device, Kind, Tensor};
use std::collections::HashMap;

/// Convert a numpy array to a Tensor
#[pyfunction]
fn numpy_to_tensor(py: Python, array: &PyArrayDyn<f32>) -> PyResult<PyTensor> {
    let array = array.readonly();
    let shape: Vec<i64> = array.shape().iter().map(|&x| x as i64).collect();
    let data = array.as_slice()?.to_vec();
    
    let tensor = Tensor::of_slice(&data).reshape(&shape);
    Ok(PyTensor { inner: tensor })
}

/// Convert a Tensor to a numpy array
#[pyfunction]
fn tensor_to_numpy(py: Python, tensor: &PyTensor) -> PyResult<Py<PyArrayDyn<f32>>> {
    let shape: Vec<usize> = tensor.inner.size().iter().map(|&x| x as usize).collect();
    let data = Vec::<f32>::from(&tensor.inner);
    
    let array = PyArrayDyn::from_vec(py, data)?;
    let array = array.reshape(shape)?;
    Ok(array.to_owned())
}

/// Python wrapper for Tensor
#[pyclass]
#[derive(Clone)]
struct PyTensor {
    inner: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<i64>) -> Self {
        let tensor = Tensor::of_slice(&data).reshape(&shape);
        Self { inner: tensor }
    }
    
    /// Get tensor shape
    #[getter]
    fn shape(&self) -> Vec<i64> {
        self.inner.size()
    }
    
    /// Get tensor dtype
    #[getter]
    fn dtype(&self) -> String {
        format!("{:?}", self.inner.kind())
    }
    
    /// Get tensor device
    #[getter]
    fn device(&self) -> String {
        format!("{:?}", self.inner.device())
    }
    
    /// Move tensor to device
    fn to(&self, device: &str) -> PyResult<PyTensor> {
        let device = match device {
            "cpu" => Device::Cpu,
            "cuda" | "cuda:0" => Device::Cuda(0),
            _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown device: {}", device)
            )),
        };
        Ok(PyTensor { inner: self.inner.to_device(device) })
    }
    
    /// Compute gradient
    fn backward(&self) -> PyResult<()> {
        self.inner.backward();
        Ok(())
    }
    
    /// Get gradient
    fn grad(&self) -> Option<PyTensor> {
        self.inner.grad().map(|g| PyTensor { inner: g })
    }
    
    /// Add tensors
    fn __add__(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: &self.inner + &other.inner }
    }
    
    /// Subtract tensors
    fn __sub__(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: &self.inner - &other.inner }
    }
    
    /// Multiply tensors
    fn __mul__(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: &self.inner * &other.inner }
    }
    
    /// Matrix multiplication
    fn matmul(&self, other: &PyTensor) -> PyTensor {
        PyTensor { inner: self.inner.matmul(&other.inner) }
    }
    
    /// Apply ReLU activation
    fn relu(&self) -> PyTensor {
        PyTensor { inner: self.inner.relu() }
    }
    
    /// Apply Sigmoid activation
    fn sigmoid(&self) -> PyTensor {
        PyTensor { inner: self.inner.sigmoid() }
    }
    
    /// Apply Softmax
    fn softmax(&self, dim: i64) -> PyTensor {
        PyTensor { inner: self.inner.softmax(dim, Kind::Float) }
    }
    
    /// Get string representation
    fn __repr__(&self) -> String {
        format!("PyTensor(shape={:?}, device={:?})", self.shape(), self.device())
    }
}

/// Python wrapper for Linear layer
#[pyclass]
struct PyLinear {
    weight: Tensor,
    bias: Option<Tensor>,
    in_features: i64,
    out_features: i64,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: i64, out_features: i64, bias: Option<bool>) -> Self {
        let vs = tch::nn::VarStore::new(Device::cuda_if_available());
        let root = vs.root();
        let use_bias = bias.unwrap_or(true);
        
        let linear = tch::nn::linear(&root, in_features, out_features, tch::nn::LinearConfig {
            bias: use_bias,
            ..Default::default()
        });
        
        Self {
            weight: linear.ws.shallow_clone(),
            bias: linear.bs.as_ref().map(|b| b.shallow_clone()),
            in_features,
            out_features,
        }
    }
    
    /// Forward pass
    fn forward(&self, input: &PyTensor) -> PyTensor {
        let output = input.inner.matmul(&self.weight.tr());
        let output = if let Some(ref b) = self.bias {
            output + b
        } else {
            output
        };
        PyTensor { inner: output }
    }
    
    #[getter]
    fn weight(&self) -> PyTensor {
        PyTensor { inner: self.weight.shallow_clone() }
    }
    
    #[getter]
    fn bias(&self) -> Option<PyTensor> {
        self.bias.as_ref().map(|b| PyTensor { inner: b.shallow_clone() })
    }
}

/// Python wrapper for Sequential model
#[pyclass]
struct PySequential {
    layers: Vec<Py<PyAny>>,
}

#[pymethods]
impl PySequential {
    #[new]
    fn new() -> Self {
        Self { layers: Vec::new() }
    }
    
    /// Add a layer to the sequential model
    fn add(&mut self, layer: Py<PyAny>) {
        self.layers.push(layer);
    }
    
    /// Forward pass through all layers
    fn forward(&self, py: Python, input: &PyTensor) -> PyResult<PyTensor> {
        let mut x = input.clone();
        
        for layer in &self.layers {
            let result = layer.call_method1(py, "forward", (&x,))?;
            x = result.extract::<PyTensor>(py)?;
        }
        
        Ok(x)
    }
}

/// Python wrapper for Adam optimizer
#[pyclass]
struct PyAdam {
    parameters: Vec<Tensor>,
    lr: f64,
    betas: (f64, f64),
    eps: f64,
    state: HashMap<usize, (Tensor, Tensor, i64)>, // (exp_avg, exp_avg_sq, step)
}

#[pymethods]
impl PyAdam {
    #[new]
    fn new(parameters: Vec<PyTensor>, lr: Option<f64>, betas: Option<(f64, f64)>, eps: Option<f64>) -> Self {
        Self {
            parameters: parameters.into_iter().map(|p| p.inner).collect(),
            lr: lr.unwrap_or(1e-3),
            betas: betas.unwrap_or((0.9, 0.999)),
            eps: eps.unwrap_or(1e-8),
            state: HashMap::new(),
        }
    }
    
    /// Perform optimization step
    fn step(&mut self) -> PyResult<()> {
        for (idx, param) in self.parameters.iter().enumerate() {
            if let Some(grad) = param.grad() {
                let state = self.state.entry(idx).or_insert_with(|| {
                    (
                        Tensor::zeros_like(param),
                        Tensor::zeros_like(param),
                        0,
                    )
                });
                
                state.2 += 1; // Increment step
                let step = state.2;
                
                // Update biased first moment estimate
                state.0 = &state.0 * self.betas.0 + &grad * (1.0 - self.betas.0);
                
                // Update biased second raw moment estimate
                state.1 = &state.1 * self.betas.1 + (&grad * &grad) * (1.0 - self.betas.1);
                
                // Compute bias-corrected first moment estimate
                let bias_correction1 = 1.0 - self.betas.0.powi(step as i32);
                let bias_correction2 = 1.0 - self.betas.1.powi(step as i32);
                
                // Update parameters
                let step_size = self.lr * bias_correction2.sqrt() / bias_correction1;
                let _ = param.sub_(&(&state.0 / (&state.1.sqrt() + self.eps)) * step_size);
            }
        }
        Ok(())
    }
    
    /// Zero all gradients
    fn zero_grad(&self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
}

/// Python wrapper for DataLoader
#[pyclass]
struct PyDataLoader {
    dataset: Py<PyAny>,
    batch_size: usize,
    shuffle: bool,
    num_samples: usize,
}

#[pymethods]
impl PyDataLoader {
    #[new]
    fn new(dataset: Py<PyAny>, batch_size: Option<usize>, shuffle: Option<bool>) -> PyResult<Self> {
        Python::with_gil(|py| {
            let len = dataset.call_method0(py, "__len__")?;
            let num_samples: usize = len.extract(py)?;
            
            Ok(Self {
                dataset,
                batch_size: batch_size.unwrap_or(32),
                shuffle: shuffle.unwrap_or(true),
                num_samples,
            })
        })
    }
    
    fn __iter__(slf: PyRef<Self>) -> PyResult<Py<PyDataLoaderIterator>> {
        let iter = PyDataLoaderIterator {
            dataloader: slf.into(),
            current_idx: 0,
            indices: (0..slf.num_samples).collect(),
        };
        
        Py::new(slf.py(), iter)
    }
}

#[pyclass]
struct PyDataLoaderIterator {
    dataloader: Py<PyDataLoader>,
    current_idx: usize,
    indices: Vec<usize>,
}

#[pymethods]
impl PyDataLoaderIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }
    
    fn __next__(&mut self, py: Python) -> PyResult<Option<(PyTensor, PyTensor)>> {
        let dataloader = self.dataloader.borrow(py);
        
        if self.current_idx >= self.indices.len() {
            return Ok(None);
        }
        
        let batch_indices: Vec<usize> = self.indices
            .iter()
            .skip(self.current_idx)
            .take(dataloader.batch_size)
            .copied()
            .collect();
        
        if batch_indices.is_empty() {
            return Ok(None);
        }
        
        self.current_idx += batch_indices.len();
        
        // Collect batch data
        let mut batch_data = Vec::new();
        let mut batch_labels = Vec::new();
        
        for idx in batch_indices {
            let item = dataloader.dataset.call_method1(py, "__getitem__", (idx,))?;
            let (data, label): (PyTensor, PyTensor) = item.extract(py)?;
            batch_data.push(data.inner);
            batch_labels.push(label.inner);
        }
        
        // Stack into batch tensors
        let batch_data = Tensor::stack(&batch_data, 0);
        let batch_labels = Tensor::stack(&batch_labels, 0);
        
        Ok(Some((
            PyTensor { inner: batch_data },
            PyTensor { inner: batch_labels },
        )))
    }
}

/// Create various tensor operations
#[pyfunction]
fn zeros(shape: Vec<i64>) -> PyTensor {
    PyTensor { inner: Tensor::zeros(&shape, (Kind::Float, Device::Cpu)) }
}

#[pyfunction]
fn ones(shape: Vec<i64>) -> PyTensor {
    PyTensor { inner: Tensor::ones(&shape, (Kind::Float, Device::Cpu)) }
}

#[pyfunction]
fn randn(shape: Vec<i64>) -> PyTensor {
    PyTensor { inner: Tensor::randn(&shape, (Kind::Float, Device::Cpu)) }
}

#[pyfunction]
fn arange(start: i64, end: i64, step: Option<i64>) -> PyTensor {
    let step = step.unwrap_or(1);
    PyTensor { inner: Tensor::arange_step(start, end, step, (Kind::Int64, Device::Cpu)) }
}

/// Python module initialization
#[pymodule]
fn torch_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    // Add tensor class and functions
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(numpy_to_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(tensor_to_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    
    // Add layers
    m.add_class::<PyLinear>()?;
    m.add_class::<PySequential>()?;
    
    // Add optimizers
    m.add_class::<PyAdam>()?;
    
    // Add data utilities
    m.add_class::<PyDataLoader>()?;
    
    Ok(())
}