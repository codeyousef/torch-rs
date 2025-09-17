//! Test fixtures and data generators

use crate::{nn, Device, Kind, Tensor};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct TestFixture {
    pub id: String,
    pub name: String,
    pub fixture_type: String,
    pub lazy_load: bool,
    pub cache_enabled: bool,
    pub path: Option<PathBuf>,
    pub cleanup_required: bool,
    pub dependencies: Vec<String>,
    loaded: std::cell::Cell<bool>,
}

impl TestFixture {
    pub fn list(_filter: Option<String>) -> Result<Vec<TestFixture>, Box<dyn std::error::Error>> {
        Ok(vec![])
    }

    pub fn is_loaded(&self) -> bool {
        self.loaded.get()
    }

    pub fn load(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        self.loaded.set(true);
        Ok(vec![1.0, 2.0, 3.0])
    }

    pub fn cleanup(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.loaded.set(false);
        Ok(())
    }
}

pub struct FixtureBuilder {
    name: String,
    fixture_type: String,
    lazy_load: bool,
    cache: bool,
    path: Option<PathBuf>,
    cleanup_required: bool,
    dependencies: Vec<String>,
}

impl FixtureBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            fixture_type: "dataset".to_string(),
            lazy_load: false,
            cache: false,
            path: None,
            cleanup_required: false,
            dependencies: vec![],
        }
    }

    pub fn fixture_type(mut self, t: &str) -> Self {
        self.fixture_type = t.to_string();
        self
    }

    pub fn lazy_load(mut self, lazy: bool) -> Self {
        self.lazy_load = lazy;
        self
    }

    pub fn cache(mut self, cache: bool) -> Self {
        self.cache = cache;
        self
    }

    pub fn path(mut self, path: PathBuf) -> Self {
        self.path = Some(path);
        self
    }

    pub fn cleanup_required(mut self, cleanup: bool) -> Self {
        self.cleanup_required = cleanup;
        self
    }

    pub fn depends_on(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }

    pub fn generator<F>(self, _f: F) -> Self
    where
        F: Fn() -> Vec<f32>,
    {
        self
    }

    pub fn generator_with_deps<F>(self, _f: F) -> Self
    where
        F: Fn(&[TestFixture]) -> Vec<f32>,
    {
        self
    }

    pub fn build(self) -> Result<TestFixture, Box<dyn std::error::Error>> {
        Ok(TestFixture {
            id: format!("{}_id", self.name),
            name: self.name,
            fixture_type: self.fixture_type,
            lazy_load: self.lazy_load,
            cache_enabled: self.cache,
            path: self.path,
            cleanup_required: self.cleanup_required,
            dependencies: self.dependencies,
            loaded: std::cell::Cell::new(false),
        })
    }
}

// Sample dataset loaders (stub implementations)
pub fn load_mnist_sample(_size: usize) -> impl Dataset {
    StubDataset
}

pub fn load_cifar10_sample(_size: usize) -> impl Dataset {
    StubDataset
}

pub fn load_imbalanced_dataset() -> impl Dataset {
    StubDataset
}

pub fn create_simple_cnn(vs: &nn::Path) -> impl nn::Module {
    nn::seq()
        .add(nn::conv2d(vs / "conv1", 1, 32, 3, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::conv2d(vs / "conv2", 32, 64, 3, Default::default()))
        .add_fn(|xs| xs.relu().max_pool2d_default(2))
        .add_fn(|xs| xs.flat_view())
        .add(nn::linear(vs / "fc", 1024, 10, Default::default()))
}

pub fn create_simple_mlp(vs: &nn::Path, input_size: i64, output_size: i64) -> impl nn::Module {
    nn::seq()
        .add(nn::linear(vs / "fc1", input_size, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "fc2", 128, output_size, Default::default()))
}

pub fn create_resnet18(vs: &nn::Path, num_classes: i64) -> impl nn::Module {
    // Simplified ResNet18
    nn::seq()
        .add(nn::conv2d(vs / "conv1", 3, 64, 7, Default::default()))
        .add_fn(|xs| xs.relu())
        .add_fn(|xs| xs.flat_view())
        .add(nn::linear(vs / "fc", 64 * 7 * 7, num_classes, Default::default()))
}

// Stub dataset implementation
#[derive(Clone)]
struct StubDataset;

pub trait Dataset: Clone {
    fn train_iter(&self, _batch_size: i64) -> impl Iterator<Item = (Tensor, Tensor)> {
        (0..10).map(|_| {
            (
                Tensor::randn(&[32, 1, 28, 28], (Kind::Float, Device::Cpu)),
                Tensor::randint(10, &[32], (Kind::Int64, Device::Cpu)),
            )
        })
    }

    fn iter(&self) -> impl Iterator<Item = (Tensor, i64)> {
        (0..100).map(|i| (Tensor::randn(&[1, 28, 28], (Kind::Float, Device::Cpu)), (i % 10) as i64))
    }

    fn with_transforms(self, _transforms: Vec<Transform>) -> Self {
        self
    }

    fn split(self, _ratio: f32) -> (Self, Self) {
        (self.clone(), self)
    }

    fn len(&self) -> usize {
        100
    }

    fn get_indices(&self) -> Vec<usize> {
        (0..self.len()).collect()
    }

    fn compute_class_weights(&self) -> Vec<f32> {
        vec![1.0; 10]
    }
}

impl Dataset for StubDataset {}

pub struct Transform;

impl Transform {
    pub fn normalize(_mean: Vec<f64>, _std: Vec<f64>) -> Self {
        Transform
    }

    pub fn random_horizontal_flip(_p: f64) -> Self {
        Transform
    }

    pub fn random_crop(_size: i64, _padding: i64) -> Self {
        Transform
    }

    pub fn center_crop(_size: i64) -> Self {
        Transform
    }

    pub fn random_rotation(_degrees: f64) -> Self {
        Transform
    }

    pub fn random_brightness(_factor: f64) -> Self {
        Transform
    }
}

pub struct WeightedRandomSampler;

impl WeightedRandomSampler {
    pub fn new(_weights: Vec<f32>, _num_samples: usize) -> Self {
        WeightedRandomSampler
    }
}

// Extension trait for Dataset
impl StubDataset {
    pub fn test_images(&self) -> Tensor {
        Tensor::randn(&[100, 1, 28, 28], (Kind::Float, Device::Cpu))
    }

    pub fn test_labels(&self) -> Tensor {
        Tensor::randint(10, &[100], (Kind::Int64, Device::Cpu))
    }
}
