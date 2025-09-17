//! Sequential container for chaining modules with automatic parameter discovery
//!
//! Provides PyTorch-compatible Sequential container with dynamic module composition

use crate::nn::{
    phoenix::{PhoenixModule, PhoenixModuleError},
    Module,
};
use crate::{Device, Tensor};
use std::collections::HashMap;

// Wrapper trait that is object-safe
trait ModuleWrapper: std::fmt::Debug {
    fn forward(&self, xs: &Tensor) -> Tensor;
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn named_parameters(&self) -> HashMap<String, &Tensor>;
    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor>;
    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError>;
    fn set_training(&mut self, training: bool);
    fn is_training(&self) -> bool;
    fn zero_grad(&mut self);
    fn device(&self) -> Option<Device>;
    fn state_dict(&self) -> HashMap<String, Tensor>;
    fn load_state_dict(
        &mut self,
        state_dict: &HashMap<String, Tensor>,
    ) -> Result<(), PhoenixModuleError>;
}

// Implementation of wrapper for any PhoenixModule
struct PhoenixModuleWrapper<M: PhoenixModule> {
    module: M,
}

impl<M: PhoenixModule + std::fmt::Debug> ModuleWrapper for PhoenixModuleWrapper<M> {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.module.forward(xs)
    }

    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        self.module.forward(xs)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.module.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.module.parameters_mut()
    }

    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        self.module.named_parameters()
    }

    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        self.module.named_parameters_mut()
    }

    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
        self.module.to_device(device)
    }

    fn set_training(&mut self, training: bool) {
        self.module.set_training(training)
    }

    fn is_training(&self) -> bool {
        self.module.is_training()
    }

    fn zero_grad(&mut self) {
        self.module.zero_grad()
    }

    fn device(&self) -> Option<Device> {
        self.module.device()
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        self.module.state_dict()
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor>,
    ) -> Result<(), PhoenixModuleError> {
        self.module.load_state_dict(state_dict)
    }
}

impl<M: PhoenixModule> std::fmt::Debug for PhoenixModuleWrapper<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PhoenixModuleWrapper")
    }
}

#[derive(Debug)]
pub struct Sequential {
    modules: Vec<Box<dyn ModuleWrapper>>,
    training: bool,
}

impl Sequential {
    pub fn new() -> Self {
        Self { modules: Vec::new(), training: true }
    }

    pub fn add<M: PhoenixModule + std::fmt::Debug + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(PhoenixModuleWrapper { module }));
        self
    }

    pub fn push<M: PhoenixModule + std::fmt::Debug + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(PhoenixModuleWrapper { module }));
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let mut output = xs.shallow_clone();
        for module in &self.modules {
            output = module.forward(&output);
        }
        output
    }
}

impl PhoenixModule for Sequential {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for module in &self.modules {
            params.extend(module.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for module in &mut self.modules {
            params.extend(module.parameters_mut());
        }
        params
    }

    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut named_params = HashMap::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.named_parameters() {
                named_params.insert(format!("{}.{}", i, name), param);
            }
        }
        named_params
    }

    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        let mut named_params = HashMap::new();
        for (i, module) in self.modules.iter_mut().enumerate() {
            for (name, param) in module.named_parameters_mut() {
                named_params.insert(format!("{}.{}", i, name), param);
            }
        }
        named_params
    }

    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
        for module in &mut self.modules {
            module.to_device(device)?;
        }
        Ok(())
    }

    fn device(&self) -> Option<Device> {
        self.modules.first().and_then(|m| m.device())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for module in &mut self.modules {
            module.set_training(training);
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state_dict = HashMap::new();
        for (i, module) in self.modules.iter().enumerate() {
            let module_state = module.state_dict();
            for (key, tensor) in module_state {
                state_dict.insert(format!("{}.{}", i, key), tensor);
            }
        }
        state_dict
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor>,
    ) -> Result<(), PhoenixModuleError> {
        for (i, module) in self.modules.iter_mut().enumerate() {
            let mut module_state = HashMap::new();
            let prefix = format!("{}.", i);

            for (key, tensor) in state_dict.iter() {
                if let Some(stripped_key) = key.strip_prefix(&prefix) {
                    module_state.insert(stripped_key.to_string(), tensor.shallow_clone());
                }
            }

            if !module_state.is_empty() {
                module.load_state_dict(&module_state)?;
            }
        }
        Ok(())
    }
}

#[macro_export]
macro_rules! sequential {
    ($($module:expr),* $(,)?) => {
        {
            let mut seq = Sequential::new();
            $(
                seq.push($module);
            )*
            seq
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::phoenix::{Dropout, Linear};
    use crate::{Device, Kind};

    #[test]
    fn test_sequential_creation() {
        let seq = Sequential::new();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_sequential_add() {
        let seq =
            Sequential::new().add(Linear::new(10, 5)).add(Dropout::new(0.5)).add(Linear::new(5, 1));

        assert_eq!(seq.len(), 3);
        assert!(!seq.is_empty());
    }

    #[test]
    fn test_sequential_push() {
        let mut seq = Sequential::new();
        seq.push(Linear::new(10, 5));
        seq.push(Dropout::new(0.5));

        assert_eq!(seq.len(), 2);
    }

    #[test]
    fn test_sequential_macro() {
        let seq = sequential![Linear::new(10, 5), Dropout::new(0.5), Linear::new(5, 1),];

        assert_eq!(seq.len(), 3);
    }

    #[test]
    fn test_sequential_forward() {
        let seq = Sequential::new().add(Linear::new(4, 3)).add(Linear::new(3, 2));

        let input = Tensor::randn(&[2, 4], (Kind::Float, Device::Cpu));
        let output = seq.forward(&input);

        assert_eq!(output.size(), &[2, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let seq = Sequential::new().add(Linear::new(4, 3)).add(Linear::new(3, 2));

        let params = seq.parameters();
        assert_eq!(params.len(), 4);

        let num_params = seq.num_parameters();
        let expected = (4 * 3 + 3) + (3 * 2 + 2);
        assert_eq!(num_params, expected);
    }

    #[test]
    fn test_sequential_named_parameters() {
        let seq = Sequential::new().add(Linear::new(4, 3)).add(Linear::new(3, 2));

        let named_params = seq.named_parameters();
        assert_eq!(named_params.len(), 4);

        let expected_names = ["0.weight", "0.bias", "1.weight", "1.bias"];
        let actual_names: Vec<&str> = named_params.iter().map(|(name, _)| name.as_str()).collect();

        for expected_name in &expected_names {
            assert!(actual_names.contains(expected_name));
        }
    }

    #[test]
    fn test_sequential_device_management() {
        let mut seq = Sequential::new().add(Linear::new(4, 3)).add(Linear::new(3, 2));

        assert!(seq.to_device(Device::Cpu).is_ok());
        assert_eq!(seq.device(), Some(Device::Cpu));
    }

    #[test]
    fn test_sequential_training_mode() {
        let mut seq = Sequential::new().add(Linear::new(4, 3)).add(Dropout::new(0.5));

        assert!(seq.is_training());

        seq.eval();
        assert!(!seq.is_training());

        seq.train();
        assert!(seq.is_training());
    }

    #[test]
    fn test_sequential_state_dict() {
        let seq = Sequential::new().add(Linear::new(2, 2)).add(Linear::new(2, 1));

        let state_dict = seq.state_dict();
        assert_eq!(state_dict.len(), 4);

        let expected_keys = ["0.weight", "0.bias", "1.weight", "1.bias"];
        for key in &expected_keys {
            assert!(state_dict.contains_key(*key));
        }
    }

    #[test]
    fn test_sequential_load_state_dict() {
        let mut seq1 = Sequential::new().add(Linear::new(2, 2)).add(Linear::new(2, 1));

        let mut seq2 = Sequential::new().add(Linear::new(2, 2)).add(Linear::new(2, 1));

        let state_dict = seq1.state_dict();
        assert!(seq2.load_state_dict(&state_dict).is_ok());
    }

    #[test]
    fn test_sequential_get() {
        let seq = Sequential::new().add(Linear::new(4, 3)).add(Dropout::new(0.5));

        assert!(seq.get(0).is_some());
        assert!(seq.get(1).is_some());
        assert!(seq.get(2).is_none());
    }

    #[test]
    fn test_sequential_get_mut() {
        let mut seq = Sequential::new().add(Linear::new(4, 3)).add(Dropout::new(0.5));

        assert!(seq.get_mut(0).is_some());
        assert!(seq.get_mut(1).is_some());
        assert!(seq.get_mut(2).is_none());
    }

    #[test]
    fn test_sequential_empty_forward() {
        let seq = Sequential::new();
        let input = Tensor::randn(&[2, 4], (Kind::Float, Device::Cpu));
        let output = seq.forward(&input);

        let diff = (&output - &input).abs().sum(Kind::Float);
        assert!(f64::try_from(diff).unwrap() < 1e-6);
    }

    #[test]
    fn test_sequential_buffers() {
        use crate::nn::phoenix::BatchNorm2d;

        let seq = Sequential::new().add(BatchNorm2d::new(3));

        let buffers = seq.buffers();
        assert_eq!(buffers.len(), 2);

        let named_buffers = seq.named_buffers();
        assert_eq!(named_buffers.len(), 2);

        let expected_buffer_names = ["0.running_mean", "0.running_var"];
        let actual_buffer_names: Vec<&str> =
            named_buffers.iter().map(|(name, _)| name.as_str()).collect();

        for expected_name in &expected_buffer_names {
            assert!(actual_buffer_names.contains(expected_name));
        }
    }
}
