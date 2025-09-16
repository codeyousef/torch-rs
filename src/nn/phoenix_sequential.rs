//! Sequential container for chaining modules with automatic parameter discovery
//!
//! Provides PyTorch-compatible Sequential container with dynamic module composition

use crate::nn::{Module, phoenix::{PhoenixModule, PhoenixModuleError}};
use crate::{Device, Tensor};
use std::collections::HashMap;

#[derive(Debug)]
pub struct Sequential {
    modules: Vec<Box<dyn PhoenixModule>>,
    training: bool,
}

impl Sequential {
    pub fn new() -> Self {
        Self {
            modules: Vec::new(),
            training: true,
        }
    }

    pub fn add<M: PhoenixModule + 'static>(mut self, module: M) -> Self {
        self.modules.push(Box::new(module));
        self
    }

    pub fn push<M: PhoenixModule + 'static>(&mut self, module: M) {
        self.modules.push(Box::new(module));
    }

    pub fn len(&self) -> usize {
        self.modules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.modules.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&dyn PhoenixModule> {
        self.modules.get(index).map(|m| m.as_ref())
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut dyn PhoenixModule> {
        self.modules.get_mut(index).map(|m| m.as_mut())
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

    fn buffers(&self) -> Vec<&Tensor> {
        let mut buffers = Vec::new();
        for module in &self.modules {
            buffers.extend(module.buffers());
        }
        buffers
    }

    fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
        let mut buffers = Vec::new();
        for module in &mut self.modules {
            buffers.extend(module.buffers_mut());
        }
        buffers
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        let mut named_params = Vec::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, param) in module.named_parameters() {
                named_params.push((format!("{}.{}", i, name), param));
            }
        }
        named_params
    }

    fn named_parameters_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut named_params = Vec::new();
        for (i, module) in self.modules.iter_mut().enumerate() {
            for (name, param) in module.named_parameters_mut() {
                named_params.push((format!("{}.{}", i, name), param));
            }
        }
        named_params
    }

    fn named_buffers(&self) -> Vec<(String, &Tensor)> {
        let mut named_buffers = Vec::new();
        for (i, module) in self.modules.iter().enumerate() {
            for (name, buffer) in module.named_buffers() {
                named_buffers.push((format!("{}.{}", i, name), buffer));
            }
        }
        named_buffers
    }

    fn named_buffers_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut named_buffers = Vec::new();
        for (i, module) in self.modules.iter_mut().enumerate() {
            for (name, buffer) in module.named_buffers_mut() {
                named_buffers.push((format!("{}.{}", i, name), buffer));
            }
        }
        named_buffers
    }

    fn num_parameters(&self) -> usize {
        self.modules.iter().map(|m| m.num_parameters()).sum()
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

    fn train(&mut self) {
        self.set_training(true);
    }

    fn eval(&mut self) {
        self.set_training(false);
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

    fn load_state_dict(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
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

pub use sequential;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::phoenix::{Linear, Dropout};
    use crate::{Kind, Device};

    #[test]
    fn test_sequential_creation() {
        let seq = Sequential::new();
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_sequential_add() {
        let seq = Sequential::new()
            .add(Linear::new(10, 5))
            .add(Dropout::new(0.5))
            .add(Linear::new(5, 1));

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
        let seq = sequential![
            Linear::new(10, 5),
            Dropout::new(0.5),
            Linear::new(5, 1),
        ];

        assert_eq!(seq.len(), 3);
    }

    #[test]
    fn test_sequential_forward() {
        let seq = Sequential::new()
            .add(Linear::new(4, 3))
            .add(Linear::new(3, 2));

        let input = Tensor::randn(&[2, 4], (Kind::Float, Device::Cpu));
        let output = seq.forward(&input);

        assert_eq!(output.size(), &[2, 2]);
    }

    #[test]
    fn test_sequential_parameters() {
        let seq = Sequential::new()
            .add(Linear::new(4, 3))
            .add(Linear::new(3, 2));

        let params = seq.parameters();
        assert_eq!(params.len(), 4);

        let num_params = seq.num_parameters();
        let expected = (4 * 3 + 3) + (3 * 2 + 2);
        assert_eq!(num_params, expected);
    }

    #[test]
    fn test_sequential_named_parameters() {
        let seq = Sequential::new()
            .add(Linear::new(4, 3))
            .add(Linear::new(3, 2));

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
        let mut seq = Sequential::new()
            .add(Linear::new(4, 3))
            .add(Linear::new(3, 2));

        assert!(seq.to_device(Device::Cpu).is_ok());
        assert_eq!(seq.device(), Some(Device::Cpu));
    }

    #[test]
    fn test_sequential_training_mode() {
        let mut seq = Sequential::new()
            .add(Linear::new(4, 3))
            .add(Dropout::new(0.5));

        assert!(seq.is_training());

        seq.eval();
        assert!(!seq.is_training());

        seq.train();
        assert!(seq.is_training());
    }

    #[test]
    fn test_sequential_state_dict() {
        let seq = Sequential::new()
            .add(Linear::new(2, 2))
            .add(Linear::new(2, 1));

        let state_dict = seq.state_dict();
        assert_eq!(state_dict.len(), 4);

        let expected_keys = ["0.weight", "0.bias", "1.weight", "1.bias"];
        for key in &expected_keys {
            assert!(state_dict.contains_key(*key));
        }
    }

    #[test]
    fn test_sequential_load_state_dict() {
        let mut seq1 = Sequential::new()
            .add(Linear::new(2, 2))
            .add(Linear::new(2, 1));

        let mut seq2 = Sequential::new()
            .add(Linear::new(2, 2))
            .add(Linear::new(2, 1));

        let state_dict = seq1.state_dict();
        assert!(seq2.load_state_dict(&state_dict).is_ok());
    }

    #[test]
    fn test_sequential_get() {
        let seq = Sequential::new()
            .add(Linear::new(4, 3))
            .add(Dropout::new(0.5));

        assert!(seq.get(0).is_some());
        assert!(seq.get(1).is_some());
        assert!(seq.get(2).is_none());
    }

    #[test]
    fn test_sequential_get_mut() {
        let mut seq = Sequential::new()
            .add(Linear::new(4, 3))
            .add(Dropout::new(0.5));

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

        let seq = Sequential::new()
            .add(BatchNorm2d::new(3));

        let buffers = seq.buffers();
        assert_eq!(buffers.len(), 2);

        let named_buffers = seq.named_buffers();
        assert_eq!(named_buffers.len(), 2);

        let expected_buffer_names = ["0.running_mean", "0.running_var"];
        let actual_buffer_names: Vec<&str> = named_buffers.iter()
            .map(|(name, _)| name.as_str())
            .collect();

        for expected_name in &expected_buffer_names {
            assert!(actual_buffer_names.contains(expected_name));
        }
    }
}