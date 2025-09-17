//! Dropout layers with automatic parameter discovery and device management
//!
//! Provides PyTorch-compatible Dropout layers with training mode support

use crate::nn::{Module, phoenix::{PhoenixModule, PhoenixModuleError}};
use crate::{Device, Kind, Tensor};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Dropout {
    pub p: f64,
    training: bool,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "Dropout probability must be between 0 and 1");
        Self {
            p,
            training: true,
        }
    }

    pub fn with_p(p: f64) -> Self {
        Self::new(p)
    }

    pub fn dropout_p(&self) -> f64 {
        self.p
    }

    pub fn set_p(&mut self, p: f64) {
        assert!((0.0..1.0).contains(&p), "Dropout probability must be between 0 and 1");
        self.p = p;
    }
}

impl Module for Dropout {
    fn forward(&self, xs: &Tensor) -> Tensor {
        if self.training && self.p > 0.0 {
            xs.dropout(self.p, self.training)
        } else {
            xs.shallow_clone()
        }
    }
}

impl PhoenixModule for Dropout {
    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn buffers(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        HashMap::new()
    }

    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        HashMap::new()
    }

    fn named_buffers(&self) -> HashMap<String, &Tensor> {
        HashMap::new()
    }

    fn named_buffers_mut(&mut self) -> HashMap<String, &mut Tensor> {
        HashMap::new()
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn to_device(&mut self, _device: Device) -> Result<(), PhoenixModuleError> {
        Ok(())
    }

    fn device(&self) -> Option<Device> {
        None
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::new()
    }

    fn load_state_dict(&mut self, _state_dict: HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Dropout2d {
    pub p: f64,
    training: bool,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        assert!((0.0..1.0).contains(&p), "Dropout probability must be between 0 and 1");
        Self {
            p,
            training: true,
        }
    }

    pub fn with_p(p: f64) -> Self {
        Self::new(p)
    }

    pub fn dropout_p(&self) -> f64 {
        self.p
    }

    pub fn set_p(&mut self, p: f64) {
        assert!((0.0..1.0).contains(&p), "Dropout probability must be between 0 and 1");
        self.p = p;
    }
}

impl Module for Dropout2d {
    fn forward(&self, xs: &Tensor) -> Tensor {
        if self.training && self.p > 0.0 {
            let shape = xs.size();
            if shape.len() == 4 {
                let dropout_shape = &[shape[0], shape[1], 1, 1];
                let mask = Tensor::rand(dropout_shape, (xs.kind(), xs.device()))
                    .ge_tensor_(&Tensor::from(self.p))
                    .to_kind(Kind::Float)
                    .expand(&shape, false);
                xs * mask / (1.0 - self.p)
            } else {
                xs.shallow_clone()
            }
        } else {
            xs.shallow_clone()
        }
    }
}

impl PhoenixModule for Dropout2d {
    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn buffers(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        HashMap::new()
    }

    fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
        HashMap::new()
    }

    fn named_buffers(&self) -> HashMap<String, &Tensor> {
        HashMap::new()
    }

    fn named_buffers_mut(&mut self) -> HashMap<String, &mut Tensor> {
        HashMap::new()
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn to_device(&mut self, _device: Device) -> Result<(), PhoenixModuleError> {
        Ok(())
    }

    fn device(&self) -> Option<Device> {
        None
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        HashMap::new()
    }

    fn load_state_dict(&mut self, _state_dict: HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Kind, Device};

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::new(0.5);
        assert_eq!(dropout.dropout_p(), 0.5);
        assert!(dropout.is_training());
    }

    #[test]
    fn test_dropout_builder() {
        let dropout = Dropout::with_p(0.3);
        assert_eq!(dropout.dropout_p(), 0.3);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be between 0 and 1")]
    fn test_dropout_invalid_p() {
        Dropout::new(1.5);
    }

    #[test]
    fn test_dropout_forward() {
        let mut dropout = Dropout::new(0.5);
        let input = Tensor::ones(&[2, 3], (Kind::Float, Device::Cpu));

        dropout.set_training(true);
        let output_train = dropout.forward(&input);
        assert_eq!(output_train.size(), input.size());

        dropout.set_training(false);
        let output_eval = dropout.forward(&input);
        assert_eq!(output_eval.size(), input.size());
        let diff = (&output_eval - &input).abs().sum(Kind::Float);
        assert!(f64::try_from(diff).unwrap() < 1e-6);
    }

    #[test]
    fn test_dropout_zero_p() {
        let dropout = Dropout::new(0.0);
        let input = Tensor::ones(&[2, 3], (Kind::Float, Device::Cpu));
        let output = dropout.forward(&input);
        let diff = (&output - &input).abs().sum(Kind::Float);
        assert!(f64::try_from(diff).unwrap() < 1e-6);
    }

    #[test]
    fn test_dropout_phoenix_module() {
        let mut dropout = Dropout::new(0.5);

        assert_eq!(dropout.parameters().len(), 0);
        assert_eq!(dropout.buffers().len(), 0);
        assert_eq!(dropout.num_parameters(), 0);
        assert!(dropout.device().is_none());
        assert!(dropout.to_device(Device::Cpu).is_ok());

        dropout.eval();
        assert!(!dropout.is_training());
        dropout.train();
        assert!(dropout.is_training());

        let state_dict = dropout.state_dict();
        assert!(state_dict.is_empty());
        assert!(dropout.load_state_dict(&state_dict).is_ok());
    }

    #[test]
    fn test_dropout2d_creation() {
        let dropout2d = Dropout2d::new(0.25);
        assert_eq!(dropout2d.dropout_p(), 0.25);
        assert!(dropout2d.is_training());
    }

    #[test]
    fn test_dropout2d_forward() {
        let mut dropout2d = Dropout2d::new(0.5);
        let input = Tensor::ones(&[2, 3, 4, 4], (Kind::Float, Device::Cpu));

        dropout2d.set_training(true);
        let output_train = dropout2d.forward(&input);
        assert_eq!(output_train.size(), input.size());

        dropout2d.set_training(false);
        let output_eval = dropout2d.forward(&input);
        assert_eq!(output_eval.size(), input.size());
        let diff = (&output_eval - &input).abs().sum(Kind::Float);
        assert!(f64::try_from(diff).unwrap() < 1e-6);
    }

    #[test]
    fn test_dropout2d_non_4d_input() {
        let dropout2d = Dropout2d::new(0.5);
        let input = Tensor::ones(&[2, 3], (Kind::Float, Device::Cpu));
        let output = dropout2d.forward(&input);
        let diff = (&output - &input).abs().sum(Kind::Float);
        assert!(f64::try_from(diff).unwrap() < 1e-6);
    }

    #[test]
    fn test_dropout2d_phoenix_module() {
        let mut dropout2d = Dropout2d::new(0.1);

        assert_eq!(dropout2d.parameters().len(), 0);
        assert_eq!(dropout2d.buffers().len(), 0);
        assert_eq!(dropout2d.num_parameters(), 0);
        assert!(dropout2d.device().is_none());
        assert!(dropout2d.to_device(Device::Cpu).is_ok());

        dropout2d.eval();
        assert!(!dropout2d.is_training());
        dropout2d.train();
        assert!(dropout2d.is_training());

        let state_dict = dropout2d.state_dict();
        assert!(state_dict.is_empty());
        assert!(dropout2d.load_state_dict(&state_dict).is_ok());
    }

    #[test]
    fn test_dropout_p_modification() {
        let mut dropout = Dropout::new(0.5);
        assert_eq!(dropout.dropout_p(), 0.5);

        dropout.set_p(0.3);
        assert_eq!(dropout.dropout_p(), 0.3);
    }

    #[test]
    fn test_dropout2d_p_modification() {
        let mut dropout2d = Dropout2d::new(0.2);
        assert_eq!(dropout2d.dropout_p(), 0.2);

        dropout2d.set_p(0.8);
        assert_eq!(dropout2d.dropout_p(), 0.8);
    }
}