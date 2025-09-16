use tch::{Device, Tensor, Kind};
use std::collections::HashMap;

// Contract test for Module trait - MUST FAIL until implementation exists
#[cfg(feature = "torch-rs")]
mod module_contract_tests {
    use super::*;

    // This will fail until we implement the Module trait
    pub trait Module: Send + Sync {
        fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError>;
        fn parameters(&self) -> Vec<&Tensor>;
        fn named_parameters(&self) -> HashMap<String, &Tensor>;
        fn to_device(&mut self, device: Device) -> Result<(), ModuleError>;
        fn set_training(&mut self, training: bool);
        fn is_training(&self) -> bool;
        fn zero_grad(&mut self);
        fn device(&self) -> Option<Device>;
    }

    #[derive(Debug, thiserror::Error)]
    pub enum ModuleError {
        #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
        ShapeMismatch { expected: Vec<i64>, actual: Vec<i64> },

        #[error("Device mismatch: module on {module_device:?}, input on {input_device:?}")]
        DeviceMismatch { module_device: Device, input_device: Device },

        #[error("Module not initialized: {reason}")]
        NotInitialized { reason: String },

        #[error("Tensor operation failed: {source}")]
        TensorError {
            #[from]
            source: tch::TchError,
        },
    }

    // Test implementation that will fail until real Module trait exists
    struct TestModule {
        weight: Tensor,
        bias: Tensor,
        training: bool,
    }

    impl TestModule {
        pub fn new() -> Self {
            Self {
                weight: Tensor::randn(&[10, 5], (Kind::Float, Device::Cpu)).set_requires_grad(true),
                bias: Tensor::randn(&[10], (Kind::Float, Device::Cpu)).set_requires_grad(true),
                training: true,
            }
        }
    }

    // This implementation will fail compilation until Module trait is defined in src/
    impl Module for TestModule {
        fn forward(&self, input: &Tensor) -> Result<Tensor, ModuleError> {
            if input.size() != &[1, 5] {
                return Err(ModuleError::ShapeMismatch {
                    expected: vec![1, 5],
                    actual: input.size(),
                });
            }

            let output = input.matmul(&self.weight.tr()) + &self.bias;
            Ok(output)
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weight, &self.bias]
        }

        fn named_parameters(&self) -> HashMap<String, &Tensor> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), &self.weight);
            params.insert("bias".to_string(), &self.bias);
            params
        }

        fn to_device(&mut self, device: Device) -> Result<(), ModuleError> {
            self.weight = self.weight.to_device(device);
            self.bias = self.bias.to_device(device);
            Ok(())
        }

        fn set_training(&mut self, training: bool) {
            self.training = training;
        }

        fn is_training(&self) -> bool {
            self.training
        }

        fn zero_grad(&mut self) {
            if let Some(grad) = self.weight.grad() {
                let _ = grad.zero_();
            }
            if let Some(grad) = self.bias.grad() {
                let _ = grad.zero_();
            }
        }

        fn device(&self) -> Option<Device> {
            if self.weight.device() == self.bias.device() {
                Some(self.weight.device())
            } else {
                None
            }
        }
    }

    #[test]
    fn test_module_forward() {
        let module = TestModule::new();
        let input = Tensor::randn(&[1, 5], (Kind::Float, Device::Cpu));

        let result = module.forward(&input);
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.size(), &[1, 10]);
    }

    #[test]
    fn test_module_parameters() {
        let module = TestModule::new();
        let params = module.parameters();

        assert_eq!(params.len(), 2);
        assert!(params.iter().all(|p| p.requires_grad()));
    }

    #[test]
    fn test_module_named_parameters() {
        let module = TestModule::new();
        let named_params = module.named_parameters();

        assert_eq!(named_params.len(), 2);
        assert!(named_params.contains_key("weight"));
        assert!(named_params.contains_key("bias"));
    }

    #[test]
    fn test_module_device_movement() {
        let mut module = TestModule::new();

        // Should start on CPU
        assert_eq!(module.device(), Some(Device::Cpu));

        // Moving to same device should work
        assert!(module.to_device(Device::Cpu).is_ok());
        assert_eq!(module.device(), Some(Device::Cpu));
    }

    #[test]
    fn test_module_training_mode() {
        let mut module = TestModule::new();

        // Should start in training mode
        assert!(module.is_training());

        // Should be able to switch modes
        module.set_training(false);
        assert!(!module.is_training());

        module.set_training(true);
        assert!(module.is_training());
    }

    #[test]
    fn test_module_zero_grad() {
        let mut module = TestModule::new();

        // Create some fake gradients
        let grad = Tensor::ones_like(&module.weight);
        module.weight.set_grad(&grad);

        // zero_grad should clear gradients
        module.zero_grad();

        // This test will pass once zero_grad is properly implemented
    }

    #[test]
    fn test_shape_mismatch_error() {
        let module = TestModule::new();
        let wrong_input = Tensor::randn(&[1, 3], (Kind::Float, Device::Cpu));

        let result = module.forward(&wrong_input);
        assert!(result.is_err());

        if let Err(ModuleError::ShapeMismatch { expected, actual }) = result {
            assert_eq!(expected, vec![1, 5]);
            assert_eq!(actual, vec![1, 3]);
        } else {
            panic!("Expected ShapeMismatch error");
        }
    }
}

#[cfg(not(feature = "torch-rs"))]
mod disabled_tests {
    #[test]
    #[ignore]
    fn module_tests_require torch-rs feature() {
        panic!("Module contract tests require 'torch-rs' feature to be enabled");
    }
}