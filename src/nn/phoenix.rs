//! Project Phoenix Module System
//!
//! Extended module traits with automatic parameter discovery, device management,
//! and PyTorch-compatible functionality.

#[cfg(feature = "torch-rs")]
pub mod module {
    use crate::{Device, TchError, Tensor};
    use std::collections::HashMap;

    /// Extended Module trait for Project Phoenix
    ///
    /// This trait extends the basic Module functionality with automatic parameter
    /// discovery, device management, and PyTorch-compatible features.
    pub trait PhoenixModule: crate::nn::Module {
        /// Get all trainable parameters recursively
        fn parameters(&self) -> Vec<&Tensor>;

        /// Get mutable references to all trainable parameters recursively
        fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

        /// Get all parameters with their names
        fn named_parameters(&self) -> HashMap<String, &Tensor>;

        /// Get mutable named parameters
        fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor>;

        /// Move module to specified device
        fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError>;

        /// Set training mode
        fn set_training(&mut self, training: bool);

        /// Check if module is in training mode
        fn is_training(&self) -> bool;

        /// Zero out gradients for all parameters
        fn zero_grad(&mut self);

        /// Get module's current device
        fn device(&self) -> Option<Device>;

        /// Get all non-trainable buffers recursively
        fn buffers(&self) -> Vec<&Tensor> {
            Vec::new() // Default implementation
        }

        /// Get mutable references to all buffers
        fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
            Vec::new() // Default implementation
        }

        /// Get all buffers with their names
        fn named_buffers(&self) -> HashMap<String, &Tensor> {
            HashMap::new() // Default implementation
        }

        /// Get mutable named buffers
        fn named_buffers_mut(&mut self) -> HashMap<String, &mut Tensor> {
            HashMap::new() // Default implementation
        }

        /// Apply a function to all parameters
        fn apply<F>(&mut self, f: F)
        where
            F: Fn(&mut Tensor) + Clone,
        {
            for param in self.parameters_mut() {
                f(param);
            }
        }

        /// Get the total number of parameters
        fn num_parameters(&self) -> usize {
            self.parameters().iter().map(|p| p.numel() as usize).sum()
        }

        /// Get the total number of trainable parameters
        fn num_trainable_parameters(&self) -> usize {
            self.parameters().iter().filter(|p| p.requires_grad()).map(|p| p.numel() as usize).sum()
        }

        /// Get state dictionary (all parameters and buffers as a map)
        fn state_dict(&self) -> HashMap<String, Tensor> {
            let mut state = HashMap::new();

            // Add parameters
            for (name, param) in self.named_parameters() {
                state.insert(name, param.copy());
            }

            // Add buffers
            for (name, buffer) in self.named_buffers() {
                state.insert(name, buffer.copy());
            }

            state
        }

        /// Load state dictionary into module
        fn load_state_dict(
            &mut self,
            state_dict: HashMap<String, Tensor>,
        ) -> Result<(), PhoenixModuleError> {
            let mut named_params = self.named_parameters_mut();
            let mut named_buffers = self.named_buffers_mut();

            for (name, tensor) in state_dict {
                if let Some(param) = named_params.get_mut(&name) {
                    if param.size() != tensor.size() {
                        return Err(PhoenixModuleError::ShapeMismatch {
                            expected: param.size(),
                            actual: tensor.size(),
                            parameter: name,
                        });
                    }
                    param.copy_(&tensor);
                } else if let Some(buffer) = named_buffers.get_mut(&name) {
                    if buffer.size() != tensor.size() {
                        return Err(PhoenixModuleError::ShapeMismatch {
                            expected: buffer.size(),
                            actual: tensor.size(),
                            parameter: name,
                        });
                    }
                    buffer.copy_(&tensor);
                } else {
                    return Err(PhoenixModuleError::UnknownParameter(name));
                }
            }

            Ok(())
        }
    }

    /// Errors that can occur during Phoenix module operations
    #[derive(Debug, thiserror::Error)]
    pub enum PhoenixModuleError {
        #[error(
            "Shape mismatch for parameter '{parameter}': expected {expected:?}, got {actual:?}"
        )]
        ShapeMismatch { expected: Vec<i64>, actual: Vec<i64>, parameter: String },

        #[error("Device mismatch: module on {module_device:?}, input on {input_device:?}")]
        DeviceMismatch { module_device: Device, input_device: Device },

        #[error("Module not initialized: {reason}")]
        NotInitialized { reason: String },

        #[error("Unknown parameter: {0}")]
        UnknownParameter(String),

        #[error("Feature not implemented: {0}")]
        NotImplemented(String),

        #[error("Invalid configuration: {0}")]
        InvalidConfiguration(String),

        #[error("Tensor operation failed: {source}")]
        TensorError {
            #[from]
            source: TchError,
        },
    }

    /// Extension trait for Tensor initialization methods
    pub trait TensorInit {
        /// Initialize using Xavier/Glorot uniform initialization
        fn xavier_uniform_(&mut self) -> &mut Self;

        /// Initialize using Xavier/Glorot normal initialization
        fn xavier_normal_(&mut self) -> &mut Self;

        /// Initialize using Kaiming uniform initialization
        fn kaiming_uniform_(&mut self) -> &mut Self;

        /// Initialize using Kaiming normal initialization
        fn kaiming_normal_(&mut self) -> &mut Self;
    }

    impl TensorInit for Tensor {
        fn xavier_uniform_(&mut self) -> &mut Self {
            let dims = self.size();
            if dims.len() >= 2 {
                let fan_in = dims[1];
                let fan_out = dims[0];
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                let bound = std * 3f64.sqrt();
                let _ = self.uniform_(-bound, bound);
            }
            self
        }

        fn xavier_normal_(&mut self) -> &mut Self {
            let dims = self.size();
            if dims.len() >= 2 {
                let fan_in = dims[1];
                let fan_out = dims[0];
                let std = (2.0 / (fan_in + fan_out) as f64).sqrt();
                let _ = self.normal_(0.0, std);
            }
            self
        }

        fn kaiming_uniform_(&mut self) -> &mut Self {
            let dims = self.size();
            if dims.len() >= 2 {
                let fan_in = dims[1];
                let gain = 2f64.sqrt(); // for ReLU
                let std = gain / (fan_in as f64).sqrt();
                let bound = std * 3f64.sqrt();
                let _ = self.uniform_(-bound, bound);
            }
            self
        }

        fn kaiming_normal_(&mut self) -> &mut Self {
            let dims = self.size();
            if dims.len() >= 2 {
                let fan_in = dims[1];
                let gain = 2f64.sqrt(); // for ReLU
                let std = gain / (fan_in as f64).sqrt();
                let _ = self.normal_(0.0, std);
            }
            self
        }
    }

    /// Helper trait for parameter initialization
    pub trait Init {
        /// Initialize parameters using Xavier/Glorot uniform initialization
        fn xavier_uniform_(&mut self);

        /// Initialize parameters using Xavier/Glorot normal initialization
        fn xavier_normal_(&mut self);

        /// Initialize parameters using Kaiming uniform initialization
        fn kaiming_uniform_(&mut self);

        /// Initialize parameters using Kaiming normal initialization
        fn kaiming_normal_(&mut self);

        /// Initialize parameters with zeros
        fn zeros_(&mut self);

        /// Initialize parameters with ones
        fn ones_(&mut self);
    }

    impl<T: PhoenixModule> Init for T {
        fn xavier_uniform_(&mut self) {
            use self::TensorInit;
            self.apply(|tensor| {
                if tensor.requires_grad() {
                    let _ = tensor.xavier_uniform_();
                }
            });
        }

        fn xavier_normal_(&mut self) {
            use self::TensorInit;
            self.apply(|tensor| {
                if tensor.requires_grad() {
                    let _ = tensor.xavier_normal_();
                }
            });
        }

        fn kaiming_uniform_(&mut self) {
            use self::TensorInit;
            self.apply(|tensor| {
                if tensor.requires_grad() {
                    let _ = tensor.kaiming_uniform_();
                }
            });
        }

        fn kaiming_normal_(&mut self) {
            use self::TensorInit;
            self.apply(|tensor| {
                if tensor.requires_grad() {
                    let _ = tensor.kaiming_normal_();
                }
            });
        }

        fn zeros_(&mut self) {
            self.apply(|tensor| {
                let _ = tensor.zero_();
            });
        }

        fn ones_(&mut self) {
            self.apply(|tensor| {
                let _ = tensor.fill_(1.0);
            });
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Kind, Tensor};

        // Simple test module for validation
        struct TestLinear {
            weight: Tensor,
            bias: Tensor,
            training: bool,
        }

        impl TestLinear {
            fn new(in_features: i64, out_features: i64) -> Self {
                Self {
                    weight: Tensor::randn(&[out_features, in_features], (Kind::Float, Device::Cpu))
                        .set_requires_grad(true),
                    bias: Tensor::randn(&[out_features], (Kind::Float, Device::Cpu))
                        .set_requires_grad(true),
                    training: true,
                }
            }
        }

        impl std::fmt::Debug for TestLinear {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("TestLinear")
                    .field("weight_size", &self.weight.size())
                    .field("bias_size", &self.bias.size())
                    .field("training", &self.training)
                    .finish()
            }
        }

        impl crate::nn::Module for TestLinear {
            fn forward(&self, xs: &Tensor) -> Tensor {
                xs.matmul(&self.weight.tr()) + &self.bias
            }
        }

        impl PhoenixModule for TestLinear {
            fn parameters(&self) -> Vec<&Tensor> {
                vec![&self.weight, &self.bias]
            }

            fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
                vec![&mut self.weight, &mut self.bias]
            }

            fn named_parameters(&self) -> HashMap<String, &Tensor> {
                let mut params = HashMap::new();
                params.insert("weight".to_string(), &self.weight);
                params.insert("bias".to_string(), &self.bias);
                params
            }

            fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
                let mut params = HashMap::new();
                params.insert("weight".to_string(), &mut self.weight);
                params.insert("bias".to_string(), &mut self.bias);
                params
            }

            fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
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
        fn test_phoenix_module_forward() {
            let module = TestLinear::new(5, 10);
            let input = Tensor::randn(&[1, 5], (Kind::Float, Device::Cpu));

            let output = module.forward(&input);
            assert_eq!(output.size(), &[1, 10]);
        }

        #[test]
        fn test_phoenix_module_parameters() {
            let module = TestLinear::new(5, 10);
            let params = module.parameters();

            assert_eq!(params.len(), 2);
            assert!(params.iter().all(|p| p.requires_grad()));
            assert_eq!(module.num_parameters(), 5 * 10 + 10); // weight + bias
            assert_eq!(module.num_trainable_parameters(), 5 * 10 + 10);
        }

        #[test]
        fn test_phoenix_module_named_parameters() {
            let module = TestLinear::new(5, 10);
            let named_params = module.named_parameters();

            assert_eq!(named_params.len(), 2);
            assert!(named_params.contains_key("weight"));
            assert!(named_params.contains_key("bias"));

            let weight = named_params.get("weight").unwrap();
            assert_eq!(weight.size(), &[10, 5]);

            let bias = named_params.get("bias").unwrap();
            assert_eq!(bias.size(), &[10]);
        }

        #[test]
        fn test_phoenix_device_handling() {
            let mut module = TestLinear::new(5, 10);

            assert_eq!(module.device(), Some(Device::Cpu));

            let result = module.to_device(Device::Cpu);
            assert!(result.is_ok());
            assert_eq!(module.device(), Some(Device::Cpu));
        }

        #[test]
        fn test_phoenix_training_mode() {
            let mut module = TestLinear::new(5, 10);

            assert!(module.is_training());

            module.set_training(false);
            assert!(!module.is_training());

            module.set_training(true);
            assert!(module.is_training());
        }

        #[test]
        fn test_phoenix_state_dict() {
            let module = TestLinear::new(5, 10);
            let state_dict = module.state_dict();

            assert_eq!(state_dict.len(), 2);
            assert!(state_dict.contains_key("weight"));
            assert!(state_dict.contains_key("bias"));
        }

        #[test]
        fn test_phoenix_initialization() {
            let mut module = TestLinear::new(5, 10);

            // Test different initialization methods
            module.xavier_uniform_();
            module.xavier_normal_();
            module.kaiming_uniform_();
            module.kaiming_normal_();
            module.zeros_();
            module.ones_();
        }
    }
}

// Re-export Phoenix module functionality when feature is enabled
#[cfg(feature = "torch-rs")]
pub use module::*;
