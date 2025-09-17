//! Phoenix Linear Layer Implementation
//!
//! Enhanced linear layer with automatic parameter discovery

#[cfg(feature = "torch-rs")]
pub mod linear {
    use crate::nn::phoenix::PhoenixModuleError;
    use crate::{nn::phoenix::PhoenixModule, nn::Module, Device, Kind, Tensor};
    use std::collections::HashMap;

    /// Phoenix Linear Layer
    ///
    /// A fully-connected linear transformation: y = xA^T + b
    ///
    /// # Example
    /// ```rust
    /// use tch::nn::phoenix::Linear;
    /// use tch::{Device, Kind, Tensor};
    ///
    /// let layer = Linear::new(784, 128);
    /// let input = Tensor::randn(&[32, 784], (Kind::Float, Device::Cpu));
    /// let output = layer.forward(&input).unwrap();
    /// assert_eq!(output.size(), &[32, 128]);
    /// ```
    #[derive(Debug)]
    pub struct Linear {
        /// Weight matrix of shape (out_features, in_features)
        pub weight: Tensor,
        /// Bias vector of shape (out_features,)
        pub bias: Option<Tensor>,
        /// Input features
        pub in_features: i64,
        /// Output features
        pub out_features: i64,
        /// Training mode
        training: bool,
    }

    impl Linear {
        /// Create a new Linear layer
        ///
        /// # Arguments
        /// * `in_features` - Size of each input sample
        /// * `out_features` - Size of each output sample
        ///
        /// # Returns
        /// A new Linear layer with randomly initialized parameters
        pub fn new(in_features: i64, out_features: i64) -> Self {
            Self::new_with_bias(in_features, out_features, true)
        }

        /// Create a new Linear layer with optional bias
        ///
        /// # Arguments
        /// * `in_features` - Size of each input sample
        /// * `out_features` - Size of each output sample
        /// * `bias` - Whether to include bias term
        pub fn new_with_bias(in_features: i64, out_features: i64, bias: bool) -> Self {
            let weight = Tensor::empty(&[out_features, in_features], (Kind::Float, Device::Cpu))
                .set_requires_grad(true);

            // Initialize with Xavier uniform
            let bound = (6.0 / (in_features + out_features) as f64).sqrt();
            let _ = weight.uniform_(-bound, bound);

            let bias_tensor = if bias {
                let bias_tensor = Tensor::empty(&[out_features], (Kind::Float, Device::Cpu))
                    .set_requires_grad(true);
                let _ = bias_tensor.uniform_(-bound, bound);
                Some(bias_tensor)
            } else {
                None
            };

            Self { weight, bias: bias_tensor, in_features, out_features, training: true }
        }

        /// Create Linear layer from existing tensors
        pub fn from_tensors(
            weight: Tensor,
            bias: Option<Tensor>,
        ) -> Result<Self, PhoenixModuleError> {
            let weight_shape = weight.size();
            if weight_shape.len() != 2 {
                return Err(PhoenixModuleError::InvalidConfiguration(format!(
                    "Weight must be 2D, got shape: {:?}",
                    weight_shape
                )));
            }

            let out_features = weight_shape[0];
            let in_features = weight_shape[1];

            if let Some(ref bias_tensor) = bias {
                let bias_shape = bias_tensor.size();
                if bias_shape.len() != 1 || bias_shape[0] != out_features {
                    return Err(PhoenixModuleError::InvalidConfiguration(format!(
                        "Bias must have shape [{}], got {:?}",
                        out_features, bias_shape
                    )));
                }
            }

            Ok(Self { weight, bias, in_features, out_features, training: true })
        }

        /// Reset parameters using Xavier uniform initialization
        pub fn reset_parameters(&mut self) {
            let bound = (6.0 / (self.in_features + self.out_features) as f64).sqrt();
            let _ = self.weight.uniform_(-bound, bound);

            if let Some(ref bias) = self.bias {
                let _ = bias.uniform_(-bound, bound);
            }
        }

        /// Set bias term
        pub fn set_bias(&mut self, bias: Option<Tensor>) -> Result<(), PhoenixModuleError> {
            if let Some(ref bias_tensor) = bias {
                let bias_shape = bias_tensor.size();
                if bias_shape.len() != 1 || bias_shape[0] != self.out_features {
                    return Err(PhoenixModuleError::InvalidConfiguration(format!(
                        "Bias must have shape [{}], got {:?}",
                        self.out_features, bias_shape
                    )));
                }
            }
            self.bias = bias;
            Ok(())
        }

        /// Get input features
        pub fn in_features(&self) -> i64 {
            self.in_features
        }

        /// Get output features
        pub fn out_features(&self) -> i64 {
            self.out_features
        }

        /// Check if layer has bias
        pub fn has_bias(&self) -> bool {
            self.bias.is_some()
        }
    }

    impl Module for Linear {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let input_shape = xs.size();

            // Validate input dimensions
            if input_shape.is_empty() || input_shape[input_shape.len() - 1] != self.in_features {
                panic!(
                    "Input size mismatch: expected last dimension to be {}, got {:?}",
                    self.in_features, input_shape
                );
            }

            // Compute linear transformation: y = xW^T + b
            let output = xs.matmul(&self.weight.tr());

            if let Some(ref bias) = self.bias {
                output + bias
            } else {
                output
            }
        }
    }

    impl PhoenixModule for Linear {
        fn parameters(&self) -> Vec<&Tensor> {
            let mut params = vec![&self.weight];
            if let Some(ref bias) = self.bias {
                params.push(bias);
            }
            params
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            let mut params = vec![&mut self.weight];
            if let Some(ref mut bias) = self.bias {
                params.push(bias);
            }
            params
        }

        fn named_parameters(&self) -> HashMap<String, &Tensor> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), &self.weight);
            if let Some(ref bias) = self.bias {
                params.insert("bias".to_string(), bias);
            }
            params
        }

        fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), &mut self.weight);
            if let Some(ref mut bias) = self.bias {
                params.insert("bias".to_string(), bias);
            }
            params
        }

        fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
            self.weight = self.weight.to_device(device);
            if let Some(ref bias) = self.bias {
                self.bias = Some(bias.to_device(device));
            }
            Ok(())
        }

        fn set_training(&mut self, training: bool) {
            self.training = training;
        }

        fn is_training(&self) -> bool {
            self.training
        }

        fn zero_grad(&mut self) {
            let mut grad = self.weight.grad();
            if grad.defined() {
                let _ = grad.zero_();
            }
            if let Some(ref bias) = self.bias {
                let mut grad = bias.grad();
                if grad.defined() {
                    let _ = grad.zero_();
                }
            }
        }

        fn device(&self) -> Option<Device> {
            let weight_device = self.weight.device();
            if let Some(ref bias) = self.bias {
                let bias_device = bias.device();
                if weight_device == bias_device {
                    Some(weight_device)
                } else {
                    None // Mixed devices
                }
            } else {
                Some(weight_device)
            }
        }
    }

    /// Builder for Linear layers
    pub struct LinearBuilder {
        in_features: i64,
        out_features: i64,
        bias: bool,
        device: Device,
        dtype: Kind,
    }

    impl LinearBuilder {
        pub fn new(in_features: i64, out_features: i64) -> Self {
            Self { in_features, out_features, bias: true, device: Device::Cpu, dtype: Kind::Float }
        }

        pub fn bias(mut self, bias: bool) -> Self {
            self.bias = bias;
            self
        }

        pub fn device(mut self, device: Device) -> Self {
            self.device = device;
            self
        }

        pub fn dtype(mut self, dtype: Kind) -> Self {
            self.dtype = dtype;
            self
        }

        pub fn build(self) -> Linear {
            let weight =
                Tensor::empty(&[self.out_features, self.in_features], (self.dtype, self.device))
                    .set_requires_grad(true);

            let bound = (6.0 / (self.in_features + self.out_features) as f64).sqrt();
            let _ = weight.uniform_(-bound, bound);

            let bias_tensor = if self.bias {
                let bias_tensor = Tensor::empty(&[self.out_features], (self.dtype, self.device))
                    .set_requires_grad(true);
                let _ = bias_tensor.uniform_(-bound, bound);
                Some(bias_tensor)
            } else {
                None
            };

            Linear {
                weight,
                bias: bias_tensor,
                in_features: self.in_features,
                out_features: self.out_features,
                training: true,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Device, Kind, Tensor};

        #[test]
        fn test_linear_creation() {
            let layer = Linear::new(784, 128);

            assert_eq!(layer.in_features(), 784);
            assert_eq!(layer.out_features(), 128);
            assert!(layer.has_bias());
            assert_eq!(layer.weight.size(), &[128, 784]);
            assert_eq!(layer.bias.as_ref().unwrap().size(), &[128]);
        }

        #[test]
        fn test_linear_without_bias() {
            let layer = Linear::new_with_bias(784, 128, false);

            assert!(!layer.has_bias());
            assert!(layer.bias.is_none());
        }

        #[test]
        fn test_linear_forward() {
            let layer = Linear::new(5, 3);
            let input = Tensor::randn(&[2, 5], (Kind::Float, Device::Cpu));

            let output = layer.forward(&input);
            assert_eq!(output.size(), &[2, 3]);
        }

        #[test]
        fn test_linear_forward_3d() {
            let layer = Linear::new(10, 5);
            let input = Tensor::randn(&[4, 6, 10], (Kind::Float, Device::Cpu));

            let output = layer.forward(&input);
            assert_eq!(output.size(), &[4, 6, 5]);
        }

        #[test]
        #[should_panic(expected = "Input size mismatch")]
        fn test_linear_forward_wrong_size() {
            let layer = Linear::new(5, 3);
            let input = Tensor::randn(&[2, 7], (Kind::Float, Device::Cpu)); // Wrong input size

            let _ = layer.forward(&input);
        }

        #[test]
        fn test_phoenix_module_implementation() {
            let mut layer = Linear::new(5, 3);

            // Test parameters
            let params = layer.parameters();
            assert_eq!(params.len(), 2); // weight + bias

            let named_params = layer.named_parameters();
            assert_eq!(named_params.len(), 2);
            assert!(named_params.contains_key("weight"));
            assert!(named_params.contains_key("bias"));

            // Test device
            assert_eq!(layer.device(), Some(Device::Cpu));

            // Test training mode
            assert!(layer.is_training());
            layer.set_training(false);
            assert!(!layer.is_training());

            // Test parameter count
            assert_eq!(layer.num_parameters(), 5 * 3 + 3); // weights + bias
        }

        #[test]
        fn test_from_tensors() {
            let weight = Tensor::randn(&[3, 5], (Kind::Float, Device::Cpu));
            let bias = Some(Tensor::randn(&[3], (Kind::Float, Device::Cpu)));

            let layer = Linear::from_tensors(weight.copy(), bias.clone()).unwrap();

            assert_eq!(layer.in_features(), 5);
            assert_eq!(layer.out_features(), 3);
            assert!(layer.has_bias());
        }

        #[test]
        fn test_from_tensors_invalid_weight() {
            let weight = Tensor::randn(&[3], (Kind::Float, Device::Cpu)); // 1D instead of 2D
            let result = Linear::from_tensors(weight, None);

            assert!(result.is_err());
        }

        #[test]
        fn test_from_tensors_invalid_bias() {
            let weight = Tensor::randn(&[3, 5], (Kind::Float, Device::Cpu));
            let bias = Some(Tensor::randn(&[4], (Kind::Float, Device::Cpu))); // Wrong size

            let result = Linear::from_tensors(weight, bias);
            assert!(result.is_err());
        }

        #[test]
        fn test_reset_parameters() {
            let mut layer = Linear::new(5, 3);
            let original_weight = layer.weight.copy();

            layer.reset_parameters();

            // Parameters should be different after reset (with high probability)
            let diff = (&layer.weight - &original_weight).abs().sum(Kind::Float);
            assert!(diff.double_value(&[]) > 0.0);
        }

        #[test]
        fn test_linear_builder() {
            let layer = LinearBuilder::new(784, 256)
                .bias(false)
                .device(Device::Cpu)
                .dtype(Kind::Float)
                .build();

            assert_eq!(layer.in_features(), 784);
            assert_eq!(layer.out_features(), 256);
            assert!(!layer.has_bias());
            assert_eq!(layer.device(), Some(Device::Cpu));
        }

        #[test]
        fn test_gradient_flow() {
            let mut layer = Linear::new(3, 2);
            let input = Tensor::randn(&[1, 3], (Kind::Float, Device::Cpu));

            let output = layer.forward(&input);
            let loss = output.sum(Kind::Float);

            loss.backward();

            // Parameters should have gradients
            assert!(layer.weight.grad().is_some());
            if let Some(ref bias) = layer.bias {
                assert!(bias.grad().is_some());
            }

            // Test zero_grad
            layer.zero_grad();
            // After zero_grad, gradients should be zeroed (implementation dependent)
        }

        #[test]
        fn test_state_dict_compatibility() {
            let layer = Linear::new(5, 3);
            let state_dict = layer.state_dict();

            assert_eq!(state_dict.len(), 2);
            assert!(state_dict.contains_key("weight"));
            assert!(state_dict.contains_key("bias"));

            // Shapes should match
            assert_eq!(state_dict["weight"].size(), &[3, 5]);
            assert_eq!(state_dict["bias"].size(), &[3]);
        }
    }
}

// Re-export when Phoenix feature is enabled
#[cfg(feature = "torch-rs")]
pub use linear::*;
