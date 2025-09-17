//! Phoenix BatchNorm Layer Implementation
//!
//! Enhanced batch normalization with automatic buffer management

#[cfg(feature = "torch-rs")]
pub mod batch_norm {
    use crate::nn::phoenix::PhoenixModuleError;
    use crate::{nn::phoenix::PhoenixModule, nn::Module, Device, Kind, Tensor};
    use std::collections::HashMap;

    /// Phoenix BatchNorm2d Layer
    ///
    /// Applies Batch Normalization over a 4D input (N, C, H, W).
    ///
    /// # Example
    /// ```rust
    /// use tch::nn::phoenix::BatchNorm2d;
    /// use tch::{Device, Kind, Tensor};
    ///
    /// let layer = BatchNorm2d::new(64); // 64 channels
    /// let input = Tensor::randn(&[1, 64, 32, 32], (Kind::Float, Device::Cpu));
    /// let output = layer.forward(&input);
    /// assert_eq!(output.size(), input.size());
    /// ```
    #[derive(Debug)]
    pub struct BatchNorm2d {
        /// Number of features/channels
        pub num_features: i64,
        /// Learnable scale parameter
        pub weight: Tensor,
        /// Learnable shift parameter
        pub bias: Tensor,
        /// Running mean (buffer, not trainable)
        pub running_mean: Tensor,
        /// Running variance (buffer, not trainable)
        pub running_var: Tensor,
        /// Momentum for running statistics
        pub momentum: f64,
        /// Small constant for numerical stability
        pub eps: f64,
        /// Whether to use affine transformation
        pub affine: bool,
        /// Whether to track running statistics
        pub track_running_stats: bool,
        /// Training mode
        training: bool,
        /// Number of batches tracked
        num_batches_tracked: i64,
    }

    impl BatchNorm2d {
        /// Create a new BatchNorm2d layer
        ///
        /// # Arguments
        /// * `num_features` - Number of channels from input
        pub fn new(num_features: i64) -> Self {
            Self::new_with_config(PhoenixBatchNormConfig {
                num_features,
                eps: 1e-5,
                momentum: 0.1,
                affine: true,
                track_running_stats: true,
            })
        }

        /// Create BatchNorm2d with custom configuration
        pub fn new_with_config(config: PhoenixBatchNormConfig) -> Self {
            let PhoenixBatchNormConfig { num_features, eps, momentum, affine, track_running_stats } =
                config;

            assert!(num_features > 0, "num_features must be positive");
            assert!(eps > 0.0, "eps must be positive");
            assert!(momentum >= 0.0 && momentum <= 1.0, "momentum must be in [0, 1]");

            // Initialize parameters
            let weight = if affine {
                Tensor::ones(&[num_features], (Kind::Float, Device::Cpu)).set_requires_grad(true)
            } else {
                Tensor::ones(&[num_features], (Kind::Float, Device::Cpu))
            };

            let bias = if affine {
                Tensor::zeros(&[num_features], (Kind::Float, Device::Cpu)).set_requires_grad(true)
            } else {
                Tensor::zeros(&[num_features], (Kind::Float, Device::Cpu))
            };

            // Initialize buffers (running statistics)
            let running_mean = Tensor::zeros(&[num_features], (Kind::Float, Device::Cpu));
            let running_var = Tensor::ones(&[num_features], (Kind::Float, Device::Cpu));

            Self {
                num_features,
                weight,
                bias,
                running_mean,
                running_var,
                momentum,
                eps,
                affine,
                track_running_stats,
                training: true,
                num_batches_tracked: 0,
            }
        }

        /// Reset running statistics
        pub fn reset_running_stats(&mut self) {
            if self.track_running_stats {
                let _ = self.running_mean.zero_();
                let _ = self.running_var.fill_(1.0);
                self.num_batches_tracked = 0;
            }
        }

        /// Reset parameters to default values
        pub fn reset_parameters(&mut self) {
            let _ = self.weight.fill_(1.0);
            let _ = self.bias.zero_();
            self.reset_running_stats();
        }

        /// Get effective momentum (handles exponential average)
        fn effective_momentum(&self) -> f64 {
            if self.momentum == 0.0 {
                // Exponential average
                1.0 / (self.num_batches_tracked as f64 + 1.0)
            } else {
                self.momentum
            }
        }
    }

    impl Module for BatchNorm2d {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let input_shape = xs.size();

            // Validate input dimensions
            if input_shape.len() != 4 {
                panic!("BatchNorm2d expects 4D input (N, C, H, W), got shape: {:?}", input_shape);
            }

            let input_channels = input_shape[1];
            if input_channels != self.num_features {
                panic!(
                    "Input channels mismatch: expected {}, got {}",
                    self.num_features, input_channels
                );
            }

            if self.training && xs.size()[0] <= 1 {
                // Single sample batch in training mode - use running stats to avoid NaN
                return xs.batch_norm(
                    Some(&self.weight),
                    Some(&self.bias),
                    Some(&self.running_mean),
                    Some(&self.running_var),
                    false,
                    0.0,
                    self.eps,
                    false,
                );
            }

            if self.training {
                // Training mode: compute batch statistics and update running stats
                let output = xs.batch_norm(
                    Some(&self.weight),
                    Some(&self.bias),
                    Some(&self.running_mean),
                    Some(&self.running_var),
                    true,
                    self.effective_momentum(),
                    self.eps,
                    true, // Use input stats
                );

                // Note: In a real implementation, we would need to update num_batches_tracked
                // This would require mutable access which isn't available in forward()
                // In practice, this would be handled by the training loop or a wrapper

                output
            } else {
                // Evaluation mode: use running statistics
                xs.batch_norm(
                    Some(&self.weight),
                    Some(&self.bias),
                    Some(&self.running_mean),
                    Some(&self.running_var),
                    false,
                    0.0,
                    self.eps,
                    false,
                )
            }
        }
    }

    impl PhoenixModule for BatchNorm2d {
        fn parameters(&self) -> Vec<&Tensor> {
            if self.affine {
                vec![&self.weight, &self.bias]
            } else {
                vec![]
            }
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            if self.affine {
                vec![&mut self.weight, &mut self.bias]
            } else {
                vec![]
            }
        }

        fn named_parameters(&self) -> HashMap<String, &Tensor> {
            let mut params = HashMap::new();
            if self.affine {
                params.insert("weight".to_string(), &self.weight);
                params.insert("bias".to_string(), &self.bias);
            }
            params
        }

        fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
            let mut params = HashMap::new();
            if self.affine {
                params.insert("weight".to_string(), &mut self.weight);
                params.insert("bias".to_string(), &mut self.bias);
            }
            params
        }

        fn buffers(&self) -> Vec<&Tensor> {
            if self.track_running_stats {
                vec![&self.running_mean, &self.running_var]
            } else {
                vec![]
            }
        }

        fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
            if self.track_running_stats {
                vec![&mut self.running_mean, &mut self.running_var]
            } else {
                vec![]
            }
        }

        fn named_buffers(&self) -> HashMap<String, &Tensor> {
            let mut buffers = HashMap::new();
            if self.track_running_stats {
                buffers.insert("running_mean".to_string(), &self.running_mean);
                buffers.insert("running_var".to_string(), &self.running_var);
                buffers.insert(
                    "num_batches_tracked".to_string(),
                    &Tensor::from(self.num_batches_tracked),
                );
            }
            buffers
        }

        fn named_buffers_mut(&mut self) -> HashMap<String, &mut Tensor> {
            let mut buffers = HashMap::new();
            if self.track_running_stats {
                buffers.insert("running_mean".to_string(), &mut self.running_mean);
                buffers.insert("running_var".to_string(), &mut self.running_var);
                // Note: num_batches_tracked would need special handling as it's not a Tensor
            }
            buffers
        }

        fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
            self.weight = self.weight.to_device(device);
            self.bias = self.bias.to_device(device);
            if self.track_running_stats {
                self.running_mean = self.running_mean.to_device(device);
                self.running_var = self.running_var.to_device(device);
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
            if self.affine {
                let mut grad = self.weight.grad();
                if grad.defined() {
                    let _ = grad.zero_();
                }
                let mut grad = self.bias.grad();
                if grad.defined() {
                    let _ = grad.zero_();
                }
            }
        }

        fn device(&self) -> Option<Device> {
            let weight_device = self.weight.device();
            let bias_device = self.bias.device();

            if weight_device == bias_device {
                if self.track_running_stats {
                    let mean_device = self.running_mean.device();
                    let var_device = self.running_var.device();

                    if weight_device == mean_device && mean_device == var_device {
                        Some(weight_device)
                    } else {
                        None
                    }
                } else {
                    Some(weight_device)
                }
            } else {
                None
            }
        }
    }

    /// Configuration for BatchNorm layers
    #[derive(Debug, Clone)]
    pub struct PhoenixBatchNormConfig {
        pub num_features: i64,
        pub eps: f64,
        pub momentum: f64,
        pub affine: bool,
        pub track_running_stats: bool,
    }

    /// Builder for BatchNorm2d layers
    pub struct BatchNorm2dBuilder {
        config: PhoenixBatchNormConfig,
        device: Device,
        dtype: Kind,
    }

    impl BatchNorm2dBuilder {
        pub fn new(num_features: i64) -> Self {
            Self {
                config: PhoenixBatchNormConfig {
                    num_features,
                    eps: 1e-5,
                    momentum: 0.1,
                    affine: true,
                    track_running_stats: true,
                },
                device: Device::Cpu,
                dtype: Kind::Float,
            }
        }

        pub fn eps(mut self, eps: f64) -> Self {
            self.config.eps = eps;
            self
        }

        pub fn momentum(mut self, momentum: f64) -> Self {
            self.config.momentum = momentum;
            self
        }

        pub fn affine(mut self, affine: bool) -> Self {
            self.config.affine = affine;
            self
        }

        pub fn track_running_stats(mut self, track_running_stats: bool) -> Self {
            self.config.track_running_stats = track_running_stats;
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

        pub fn build(self) -> BatchNorm2d {
            let mut bn = BatchNorm2d::new_with_config(self.config);
            let _ = bn.to_device(self.device);
            bn
        }
    }

    /// 1D Batch Normalization
    #[derive(Debug)]
    pub struct BatchNorm1d {
        inner: BatchNorm2d,
    }

    impl BatchNorm1d {
        pub fn new(num_features: i64) -> Self {
            Self { inner: BatchNorm2d::new(num_features) }
        }

        pub fn new_with_config(config: PhoenixBatchNormConfig) -> Self {
            Self { inner: BatchNorm2d::new_with_config(config) }
        }
    }

    impl Module for BatchNorm1d {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let input_shape = xs.size();

            match input_shape.len() {
                2 => {
                    // (N, C) -> (N, C, 1, 1) -> (N, C)
                    let reshaped = xs.unsqueeze(-1).unsqueeze(-1);
                    let output = self.inner.forward(&reshaped);
                    output.squeeze_dim(-1).squeeze_dim(-1)
                }
                3 => {
                    // (N, C, L) -> (N, C, L, 1) -> (N, C, L)
                    let reshaped = xs.unsqueeze(-1);
                    let output = self.inner.forward(&reshaped);
                    output.squeeze_dim(-1)
                }
                _ => panic!(
                    "BatchNorm1d expects 2D (N, C) or 3D (N, C, L) input, got shape: {:?}",
                    input_shape
                ),
            }
        }
    }

    crate::impl_phoenix_module!(BatchNorm1d { inner: BatchNorm2d });

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Device, Kind, Tensor};

        #[test]
        fn test_batch_norm_2d_creation() {
            let bn = BatchNorm2d::new(64);

            assert_eq!(bn.num_features, 64);
            assert_eq!(bn.eps, 1e-5);
            assert_eq!(bn.momentum, 0.1);
            assert!(bn.affine);
            assert!(bn.track_running_stats);
            assert_eq!(bn.weight.size(), &[64]);
            assert_eq!(bn.bias.size(), &[64]);
            assert_eq!(bn.running_mean.size(), &[64]);
            assert_eq!(bn.running_var.size(), &[64]);
        }

        #[test]
        fn test_batch_norm_2d_forward() {
            let bn = BatchNorm2d::new(3);
            let input = Tensor::randn(&[4, 3, 32, 32], (Kind::Float, Device::Cpu));

            let output = bn.forward(&input);
            assert_eq!(output.size(), input.size());
        }

        #[test]
        fn test_batch_norm_training_vs_eval() {
            let mut bn = BatchNorm2d::new(3);
            let input = Tensor::randn(&[4, 3, 32, 32], (Kind::Float, Device::Cpu));

            // Training mode
            bn.set_training(true);
            assert!(bn.is_training());
            let train_output = bn.forward(&input);

            // Eval mode
            bn.set_training(false);
            assert!(!bn.is_training());
            let eval_output = bn.forward(&input);

            // Outputs should have same shape but potentially different values
            assert_eq!(train_output.size(), eval_output.size());
        }

        #[test]
        fn test_batch_norm_without_affine() {
            let bn = BatchNorm2dBuilder::new(64).affine(false).build();

            assert!(!bn.affine);
            assert_eq!(bn.parameters().len(), 0); // No learnable parameters
        }

        #[test]
        fn test_batch_norm_without_tracking() {
            let bn = BatchNorm2dBuilder::new(64).track_running_stats(false).build();

            assert!(!bn.track_running_stats);
            assert_eq!(bn.buffers().len(), 0); // No buffers
        }

        #[test]
        fn test_batch_norm_1d() {
            let bn = BatchNorm1d::new(128);

            // Test 2D input (N, C)
            let input_2d = Tensor::randn(&[32, 128], (Kind::Float, Device::Cpu));
            let output_2d = bn.forward(&input_2d);
            assert_eq!(output_2d.size(), input_2d.size());

            // Test 3D input (N, C, L)
            let input_3d = Tensor::randn(&[32, 128, 64], (Kind::Float, Device::Cpu));
            let output_3d = bn.forward(&input_3d);
            assert_eq!(output_3d.size(), input_3d.size());
        }

        #[test]
        fn test_phoenix_module_implementation() {
            let mut bn = BatchNorm2d::new(64);

            // Test parameters (affine=true by default)
            let params = bn.parameters();
            assert_eq!(params.len(), 2); // weight + bias

            let named_params = bn.named_parameters();
            assert_eq!(named_params.len(), 2);
            assert!(named_params.contains_key("weight"));
            assert!(named_params.contains_key("bias"));

            // Test buffers
            let buffers = bn.buffers();
            assert_eq!(buffers.len(), 2); // running_mean + running_var

            let named_buffers = bn.named_buffers();
            assert!(named_buffers.contains_key("running_mean"));
            assert!(named_buffers.contains_key("running_var"));

            // Test device
            assert_eq!(bn.device(), Some(Device::Cpu));

            // Test training mode
            assert!(bn.is_training());
            bn.set_training(false);
            assert!(!bn.is_training());
        }

        #[test]
        #[should_panic(expected = "BatchNorm2d expects 4D input")]
        fn test_invalid_input_dimensions() {
            let bn = BatchNorm2d::new(3);
            let input = Tensor::randn(&[3, 32, 32], (Kind::Float, Device::Cpu)); // 3D instead of 4D
            let _ = bn.forward(&input);
        }

        #[test]
        #[should_panic(expected = "Input channels mismatch")]
        fn test_input_channels_mismatch() {
            let bn = BatchNorm2d::new(3);
            let input = Tensor::randn(&[1, 6, 32, 32], (Kind::Float, Device::Cpu)); // 6 channels instead of 3
            let _ = bn.forward(&input);
        }

        #[test]
        fn test_reset_parameters() {
            let mut bn = BatchNorm2d::new(64);

            // Modify parameters
            let _ = bn.weight.fill_(2.0);
            let _ = bn.bias.fill_(1.0);

            // Reset should restore defaults
            bn.reset_parameters();

            // Weight should be 1, bias should be 0
            let weight_sum = bn.weight.sum(Kind::Float).double_value(&[]);
            let bias_sum = bn.bias.sum(Kind::Float).double_value(&[]);

            assert!((weight_sum - 64.0).abs() < 1e-6);
            assert!(bias_sum.abs() < 1e-6);
        }

        #[test]
        fn test_builder_pattern() {
            let bn = BatchNorm2dBuilder::new(128)
                .eps(1e-6)
                .momentum(0.01)
                .affine(false)
                .track_running_stats(false)
                .build();

            assert_eq!(bn.eps, 1e-6);
            assert_eq!(bn.momentum, 0.01);
            assert!(!bn.affine);
            assert!(!bn.track_running_stats);
        }

        #[test]
        fn test_state_dict() {
            let bn = BatchNorm2d::new(64);
            let state_dict = bn.state_dict();

            // Should contain parameters and buffers
            assert!(state_dict.len() >= 4); // weight, bias, running_mean, running_var
            assert!(state_dict.contains_key("weight"));
            assert!(state_dict.contains_key("bias"));
            assert!(state_dict.contains_key("running_mean"));
            assert!(state_dict.contains_key("running_var"));
        }
    }
}

// Re-export when Phoenix feature is enabled
#[cfg(feature = "torch-rs")]
pub use batch_norm::*;
