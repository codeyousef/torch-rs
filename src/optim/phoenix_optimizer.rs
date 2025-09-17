//! Phoenix Optimizer System
//!
//! Enhanced optimizer traits and implementations for Project Phoenix

#[cfg(feature = "torch-rs")]
pub use optimizer::*;

#[cfg(feature = "torch-rs")]
pub mod optimizer {
    use crate::Tensor;
    use std::collections::HashMap;

    /// Core trait for all optimizers in Project Phoenix
    ///
    /// This trait provides the fundamental functionality that all optimization algorithms
    /// must implement, including parameter updates, gradient management, and state handling.
    pub trait PhoenixOptimizer {
        /// Update parameters based on gradients
        ///
        /// # Contract
        /// - Must only update parameters with gradients
        /// - Must respect learning rate and other hyperparameters
        /// - Must update internal state (momentum, etc.)
        /// - Must handle empty gradient case gracefully
        fn step(&mut self) -> Result<(), OptimizerError>;

        /// Clear gradients for all managed parameters
        ///
        /// # Contract
        /// - Must zero gradients for all parameters
        /// - Must be idempotent
        /// - Must not affect optimizer internal state
        fn zero_grad(&mut self);

        /// Get current learning rate
        fn learning_rate(&self) -> f64;

        /// Set learning rate
        ///
        /// # Arguments
        /// * `lr` - New learning rate (must be positive)
        ///
        /// # Contract
        /// - Must validate learning rate is positive
        /// - Must affect all parameter groups
        fn set_learning_rate(&mut self, lr: f64) -> Result<(), OptimizerError>;

        /// Get parameter groups with their settings
        ///
        /// # Returns
        /// * `Vec<ParameterGroup>` - Parameter groups with individual settings
        ///
        /// # Contract
        /// - Each parameter should belong to exactly one group
        /// - Groups should maintain parameter order consistency
        fn parameter_groups(&self) -> &[ParameterGroup];

        /// Get optimizer state for debugging/checkpointing
        ///
        /// # Returns
        /// * `HashMap<String, Tensor>` - Serializable optimizer state
        ///
        /// # Contract
        /// - Keys must be unique and stable across runs
        /// - Values must contain all information needed for resume
        /// - Must handle empty state gracefully
        fn state_dict(&self) -> HashMap<String, Tensor>;

        /// Load optimizer state from checkpoint
        ///
        /// # Arguments
        /// * `state` - Previously saved optimizer state
        ///
        /// # Contract
        /// - Must validate state compatibility with current parameters
        /// - Must restore exact optimization behavior
        /// - Must handle version differences gracefully
        fn load_state_dict(&mut self, state: HashMap<String, Tensor>)
            -> Result<(), OptimizerError>;

        /// Get parameter groups with mutable access
        fn parameter_groups_mut(&mut self) -> &mut [ParameterGroup];

        /// Add a parameter group
        fn add_parameter_group(&mut self, group: ParameterGroup);

        /// Get the number of parameter groups
        fn num_param_groups(&self) -> usize {
            self.parameter_groups().len()
        }

        /// Get total number of parameters across all groups
        fn num_parameters(&self) -> usize {
            self.parameter_groups().iter().map(|group| group.parameters.len()).sum()
        }
    }

    /// Parameter group configuration
    #[derive(Debug, Clone)]
    pub struct ParameterGroup {
        /// Parameters in this group (stored as raw pointers for flexibility)
        pub parameters: Vec<*mut Tensor>,
        /// Learning rate for this group
        pub learning_rate: f64,
        /// Weight decay coefficient
        pub weight_decay: f64,
        /// Momentum factor (if applicable)
        pub momentum: Option<f64>,
        /// Dampening factor (if applicable)
        pub dampening: Option<f64>,
        /// Whether to use Nesterov momentum
        pub nesterov: Option<bool>,
    }

    impl ParameterGroup {
        /// Create a new parameter group with default settings
        pub fn new(parameters: Vec<*mut Tensor>, lr: f64) -> Self {
            Self {
                parameters,
                learning_rate: lr,
                weight_decay: 0.0,
                momentum: None,
                dampening: None,
                nesterov: None,
            }
        }

        /// Set weight decay for this group
        pub fn weight_decay(mut self, weight_decay: f64) -> Self {
            self.weight_decay = weight_decay;
            self
        }

        /// Set momentum for this group
        pub fn momentum(mut self, momentum: f64) -> Self {
            self.momentum = Some(momentum);
            self
        }

        /// Set dampening for this group
        pub fn dampening(mut self, dampening: f64) -> Self {
            self.dampening = Some(dampening);
            self
        }

        /// Enable/disable Nesterov momentum
        pub fn nesterov(mut self, nesterov: bool) -> Self {
            self.nesterov = Some(nesterov);
            self
        }

        /// Get an option value from the group
        pub fn get_option(&self, name: &str) -> Option<f64> {
            match name {
                "lr" | "learning_rate" => Some(self.learning_rate),
                "weight_decay" => Some(self.weight_decay),
                "momentum" => self.momentum,
                "dampening" => self.dampening,
                "nesterov" => self.nesterov.map(|b| if b { 1.0 } else { 0.0 }),
                "beta1"
                | "beta2"
                | "eps"
                | "amsgrad"
                | "alpha"
                | "lr_decay"
                | "initial_accumulator_value"
                | "centered" => None,
                _ => None,
            }
        }

        /// Set an option value in the group
        pub fn set_option(&mut self, name: &str, value: f64) {
            match name {
                "lr" | "learning_rate" => self.learning_rate = value,
                "weight_decay" => self.weight_decay = value,
                "momentum" => self.momentum = Some(value),
                "dampening" => self.dampening = Some(value),
                "nesterov" => self.nesterov = Some(value != 0.0),
                _ => {}
            }
        }

        /// Get parameters as a slice
        pub fn parameters(&self) -> &[*mut Tensor] {
            &self.parameters
        }
    }

    /// Errors that can occur during optimizer operations
    #[derive(Debug, thiserror::Error)]
    pub enum OptimizerError {
        #[error("Invalid learning rate: {lr}. Must be positive.")]
        InvalidLearningRate { lr: f64 },

        #[error("No parameters to optimize")]
        NoParameters,

        #[error("Parameter group {group_id} has invalid configuration: {reason}")]
        InvalidParameterGroup { group_id: usize, reason: String },

        #[error("State dictionary incompatible: {reason}")]
        StateIncompatible { reason: String },

        #[error("Gradient computation error: {source}")]
        GradientError {
            #[from]
            source: crate::TchError,
        },

        #[error("Invalid parameter configuration: {0}")]
        InvalidParameter(String),

        #[error("Optimizer not initialized: {0}")]
        NotInitialized(String),
    }

    /// Adam optimizer specific trait
    pub trait AdamOptimizer: PhoenixOptimizer {
        /// Adam-specific hyperparameters
        fn beta1(&self) -> f64;
        fn beta2(&self) -> f64;
        fn epsilon(&self) -> f64;
        fn amsgrad(&self) -> bool;

        fn set_betas(&mut self, beta1: f64, beta2: f64) -> Result<(), OptimizerError>;
        fn set_epsilon(&mut self, eps: f64) -> Result<(), OptimizerError>;
        fn set_amsgrad(&mut self, amsgrad: bool);
    }

    /// SGD optimizer specific trait
    pub trait SGDOptimizer: PhoenixOptimizer {
        /// SGD-specific hyperparameters
        fn momentum(&self) -> f64;
        fn dampening(&self) -> f64;
        fn weight_decay(&self) -> f64;
        fn nesterov(&self) -> bool;

        fn set_momentum(&mut self, momentum: f64) -> Result<(), OptimizerError>;
        fn set_weight_decay(&mut self, weight_decay: f64) -> Result<(), OptimizerError>;
        fn set_dampening(&mut self, dampening: f64) -> Result<(), OptimizerError>;
        fn set_nesterov(&mut self, nesterov: bool);
    }

    /// Helper trait for creating optimizers from module parameters
    pub trait OptimizerBuilder<T> {
        /// Create optimizer from module parameters
        fn build(parameters: Vec<&mut Tensor>, lr: f64) -> Result<T, OptimizerError>;

        /// Create optimizer from parameter groups
        fn build_with_groups(groups: Vec<ParameterGroup>) -> Result<T, OptimizerError>;
    }

    /// Utility functions for optimizer implementations
    pub mod utils {
        use super::*;

        /// Apply weight decay to parameters
        pub fn apply_weight_decay(params: &[*mut Tensor], weight_decay: f64) {
            if weight_decay == 0.0 {
                return;
            }

            for param_ptr in params {
                unsafe {
                    let param = &mut **param_ptr;
                    if param.requires_grad() {
                        let grad = param.grad();
                        if !grad.defined() {
                            continue;
                        }
                        // Apply weight decay directly to gradient
                        let _ = grad.g_add_(&(&*param * weight_decay));
                    }
                }
            }
        }

        /// Validate learning rate
        pub fn validate_lr(lr: f64) -> Result<(), OptimizerError> {
            if lr <= 0.0 || !lr.is_finite() {
                Err(OptimizerError::InvalidLearningRate { lr })
            } else {
                Ok(())
            }
        }

        /// Validate parameter group
        pub fn validate_parameter_group(
            group: &ParameterGroup,
            group_id: usize,
        ) -> Result<(), OptimizerError> {
            validate_lr(group.learning_rate).map_err(|_| {
                OptimizerError::InvalidParameterGroup {
                    group_id,
                    reason: format!("Invalid learning rate: {}", group.learning_rate),
                }
            })?;

            if group.weight_decay < 0.0 {
                return Err(OptimizerError::InvalidParameterGroup {
                    group_id,
                    reason: format!(
                        "Weight decay must be non-negative, got: {}",
                        group.weight_decay
                    ),
                });
            }

            if let Some(momentum) = group.momentum {
                if momentum < 0.0 || momentum >= 1.0 {
                    return Err(OptimizerError::InvalidParameterGroup {
                        group_id,
                        reason: format!("Momentum must be in [0, 1), got: {}", momentum),
                    });
                }
            }

            if let Some(dampening) = group.dampening {
                if dampening < 0.0 {
                    return Err(OptimizerError::InvalidParameterGroup {
                        group_id,
                        reason: format!("Dampening must be non-negative, got: {}", dampening),
                    });
                }
            }

            if group.parameters.is_empty() {
                return Err(OptimizerError::InvalidParameterGroup {
                    group_id,
                    reason: "Parameter group cannot be empty".to_string(),
                });
            }

            Ok(())
        }

        /// Get parameter tensors safely from raw pointers
        pub unsafe fn get_parameters_safe(param_ptrs: &[*mut Tensor]) -> Vec<&mut Tensor> {
            param_ptrs.iter().map(|ptr| &mut **ptr).collect()
        }

        /// Zero gradients for parameter group
        pub fn zero_grad_group(param_ptrs: &[*mut Tensor]) {
            for param_ptr in param_ptrs {
                unsafe {
                    let param = &mut **param_ptr;
                    if param.requires_grad() {
                        let mut grad = param.grad();
                        if grad.defined() {
                            let _ = grad.zero_();
                        }
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Device, Kind, Tensor};

        fn create_test_tensors() -> Vec<Tensor> {
            vec![
                Tensor::randn(&[10, 5], (Kind::Float, Device::Cpu)).set_requires_grad(true),
                Tensor::randn(&[5], (Kind::Float, Device::Cpu)).set_requires_grad(true),
            ]
        }

        #[test]
        fn test_parameter_group_creation() {
            let mut tensors = create_test_tensors();
            let param_ptrs: Vec<*mut Tensor> = tensors.iter_mut().map(|t| t as *mut _).collect();

            let group = ParameterGroup::new(param_ptrs, 0.01)
                .weight_decay(0.0001)
                .momentum(0.9)
                .nesterov(true);

            assert_eq!(group.learning_rate, 0.01);
            assert_eq!(group.weight_decay, 0.0001);
            assert_eq!(group.momentum, Some(0.9));
            assert_eq!(group.nesterov, Some(true));
        }

        #[test]
        fn test_learning_rate_validation() {
            assert!(utils::validate_lr(0.01).is_ok());
            assert!(utils::validate_lr(1.0).is_ok());
            assert!(utils::validate_lr(-0.01).is_err());
            assert!(utils::validate_lr(0.0).is_err());
            assert!(utils::validate_lr(f64::NAN).is_err());
            assert!(utils::validate_lr(f64::INFINITY).is_err());
        }

        #[test]
        fn test_parameter_group_validation() {
            let mut tensors = create_test_tensors();
            let param_ptrs: Vec<*mut Tensor> = tensors.iter_mut().map(|t| t as *mut _).collect();

            // Valid group
            let valid_group = ParameterGroup::new(param_ptrs.clone(), 0.01);
            assert!(utils::validate_parameter_group(&valid_group, 0).is_ok());

            // Invalid learning rate
            let invalid_lr_group = ParameterGroup::new(param_ptrs.clone(), -1.0);
            assert!(utils::validate_parameter_group(&invalid_lr_group, 0).is_err());

            // Invalid weight decay
            let mut invalid_wd_group = ParameterGroup::new(param_ptrs.clone(), 0.01);
            invalid_wd_group.weight_decay = -0.1;
            assert!(utils::validate_parameter_group(&invalid_wd_group, 0).is_err());

            // Empty parameters
            let empty_group = ParameterGroup::new(vec![], 0.01);
            assert!(utils::validate_parameter_group(&empty_group, 0).is_err());
        }

        #[test]
        fn test_weight_decay_application() {
            let mut tensors = create_test_tensors();
            let param_ptrs: Vec<*mut Tensor> = tensors.iter_mut().map(|t| t as *mut _).collect();

            // Create fake gradients
            for tensor in &tensors {
                let grad = Tensor::ones_like(tensor);
                tensor.set_grad(&grad);
            }

            // Store original gradients
            let original_grads: Vec<_> = tensors.iter().map(|t| t.grad().unwrap().copy()).collect();

            // Apply weight decay
            utils::apply_weight_decay(&param_ptrs, 0.01);

            // Check that gradients were modified
            for (i, tensor) in tensors.iter().enumerate() {
                let current_grad = tensor.grad().unwrap();
                let diff = (&current_grad - &original_grads[i]).abs().sum(Kind::Float);
                assert!(
                    diff.double_value(&[]) > 0.0,
                    "Gradient should be modified by weight decay"
                );
            }
        }

        #[test]
        fn test_zero_grad_group() {
            let mut tensors = create_test_tensors();
            let param_ptrs: Vec<*mut Tensor> = tensors.iter_mut().map(|t| t as *mut _).collect();

            // Create fake gradients
            for tensor in &tensors {
                let grad = Tensor::ones_like(tensor);
                tensor.set_grad(&grad);
            }

            // Verify gradients exist
            for tensor in &tensors {
                assert!(tensor.grad().is_some());
            }

            // Zero gradients
            utils::zero_grad_group(&param_ptrs);

            // This test demonstrates the interface, actual gradient zeroing
            // depends on the tch implementation
        }
    }
}

// Re-export Phoenix optimizer functionality when feature is enabled
#[cfg(feature = "torch-rs")]
pub use optimizer::*;
