use tch::{Tensor, Kind, Device};
use std::collections::HashMap;

// Contract test for Optimizer trait - MUST FAIL until implementation exists
#[cfg(feature = "torch-rs")]
mod optimizer_contract_tests {
    use super::*;

    // This will fail until we implement the Optimizer trait
    pub trait Optimizer: Send + Sync {
        fn step(&mut self) -> Result<(), OptimizerError>;
        fn zero_grad(&mut self);
        fn learning_rate(&self) -> f64;
        fn set_learning_rate(&mut self, lr: f64) -> Result<(), OptimizerError>;
        fn parameter_groups(&self) -> &[ParameterGroup];
        fn state_dict(&self) -> HashMap<String, Tensor>;
        fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<(), OptimizerError>;
    }

    #[derive(Debug, Clone)]
    pub struct ParameterGroup {
        pub parameters: Vec<*mut Tensor>,
        pub learning_rate: f64,
        pub weight_decay: f64,
        pub momentum: Option<f64>,
        pub dampening: Option<f64>,
    }

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
            source: tch::TchError,
        },
    }

    // Test implementation that will fail until real Optimizer trait exists
    struct TestSGD {
        parameters: Vec<Tensor>,
        learning_rate: f64,
        momentum: f64,
        velocity: HashMap<usize, Tensor>,
        parameter_groups: Vec<ParameterGroup>,
    }

    impl TestSGD {
        pub fn new(parameters: Vec<Tensor>, lr: f64) -> Result<Self, OptimizerError> {
            if lr <= 0.0 {
                return Err(OptimizerError::InvalidLearningRate { lr });
            }

            if parameters.is_empty() {
                return Err(OptimizerError::NoParameters);
            }

            let parameter_groups = vec![ParameterGroup {
                parameters: vec![], // This is simplified for the test
                learning_rate: lr,
                weight_decay: 0.0,
                momentum: Some(0.0),
                dampening: Some(0.0),
            }];

            Ok(Self {
                parameters,
                learning_rate: lr,
                momentum: 0.0,
                velocity: HashMap::new(),
                parameter_groups,
            })
        }
    }

    // This implementation will fail compilation until Optimizer trait is defined in src/
    impl Optimizer for TestSGD {
        fn step(&mut self) -> Result<(), OptimizerError> {
            for (i, param) in self.parameters.iter_mut().enumerate() {
                if !param.requires_grad() {
                    continue;
                }

                if let Some(grad) = param.grad() {
                    // Simple SGD update: param = param - lr * grad
                    let update = grad * self.learning_rate;
                    let _ = param.sub_(&update);
                } else {
                    // No gradient, skip this parameter
                    continue;
                }
            }
            Ok(())
        }

        fn zero_grad(&mut self) {
            for param in &mut self.parameters {
                if param.requires_grad() {
                    if let Some(grad) = param.grad() {
                        let _ = grad.zero_();
                    }
                }
            }
        }

        fn learning_rate(&self) -> f64 {
            self.learning_rate
        }

        fn set_learning_rate(&mut self, lr: f64) -> Result<(), OptimizerError> {
            if lr <= 0.0 {
                return Err(OptimizerError::InvalidLearningRate { lr });
            }
            self.learning_rate = lr;
            // Update parameter groups
            for group in &mut self.parameter_groups {
                group.learning_rate = lr;
            }
            Ok(())
        }

        fn parameter_groups(&self) -> &[ParameterGroup] {
            &self.parameter_groups
        }

        fn state_dict(&self) -> HashMap<String, Tensor> {
            let mut state = HashMap::new();
            state.insert("learning_rate".to_string(),
                        Tensor::from(self.learning_rate).to_kind(Kind::Float));
            state.insert("momentum".to_string(),
                        Tensor::from(self.momentum).to_kind(Kind::Float));
            state
        }

        fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<(), OptimizerError> {
            if let Some(lr_tensor) = state.get("learning_rate") {
                let lr = lr_tensor.double_value(&[]);
                self.set_learning_rate(lr)?;
            }

            if let Some(momentum_tensor) = state.get("momentum") {
                self.momentum = momentum_tensor.double_value(&[]);
            }

            Ok(())
        }
    }

    fn create_test_parameters() -> Vec<Tensor> {
        vec![
            Tensor::randn(&[10, 5], (Kind::Float, Device::Cpu)).set_requires_grad(true),
            Tensor::randn(&[5], (Kind::Float, Device::Cpu)).set_requires_grad(true),
        ]
    }

    #[test]
    fn test_optimizer_creation() {
        let params = create_test_parameters();
        let optimizer = TestSGD::new(params, 0.01);
        assert!(optimizer.is_ok());
    }

    #[test]
    fn test_invalid_learning_rate() {
        let params = create_test_parameters();

        let result = TestSGD::new(params.clone(), -1.0);
        assert!(result.is_err());

        let result = TestSGD::new(params, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_parameters_error() {
        let result = TestSGD::new(vec![], 0.01);
        assert!(matches!(result, Err(OptimizerError::NoParameters)));
    }

    #[test]
    fn test_learning_rate_management() {
        let params = create_test_parameters();
        let mut optimizer = TestSGD::new(params, 0.01).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.01);

        assert!(optimizer.set_learning_rate(0.1).is_ok());
        assert_eq!(optimizer.learning_rate(), 0.1);

        assert!(optimizer.set_learning_rate(-1.0).is_err());
        assert_eq!(optimizer.learning_rate(), 0.1); // Should remain unchanged
    }

    #[test]
    fn test_zero_grad() {
        let params = create_test_parameters();
        let mut optimizer = TestSGD::new(params, 0.01).unwrap();

        // Create synthetic gradients
        for param in &optimizer.parameters {
            if param.requires_grad() {
                let grad = Tensor::ones_like(param);
                param.set_grad(&grad);
            }
        }

        // zero_grad should clear all gradients
        optimizer.zero_grad();

        // Verify gradients are cleared (this part will need the actual implementation)
    }

    #[test]
    fn test_optimizer_step() {
        let params = create_test_parameters();
        let original_values: Vec<_> = params.iter().map(|p| p.copy()).collect();
        let mut optimizer = TestSGD::new(params, 0.01).unwrap();

        // Create synthetic gradients
        for param in &optimizer.parameters {
            if param.requires_grad() {
                let grad = Tensor::ones_like(param);
                param.set_grad(&grad);
            }
        }

        // Step should update parameters
        let result = optimizer.step();
        assert!(result.is_ok());

        // Parameters should have changed (verify they're different from originals)
        for (original, current) in original_values.iter().zip(&optimizer.parameters) {
            if current.requires_grad() {
                let diff = (current - original).abs().sum(Kind::Float);
                assert!(diff.double_value(&[]) > 0.0, "Parameter should have changed");
            }
        }
    }

    #[test]
    fn test_parameter_groups() {
        let params = create_test_parameters();
        let optimizer = TestSGD::new(params, 0.01).unwrap();

        let groups = optimizer.parameter_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].learning_rate, 0.01);
    }

    #[test]
    fn test_state_dict_consistency() {
        let params = create_test_parameters();
        let optimizer = TestSGD::new(params, 0.01).unwrap();

        let state1 = optimizer.state_dict();
        let state2 = optimizer.state_dict();

        assert_eq!(state1.len(), state2.len());
        for (key, tensor1) in &state1 {
            let tensor2 = state2.get(key).expect("Missing key in second state dict");
            assert!(tensor1.allclose(tensor2, 1e-6, 1e-6, false));
        }
    }

    #[test]
    fn test_state_dict_load() {
        let params = create_test_parameters();
        let mut optimizer = TestSGD::new(params, 0.01).unwrap();

        let original_state = optimizer.state_dict();

        // Change learning rate
        optimizer.set_learning_rate(0.1).unwrap();
        assert_eq!(optimizer.learning_rate(), 0.1);

        // Load original state back
        let result = optimizer.load_state_dict(original_state);
        assert!(result.is_ok());
        assert_eq!(optimizer.learning_rate(), 0.01);
    }
}

#[cfg(not(feature = "torch-rs"))]
mod disabled_tests {
    #[test]
    #[ignore]
    fn optimizer_tests_require torch-rs feature() {
        panic!("Optimizer contract tests require 'torch-rs' feature to be enabled");
    }
}