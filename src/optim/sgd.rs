//! SGD Optimizer Implementation for Project Phoenix

#[cfg(feature = "torch-rs")]
pub mod sgd {
    use crate::optim::{utils, OptimizerError, ParameterGroup, PhoenixOptimizer, SGDOptimizer};
    use crate::Tensor;
    use std::collections::HashMap;

    /// Stochastic Gradient Descent (SGD) optimizer
    ///
    /// Implements SGD with optional momentum, weight decay, and Nesterov acceleration.
    ///
    /// # Example
    /// ```rust
    /// use tch::optim::phoenix::{SGD, PhoenixOptimizer};
    /// use tch::nn::phoenix::PhoenixModule;
    ///
    /// let mut model = create_model(); // Some model implementing PhoenixModule
    /// let parameters = model.parameters_mut();
    /// let mut optimizer = SGD::new(parameters, 0.01).unwrap();
    ///
    /// // Training loop
    /// for _ in 0..epochs {
    ///     optimizer.zero_grad();
    ///     let loss = compute_loss(&model);
    ///     loss.backward();
    ///     optimizer.step().unwrap();
    /// }
    /// ```
    pub struct SGD {
        parameter_groups: Vec<ParameterGroup>,
        momentum_buffers: HashMap<usize, HashMap<usize, Tensor>>,
        step_count: usize,
    }

    impl SGD {
        /// Create a new SGD optimizer
        ///
        /// # Arguments
        /// * `parameters` - Model parameters to optimize
        /// * `lr` - Learning rate
        pub fn new(parameters: Vec<&mut Tensor>, lr: f64) -> Result<Self, OptimizerError> {
            utils::validate_lr(lr)?;

            if parameters.is_empty() {
                return Err(OptimizerError::NoParameters);
            }

            let param_ptrs: Vec<*mut Tensor> =
                parameters.into_iter().map(|p| p as *mut _).collect();
            let group = ParameterGroup::new(param_ptrs, lr);
            utils::validate_parameter_group(&group, 0)?;

            Ok(Self {
                parameter_groups: vec![group],
                momentum_buffers: HashMap::new(),
                step_count: 0,
            })
        }

        /// Create SGD with parameter groups
        pub fn with_groups(groups: Vec<ParameterGroup>) -> Result<Self, OptimizerError> {
            if groups.is_empty() {
                return Err(OptimizerError::NoParameters);
            }

            for (i, group) in groups.iter().enumerate() {
                utils::validate_parameter_group(group, i)?;
            }

            Ok(Self { parameter_groups: groups, momentum_buffers: HashMap::new(), step_count: 0 })
        }

        /// Set momentum for all parameter groups
        pub fn set_momentum_all(&mut self, momentum: f64) -> Result<(), OptimizerError> {
            if momentum < 0.0 || momentum >= 1.0 {
                return Err(OptimizerError::InvalidParameter(format!(
                    "Momentum must be in [0, 1), got: {}",
                    momentum
                )));
            }

            for group in &mut self.parameter_groups {
                group.momentum = Some(momentum);
            }
            Ok(())
        }

        /// Set weight decay for all parameter groups
        pub fn set_weight_decay_all(&mut self, weight_decay: f64) -> Result<(), OptimizerError> {
            if weight_decay < 0.0 {
                return Err(OptimizerError::InvalidParameter(format!(
                    "Weight decay must be non-negative, got: {}",
                    weight_decay
                )));
            }

            for group in &mut self.parameter_groups {
                group.weight_decay = weight_decay;
            }
            Ok(())
        }

        /// Enable/disable Nesterov momentum for all parameter groups
        pub fn set_nesterov_all(&mut self, nesterov: bool) {
            for group in &mut self.parameter_groups {
                group.nesterov = Some(nesterov);
            }
        }

        fn step_group(&mut self, group_id: usize) -> Result<(), OptimizerError> {
            let group = &self.parameter_groups[group_id];
            let momentum = group.momentum.unwrap_or(0.0);
            let weight_decay = group.weight_decay;
            let nesterov = group.nesterov.unwrap_or(false);
            let lr = group.learning_rate;
            let dampening = group.dampening.unwrap_or(0.0);

            // Apply weight decay if specified
            if weight_decay != 0.0 {
                utils::apply_weight_decay(&group.parameters, weight_decay);
            }

            // Get or create momentum buffers for this group
            let group_buffers = self.momentum_buffers.entry(group_id).or_insert_with(HashMap::new);

            for (param_id, param_ptr) in group.parameters.iter().enumerate() {
                unsafe {
                    let param = &mut **param_ptr;

                    if !param.requires_grad() {
                        continue;
                    }

                    let grad = param.grad();
                    if !grad.defined() {
                        continue; // No gradient, skip parameter
                    }

                    if momentum != 0.0 {
                        let momentum_buffer = group_buffers
                            .entry(param_id)
                            .or_insert_with(|| Tensor::zeros_like(&grad));

                        // momentum_buffer = momentum * momentum_buffer + (1 - dampening) * grad
                        *momentum_buffer = &*momentum_buffer * momentum + &grad * (1.0 - dampening);

                        let update = if nesterov {
                            // Nesterov momentum: param = param - lr * (momentum * momentum_buffer + grad)
                            &grad + &*momentum_buffer * momentum
                        } else {
                            // Standard momentum: param = param - lr * momentum_buffer
                            momentum_buffer.shallow_clone()
                        };

                        let _ = param.g_add_(&(update * (-lr)));
                    } else {
                        // No momentum: param = param - lr * grad
                        let _ = param.g_add_(&(grad * (-lr)));
                    }
                }
            }

            Ok(())
        }
    }

    impl PhoenixOptimizer for SGD {
        fn step(&mut self) -> Result<(), OptimizerError> {
            self.step_count += 1;

            for group_id in 0..self.parameter_groups.len() {
                self.step_group(group_id)?;
            }

            Ok(())
        }

        fn zero_grad(&mut self) {
            for group in &self.parameter_groups {
                utils::zero_grad_group(&group.parameters);
            }
        }

        fn learning_rate(&self) -> f64 {
            if self.parameter_groups.is_empty() {
                0.0
            } else {
                self.parameter_groups[0].learning_rate
            }
        }

        fn set_learning_rate(&mut self, lr: f64) -> Result<(), OptimizerError> {
            utils::validate_lr(lr)?;

            for group in &mut self.parameter_groups {
                group.learning_rate = lr;
            }
            Ok(())
        }

        fn parameter_groups(&self) -> &[ParameterGroup] {
            &self.parameter_groups
        }

        fn parameter_groups_mut(&mut self) -> &mut [ParameterGroup] {
            &mut self.parameter_groups
        }

        fn add_parameter_group(&mut self, group: ParameterGroup) {
            let group_id = self.parameter_groups.len();
            if let Err(e) = utils::validate_parameter_group(&group, group_id) {
                eprintln!("Warning: Invalid parameter group added: {}", e);
            }
            self.parameter_groups.push(group);
        }

        fn state_dict(&self) -> HashMap<String, Tensor> {
            let mut state = HashMap::new();

            // Save basic optimizer state
            state.insert(
                "step_count".to_string(),
                Tensor::from(self.step_count as i64).to_kind(crate::Kind::Int64),
            );

            // Save momentum buffers
            for (group_id, group_buffers) in &self.momentum_buffers {
                for (param_id, buffer) in group_buffers {
                    let key = format!("momentum_buffer_{}_{}", group_id, param_id);
                    state.insert(key, buffer.copy());
                }
            }

            // Save parameter group configurations
            for (i, group) in self.parameter_groups.iter().enumerate() {
                state.insert(
                    format!("group_{}_lr", i),
                    Tensor::from(group.learning_rate).to_kind(crate::Kind::Float),
                );
                state.insert(
                    format!("group_{}_weight_decay", i),
                    Tensor::from(group.weight_decay).to_kind(crate::Kind::Float),
                );

                if let Some(momentum) = group.momentum {
                    state.insert(
                        format!("group_{}_momentum", i),
                        Tensor::from(momentum).to_kind(crate::Kind::Float),
                    );
                }

                if let Some(dampening) = group.dampening {
                    state.insert(
                        format!("group_{}_dampening", i),
                        Tensor::from(dampening).to_kind(crate::Kind::Float),
                    );
                }

                if let Some(nesterov) = group.nesterov {
                    state.insert(
                        format!("group_{}_nesterov", i),
                        Tensor::from(nesterov as i64).to_kind(crate::Kind::Int64),
                    );
                }
            }

            state
        }

        fn load_state_dict(
            &mut self,
            state_dict: HashMap<String, Tensor>,
        ) -> Result<(), OptimizerError> {
            // Load step count
            if let Some(step_tensor) = state_dict.get("step_count") {
                self.step_count = step_tensor.int64_value(&[]) as usize;
            }

            // Load momentum buffers
            self.momentum_buffers.clear();
            for (key, tensor) in &state_dict {
                if key.starts_with("momentum_buffer_") {
                    let parts: Vec<&str> = key.split('_').collect();
                    if parts.len() == 4 {
                        if let (Ok(group_id), Ok(param_id)) =
                            (parts[2].parse::<usize>(), parts[3].parse::<usize>())
                        {
                            let group_buffers =
                                self.momentum_buffers.entry(group_id).or_insert_with(HashMap::new);
                            group_buffers.insert(param_id, tensor.copy());
                        }
                    }
                }
            }

            // Load parameter group configurations
            for (i, group) in self.parameter_groups.iter_mut().enumerate() {
                if let Some(lr_tensor) = state_dict.get(&format!("group_{}_lr", i)) {
                    group.learning_rate = lr_tensor.double_value(&[]);
                }

                if let Some(wd_tensor) = state_dict.get(&format!("group_{}_weight_decay", i)) {
                    group.weight_decay = wd_tensor.double_value(&[]);
                }

                if let Some(momentum_tensor) = state_dict.get(&format!("group_{}_momentum", i)) {
                    group.momentum = Some(momentum_tensor.double_value(&[]));
                }

                if let Some(dampening_tensor) = state_dict.get(&format!("group_{}_dampening", i)) {
                    group.dampening = Some(dampening_tensor.double_value(&[]));
                }

                if let Some(nesterov_tensor) = state_dict.get(&format!("group_{}_nesterov", i)) {
                    group.nesterov = Some(nesterov_tensor.int64_value(&[]) != 0);
                }
            }

            Ok(())
        }
    }

    impl SGDOptimizer for SGD {
        fn momentum(&self) -> f64 {
            self.parameter_groups.get(0).and_then(|g| g.momentum).unwrap_or(0.0)
        }

        fn dampening(&self) -> f64 {
            self.parameter_groups.get(0).and_then(|g| g.dampening).unwrap_or(0.0)
        }

        fn weight_decay(&self) -> f64 {
            self.parameter_groups.get(0).map(|g| g.weight_decay).unwrap_or(0.0)
        }

        fn nesterov(&self) -> bool {
            self.parameter_groups.get(0).and_then(|g| g.nesterov).unwrap_or(false)
        }

        fn set_momentum(&mut self, momentum: f64) -> Result<(), OptimizerError> {
            self.set_momentum_all(momentum)
        }

        fn set_weight_decay(&mut self, weight_decay: f64) -> Result<(), OptimizerError> {
            self.set_weight_decay_all(weight_decay)
        }

        fn set_dampening(&mut self, dampening: f64) -> Result<(), OptimizerError> {
            if dampening < 0.0 {
                return Err(OptimizerError::InvalidParameter(format!(
                    "Dampening must be non-negative, got: {}",
                    dampening
                )));
            }

            for group in &mut self.parameter_groups {
                group.dampening = Some(dampening);
            }
            Ok(())
        }

        fn set_nesterov(&mut self, nesterov: bool) {
            self.set_nesterov_all(nesterov);
        }
    }

    /// SGD Builder for convenient construction
    pub struct SGDBuilder {
        lr: f64,
        momentum: Option<f64>,
        weight_decay: f64,
        dampening: f64,
        nesterov: bool,
    }

    impl SGDBuilder {
        pub fn new(lr: f64) -> Self {
            Self { lr, momentum: None, weight_decay: 0.0, dampening: 0.0, nesterov: false }
        }

        pub fn momentum(mut self, momentum: f64) -> Self {
            self.momentum = Some(momentum);
            self
        }

        pub fn weight_decay(mut self, weight_decay: f64) -> Self {
            self.weight_decay = weight_decay;
            self
        }

        pub fn dampening(mut self, dampening: f64) -> Self {
            self.dampening = dampening;
            self
        }

        pub fn nesterov(mut self, nesterov: bool) -> Self {
            self.nesterov = nesterov;
            self
        }

        pub fn build(self, parameters: Vec<&mut Tensor>) -> Result<SGD, OptimizerError> {
            let param_ptrs: Vec<*mut Tensor> =
                parameters.into_iter().map(|p| p as *mut _).collect();

            let group = ParameterGroup::new(param_ptrs, self.lr)
                .weight_decay(self.weight_decay)
                .dampening(self.dampening)
                .nesterov(self.nesterov);

            let group =
                if let Some(momentum) = self.momentum { group.momentum(momentum) } else { group };

            SGD::with_groups(vec![group])
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Device, Kind, Tensor};

        fn create_test_parameters() -> Vec<Tensor> {
            vec![
                Tensor::randn(&[10, 5], (Kind::Float, Device::Cpu)).set_requires_grad(true),
                Tensor::randn(&[5], (Kind::Float, Device::Cpu)).set_requires_grad(true),
            ]
        }

        #[test]
        fn test_sgd_creation() {
            let mut params = create_test_parameters();
            let param_refs: Vec<&mut Tensor> = params.iter_mut().collect();

            let optimizer = SGD::new(param_refs, 0.01);
            assert!(optimizer.is_ok());

            let opt = optimizer.unwrap();
            assert_eq!(opt.learning_rate(), 0.01);
            assert_eq!(opt.momentum(), 0.0);
            assert_eq!(opt.weight_decay(), 0.0);
            assert!(!opt.nesterov());
        }

        #[test]
        fn test_sgd_invalid_lr() {
            let mut params = create_test_parameters();
            let param_refs: Vec<&mut Tensor> = params.iter_mut().collect();

            assert!(SGD::new(param_refs, -0.01).is_err());
            assert!(SGD::new(vec![], 0.01).is_err());
        }

        #[test]
        fn test_sgd_step() {
            let mut params = create_test_parameters();
            let original_values: Vec<_> = params.iter().map(|p| p.copy()).collect();

            // Set up gradients
            for param in &params {
                let grad = Tensor::ones_like(param);
                param.set_grad(&grad);
            }

            let param_refs: Vec<&mut Tensor> = params.iter_mut().collect();
            let mut optimizer = SGD::new(param_refs, 0.01).unwrap();

            assert!(optimizer.step().is_ok());

            // Parameters should have changed
            for (original, current) in original_values.iter().zip(&params) {
                let diff = (current - original).abs().sum(Kind::Float);
                assert!(diff.double_value(&[]) > 0.0);
            }
        }

        #[test]
        fn test_sgd_with_momentum() {
            let mut params = create_test_parameters();
            let param_refs: Vec<&mut Tensor> = params.iter_mut().collect();

            let mut optimizer = SGD::new(param_refs, 0.01).unwrap();
            optimizer.set_momentum(0.9).unwrap();

            assert_eq!(optimizer.momentum(), 0.9);

            // Set gradients and step
            for param in &params {
                let grad = Tensor::ones_like(param);
                param.set_grad(&grad);
            }

            assert!(optimizer.step().is_ok());
        }

        #[test]
        fn test_sgd_with_weight_decay() {
            let mut params = create_test_parameters();
            let param_refs: Vec<&mut Tensor> = params.iter_mut().collect();

            let mut optimizer = SGD::new(param_refs, 0.01).unwrap();
            optimizer.set_weight_decay(0.0001).unwrap();

            assert_eq!(optimizer.weight_decay(), 0.0001);
        }

        #[test]
        fn test_sgd_builder() {
            let mut params = create_test_parameters();
            let param_refs: Vec<&mut Tensor> = params.iter_mut().collect();

            let optimizer = SGDBuilder::new(0.01)
                .momentum(0.9)
                .weight_decay(0.0001)
                .nesterov(true)
                .build(param_refs);

            assert!(optimizer.is_ok());
            let opt = optimizer.unwrap();

            assert_eq!(opt.learning_rate(), 0.01);
            assert_eq!(opt.momentum(), 0.9);
            assert_eq!(opt.weight_decay(), 0.0001);
            assert!(opt.nesterov());
        }

        #[test]
        fn test_sgd_state_dict() {
            let mut params = create_test_parameters();
            let param_refs: Vec<&mut Tensor> = params.iter_mut().collect();

            let mut optimizer = SGDBuilder::new(0.01).momentum(0.9).build(param_refs).unwrap();

            // Take a step to create momentum buffers
            for param in &params {
                let grad = Tensor::ones_like(param);
                param.set_grad(&grad);
            }
            optimizer.step().unwrap();

            let state_dict = optimizer.state_dict();
            assert!(state_dict.contains_key("step_count"));
            assert!(state_dict.contains_key("group_0_lr"));
            assert!(state_dict.contains_key("group_0_momentum"));

            // Test loading state dict
            let mut new_params = create_test_parameters();
            let new_param_refs: Vec<&mut Tensor> = new_params.iter_mut().collect();
            let mut new_optimizer = SGD::new(new_param_refs, 0.05).unwrap();

            assert!(new_optimizer.load_state_dict(state_dict).is_ok());
            assert_eq!(new_optimizer.learning_rate(), 0.01); // Should be loaded from state dict
        }

        #[test]
        fn test_zero_grad() {
            let mut params = create_test_parameters();

            // Set gradients
            for param in &params {
                let grad = Tensor::ones_like(param);
                param.set_grad(&grad);
            }

            let param_refs: Vec<&mut Tensor> = params.iter_mut().collect();
            let mut optimizer = SGD::new(param_refs, 0.01).unwrap();

            optimizer.zero_grad();

            // This test demonstrates the interface - actual gradient checking
            // depends on the tch implementation details
        }
    }
}

// Re-export when Phoenix feature is enabled
#[cfg(feature = "torch-rs")]
pub use sgd::*;
