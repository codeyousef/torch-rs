//! Adam optimizer implementation with automatic parameter discovery
//!
//! Provides PyTorch-compatible Adam optimizer with adaptive learning rates

use crate::optim::{OptimizerError, ParameterGroup, PhoenixOptimizer};
use crate::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AdamConfig {
    pub lr: f64,
    pub learning_rate: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            learning_rate: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        }
    }
}

#[derive(Debug)]
pub struct Adam {
    parameter_groups: Vec<ParameterGroup>,
    state: HashMap<usize, AdamState>,
    defaults: AdamConfig,
}

#[derive(Debug)]
struct AdamState {
    step: i64,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Option<Tensor>,
}

impl Adam {
    pub fn new<I>(params: I, config: AdamConfig) -> Result<Self, OptimizerError>
    where
        I: IntoIterator<Item = *mut Tensor>,
    {
        let parameter_groups = vec![ParameterGroup::new(params.into_iter().collect())];

        Ok(Self { parameter_groups, state: HashMap::new(), defaults: config })
    }

    pub fn new_with_defaults<I>(params: I) -> Result<Self, OptimizerError>
    where
        I: IntoIterator<Item = *mut Tensor>,
    {
        Self::new(params, AdamConfig::default())
    }

    pub fn with_lr<I>(params: I, lr: f64) -> Result<Self, OptimizerError>
    where
        I: IntoIterator<Item = *mut Tensor>,
    {
        Self::new(params, AdamConfig { lr, ..Default::default() })
    }

    pub fn learning_rate(&self) -> f64 {
        self.defaults.lr
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.defaults.lr = lr;
        for group in &mut self.parameter_groups {
            group.set_option("lr", lr.into());
        }
    }

    pub fn betas(&self) -> (f64, f64) {
        self.defaults.betas
    }

    pub fn set_betas(&mut self, betas: (f64, f64)) {
        self.defaults.betas = betas;
        for group in &mut self.parameter_groups {
            group.set_option("beta1", betas.0.into());
            group.set_option("beta2", betas.1.into());
        }
    }

    pub fn eps(&self) -> f64 {
        self.defaults.eps
    }

    pub fn set_eps(&mut self, eps: f64) {
        self.defaults.eps = eps;
        for group in &mut self.parameter_groups {
            group.set_option("eps", eps.into());
        }
    }

    pub fn weight_decay(&self) -> f64 {
        self.defaults.weight_decay
    }

    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.defaults.weight_decay = weight_decay;
        for group in &mut self.parameter_groups {
            group.set_option("weight_decay", weight_decay.into());
        }
    }

    pub fn amsgrad(&self) -> bool {
        self.defaults.amsgrad
    }

    pub fn set_amsgrad(&mut self, amsgrad: bool) {
        self.defaults.amsgrad = amsgrad;
        for group in &mut self.parameter_groups {
            group.set_option("amsgrad", if amsgrad { 1.0 } else { 0.0 }.into());
        }
    }

    fn get_lr(&self, group: &ParameterGroup) -> f64 {
        group.get_option("lr").unwrap_or(self.defaults.lr.into()).into()
    }

    fn get_beta1(&self, group: &ParameterGroup) -> f64 {
        group.get_option("beta1").unwrap_or(self.defaults.betas.0.into()).into()
    }

    fn get_beta2(&self, group: &ParameterGroup) -> f64 {
        group.get_option("beta2").unwrap_or(self.defaults.betas.1.into()).into()
    }

    fn get_eps(&self, group: &ParameterGroup) -> f64 {
        group.get_option("eps").unwrap_or(self.defaults.eps.into()).into()
    }

    fn get_weight_decay(&self, group: &ParameterGroup) -> f64 {
        group.get_option("weight_decay").unwrap_or(self.defaults.weight_decay.into()).into()
    }

    fn get_amsgrad(&self, group: &ParameterGroup) -> bool {
        let value: f64 = group
            .get_option("amsgrad")
            .unwrap_or((if self.defaults.amsgrad { 1.0 } else { 0.0 }).into())
            .into();
        value > 0.5
    }
}

impl PhoenixOptimizer for Adam {
    fn step(&mut self) -> Result<(), OptimizerError> {
        for group in &mut self.parameter_groups {
            let lr = self.get_lr(group);
            let beta1 = self.get_beta1(group);
            let beta2 = self.get_beta2(group);
            let eps = self.get_eps(group);
            let weight_decay = self.get_weight_decay(group);
            let amsgrad = self.get_amsgrad(group);

            for (param_id, param_ptr) in group.parameters().iter().enumerate() {
                let param = unsafe { &mut **param_ptr };

                let grad = param.grad();
                if grad.defined() {
                    let state = self.state.entry(param_id).or_insert_with(|| AdamState {
                        step: 0,
                        exp_avg: Tensor::zeros_like(&param),
                        exp_avg_sq: Tensor::zeros_like(&param),
                        max_exp_avg_sq: if amsgrad {
                            Some(Tensor::zeros_like(&param))
                        } else {
                            None
                        },
                    });

                    state.step += 1;

                    let mut grad = grad;
                    if weight_decay != 0.0 {
                        grad = grad + &*param * weight_decay;
                    }

                    state.exp_avg = &state.exp_avg * beta1 + &grad * (1.0 - beta1);
                    state.exp_avg_sq = &state.exp_avg_sq * beta2 + (&grad * &grad) * (1.0 - beta2);

                    let bias_correction1 = 1.0 - beta1.powi(state.step as i32);
                    let bias_correction2 = 1.0 - beta2.powi(state.step as i32);

                    let corrected_exp_avg = &state.exp_avg / bias_correction1;

                    let denom = if amsgrad {
                        let max_exp_avg_sq = state.max_exp_avg_sq.as_mut().unwrap();
                        *max_exp_avg_sq = Tensor::max_tensor(max_exp_avg_sq, &state.exp_avg_sq);
                        (max_exp_avg_sq / bias_correction2).sqrt() + eps
                    } else {
                        (&state.exp_avg_sq / bias_correction2).sqrt() + eps
                    };

                    let step_size = lr;
                    let update = &corrected_exp_avg / denom * step_size;

                    let _ = param.g_add_(&(-update));
                }
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &mut self.parameter_groups {
            for param_ptr in group.parameters() {
                let param = unsafe { &mut **param_ptr };
                param.zero_grad();
            }
        }
    }

    fn state_dict(&self) -> HashMap<String, crate::nn::OptimizerValue> {
        use crate::nn::OptimizerValue;

        let mut state_dict = HashMap::new();

        state_dict.insert("lr".to_string(), OptimizerValue::Float(self.defaults.lr));
        state_dict.insert("beta1".to_string(), OptimizerValue::Float(self.defaults.betas.0));
        state_dict.insert("beta2".to_string(), OptimizerValue::Float(self.defaults.betas.1));
        state_dict.insert("eps".to_string(), OptimizerValue::Float(self.defaults.eps));
        state_dict
            .insert("weight_decay".to_string(), OptimizerValue::Float(self.defaults.weight_decay));
        state_dict.insert("amsgrad".to_string(), OptimizerValue::Bool(self.defaults.amsgrad));

        for (param_id, state) in &self.state {
            let prefix = format!("state.{}", param_id);
            state_dict.insert(format!("{}.step", prefix), OptimizerValue::Int(state.step));
            state_dict.insert(
                format!("{}.exp_avg", prefix),
                OptimizerValue::Tensor(state.exp_avg.shallow_clone()),
            );
            state_dict.insert(
                format!("{}.exp_avg_sq", prefix),
                OptimizerValue::Tensor(state.exp_avg_sq.shallow_clone()),
            );

            if let Some(ref max_exp_avg_sq) = state.max_exp_avg_sq {
                state_dict.insert(
                    format!("{}.max_exp_avg_sq", prefix),
                    OptimizerValue::Tensor(max_exp_avg_sq.shallow_clone()),
                );
            }
        }

        state_dict
    }

    fn load_state_dict(
        &mut self,
        state_dict: HashMap<String, Tensor>,
    ) -> Result<(), OptimizerError> {
        use crate::nn::OptimizerValue;

        if let Some(OptimizerValue::Float(lr)) = state_dict.get("lr") {
            self.defaults.lr = *lr;
        }
        if let Some(OptimizerValue::Float(beta1)) = state_dict.get("beta1") {
            self.defaults.betas.0 = *beta1;
        }
        if let Some(OptimizerValue::Float(beta2)) = state_dict.get("beta2") {
            self.defaults.betas.1 = *beta2;
        }
        if let Some(OptimizerValue::Float(eps)) = state_dict.get("eps") {
            self.defaults.eps = *eps;
        }
        if let Some(OptimizerValue::Float(weight_decay)) = state_dict.get("weight_decay") {
            self.defaults.weight_decay = *weight_decay;
        }
        if let Some(OptimizerValue::Bool(amsgrad)) = state_dict.get("amsgrad") {
            self.defaults.amsgrad = *amsgrad;
        }

        self.state.clear();

        let mut param_states: HashMap<usize, AdamState> = HashMap::new();

        for (key, value) in state_dict {
            if let Some(state_key) = key.strip_prefix("state.") {
                if let Some(dot_pos) = state_key.find('.') {
                    let param_id: usize = state_key[..dot_pos].parse().map_err(|_| {
                        OptimizerError::StateIncompatible {
                            reason: "Invalid parameter ID".to_string(),
                        }
                    })?;
                    let field = &state_key[dot_pos + 1..];

                    let state = param_states.entry(param_id).or_insert_with(|| AdamState {
                        step: 0,
                        exp_avg: Tensor::new(),
                        exp_avg_sq: Tensor::new(),
                        max_exp_avg_sq: if self.defaults.amsgrad {
                            Some(Tensor::new())
                        } else {
                            None
                        },
                    });

                    match (field, value) {
                        ("step", OptimizerValue::Int(step)) => state.step = *step,
                        ("exp_avg", OptimizerValue::Tensor(tensor)) => {
                            state.exp_avg = tensor.shallow_clone()
                        }
                        ("exp_avg_sq", OptimizerValue::Tensor(tensor)) => {
                            state.exp_avg_sq = tensor.shallow_clone()
                        }
                        ("max_exp_avg_sq", OptimizerValue::Tensor(tensor)) => {
                            state.max_exp_avg_sq = Some(tensor.shallow_clone());
                        }
                        _ => {
                            return Err(OptimizerError::StateIncompatible {
                                reason: format!("Unknown state field: {}", field),
                            })
                        }
                    }
                }
            }
        }

        self.state = param_states;
        Ok(())
    }

    fn parameter_groups(&self) -> &[ParameterGroup] {
        &self.parameter_groups
    }

    fn parameter_groups_mut(&mut self) -> &mut [ParameterGroup] {
        &mut self.parameter_groups
    }

    fn add_parameter_group(&mut self, group: ParameterGroup) {
        self.parameter_groups.push(group);
    }

    fn learning_rate(&self) -> f64 {
        self.defaults.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) -> Result<(), OptimizerError> {
        crate::optim::phoenix_optimizer::utils::validate_lr(lr)?;
        self.defaults.learning_rate = lr;
        for group in &mut self.parameter_groups {
            group.learning_rate = lr;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Kind, Tensor};

    #[test]
    fn test_adam_creation() {
        let param1 = Tensor::randn(&[2, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let param2 = Tensor::randn(&[3, 1], (Kind::Float, Device::Cpu)).set_requires_grad(true);

        let params = vec![param1.as_mut_ptr(), param2.as_mut_ptr()];
        let optimizer = Adam::new_with_defaults(params).unwrap();

        assert_eq!(optimizer.learning_rate(), 1e-3);
        assert_eq!(optimizer.betas(), (0.9, 0.999));
        assert_eq!(optimizer.eps(), 1e-8);
        assert_eq!(optimizer.weight_decay(), 0.0);
        assert!(!optimizer.amsgrad());
    }

    #[test]
    fn test_adam_custom_config() {
        let param = Tensor::randn(&[2, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let params = vec![param.as_mut_ptr()];

        let config = AdamConfig {
            lr: 1e-2,
            betas: (0.95, 0.999),
            eps: 1e-6,
            weight_decay: 1e-4,
            amsgrad: true,
        };

        let optimizer = Adam::new(params, config).unwrap();

        assert_eq!(optimizer.learning_rate(), 1e-2);
        assert_eq!(optimizer.betas(), (0.95, 0.999));
        assert_eq!(optimizer.eps(), 1e-6);
        assert_eq!(optimizer.weight_decay(), 1e-4);
        assert!(optimizer.amsgrad());
    }

    #[test]
    fn test_adam_with_lr() {
        let param = Tensor::randn(&[2, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let params = vec![param.as_mut_ptr()];

        let optimizer = Adam::with_lr(params, 5e-4).unwrap();
        assert_eq!(optimizer.learning_rate(), 5e-4);
    }

    #[test]
    fn test_adam_parameter_updates() {
        let mut param = Tensor::randn(&[2, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let params = vec![param.as_mut_ptr()];

        let mut optimizer = Adam::with_lr(params, 1e-2).unwrap();

        let original_param = param.shallow_clone();

        let grad = Tensor::ones_like(&param);
        param.set_grad(&grad);

        optimizer.step().unwrap();

        let diff = (&param - &original_param).abs().sum(Kind::Float);
        assert!(f64::try_from(diff).unwrap() > 1e-6);
    }

    #[test]
    fn test_adam_zero_grad() {
        let mut param = Tensor::randn(&[2, 3], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let params = vec![param.as_mut_ptr()];

        let mut optimizer = Adam::new_with_defaults(params).unwrap();

        let grad = Tensor::ones_like(&param);
        param.set_grad(&grad);
        assert!(param.grad().is_some());

        optimizer.zero_grad();
        assert!(param.grad().is_none());
    }

    #[test]
    fn test_adam_state_dict() {
        let mut param = Tensor::randn(&[2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let params = vec![param.as_mut_ptr()];

        let mut optimizer = Adam::with_lr(params, 1e-2).unwrap();

        let grad = Tensor::ones_like(&param);
        param.set_grad(&grad);
        optimizer.step().unwrap();

        let state_dict = optimizer.state_dict();
        assert!(state_dict.contains_key("lr"));
        assert!(state_dict.contains_key("beta1"));
        assert!(state_dict.contains_key("beta2"));
        assert!(state_dict.contains_key("state.0.step"));
        assert!(state_dict.contains_key("state.0.exp_avg"));
        assert!(state_dict.contains_key("state.0.exp_avg_sq"));
    }

    #[test]
    fn test_adam_load_state_dict() {
        let mut param1 = Tensor::randn(&[2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let mut param2 = Tensor::randn(&[2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);

        let params1 = vec![param1.as_mut_ptr()];
        let params2 = vec![param2.as_mut_ptr()];

        let mut opt1 = Adam::with_lr(params1, 1e-2).unwrap();
        let mut opt2 = Adam::with_lr(params2, 5e-3).unwrap();

        let grad = Tensor::ones_like(&param1);
        param1.set_grad(&grad);
        opt1.step().unwrap();

        let state_dict = opt1.state_dict();
        opt2.load_state_dict(&state_dict).unwrap();

        assert_eq!(opt2.learning_rate(), opt1.learning_rate());
        assert_eq!(opt2.betas(), opt1.betas());
    }

    #[test]
    fn test_adam_amsgrad() {
        let mut param = Tensor::randn(&[2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let params = vec![param.as_mut_ptr()];

        let config = AdamConfig { amsgrad: true, ..Default::default() };
        let mut optimizer = Adam::new(params, config).unwrap();

        let grad = Tensor::ones_like(&param);
        param.set_grad(&grad);
        optimizer.step().unwrap();

        let state_dict = optimizer.state_dict();
        assert!(state_dict.contains_key("state.0.max_exp_avg_sq"));
    }

    #[test]
    fn test_adam_weight_decay() {
        let mut param = Tensor::randn(&[2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let params = vec![param.as_mut_ptr()];

        let config = AdamConfig { weight_decay: 1e-2, ..Default::default() };
        let mut optimizer = Adam::new(params, config).unwrap();

        let original_param = param.shallow_clone();

        param.set_grad(&Tensor::zeros_like(&param));
        optimizer.step().unwrap();

        let diff = (&param - &original_param).abs().sum(Kind::Float);
        assert!(f64::try_from(diff).unwrap() > 1e-6);
    }

    #[test]
    fn test_adam_setters() {
        let param = Tensor::randn(&[2, 2], (Kind::Float, Device::Cpu)).set_requires_grad(true);
        let params = vec![param.as_mut_ptr()];

        let mut optimizer = Adam::new_with_defaults(params).unwrap();

        optimizer.set_learning_rate(5e-4);
        assert_eq!(optimizer.learning_rate(), 5e-4);

        optimizer.set_betas((0.8, 0.99));
        assert_eq!(optimizer.betas(), (0.8, 0.99));

        optimizer.set_eps(1e-7);
        assert_eq!(optimizer.eps(), 1e-7);

        optimizer.set_weight_decay(1e-3);
        assert_eq!(optimizer.weight_decay(), 1e-3);

        optimizer.set_amsgrad(true);
        assert!(optimizer.amsgrad());
    }
}
