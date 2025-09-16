//! AdamW optimizer implementation with decoupled weight decay
//!
//! AdamW differs from Adam by decoupling weight decay from gradient-based updates

use super::phoenix_optimizer::{PhoenixOptimizer, OptimizerError, ParameterGroup};
use crate::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub lr: f64,
    pub betas: (f64, f64),
    pub eps: f64,
    pub weight_decay: f64,
    pub amsgrad: bool,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01, // Higher default than Adam
            amsgrad: false,
        }
    }
}

#[derive(Debug)]
pub struct AdamW {
    parameter_groups: Vec<ParameterGroup>,
    state: HashMap<usize, AdamWState>,
    defaults: AdamWConfig,
}

#[derive(Debug, Clone)]
struct AdamWState {
    step: i64,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    max_exp_avg_sq: Option<Tensor>,
}

impl AdamW {
    pub fn new<I>(params: I, config: AdamWConfig) -> Result<Self, OptimizerError>
    where
        I: IntoIterator<Item = *mut Tensor>,
    {
        let parameter_groups = vec![ParameterGroup::new(params.into_iter().collect())];

        Ok(Self {
            parameter_groups,
            state: HashMap::new(),
            defaults: config,
        })
    }

    pub fn new_with_defaults<I>(params: I) -> Result<Self, OptimizerError>
    where
        I: IntoIterator<Item = *mut Tensor>,
    {
        Self::new(params, AdamWConfig::default())
    }

    pub fn with_lr<I>(params: I, lr: f64) -> Result<Self, OptimizerError>
    where
        I: IntoIterator<Item = *mut Tensor>,
    {
        Self::new(params, AdamWConfig { lr, ..Default::default() })
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

    pub fn weight_decay(&self) -> f64 {
        self.defaults.weight_decay
    }

    pub fn set_weight_decay(&mut self, weight_decay: f64) {
        self.defaults.weight_decay = weight_decay;
        for group in &mut self.parameter_groups {
            group.set_option("weight_decay", weight_decay.into());
        }
    }

    fn get_lr(&self, group: &ParameterGroup) -> f64 {
        group.get_option("lr").unwrap_or(self.defaults.lr.into()).into()
    }

    fn get_weight_decay(&self, group: &ParameterGroup) -> f64 {
        group.get_option("weight_decay").unwrap_or(self.defaults.weight_decay.into()).into()
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

    fn get_amsgrad(&self, group: &ParameterGroup) -> bool {
        let value: f64 = group.get_option("amsgrad")
            .unwrap_or((if self.defaults.amsgrad { 1.0 } else { 0.0 }).into())
            .into();
        value > 0.5
    }
}

impl PhoenixOptimizer for AdamW {
    fn step(&mut self) -> Result<(), OptimizerError> {
        for group in &mut self.parameter_groups {
            let lr = self.get_lr(group);
            let weight_decay = self.get_weight_decay(group);
            let beta1 = self.get_beta1(group);
            let beta2 = self.get_beta2(group);
            let eps = self.get_eps(group);
            let amsgrad = self.get_amsgrad(group);

            for (param_id, param_ptr) in group.parameters().iter().enumerate() {
                let param = unsafe { &mut **param_ptr };

                if let Some(grad) = param.grad() {
                    let state = self.state.entry(param_id).or_insert_with(|| {
                        AdamWState {
                            step: 0,
                            exp_avg: Tensor::zeros_like(&param),
                            exp_avg_sq: Tensor::zeros_like(&param),
                            max_exp_avg_sq: if amsgrad {
                                Some(Tensor::zeros_like(&param))
                            } else {
                                None
                            },
                        }
                    });

                    state.step += 1;

                    // Exponential moving average of gradient values
                    state.exp_avg = &state.exp_avg * beta1 + &grad * (1.0 - beta1);

                    // Exponential moving average of squared gradient values
                    state.exp_avg_sq = &state.exp_avg_sq * beta2 + (&grad * &grad) * (1.0 - beta2);

                    // Bias correction
                    let bias_correction1 = 1.0 - beta1.powi(state.step as i32);
                    let bias_correction2 = 1.0 - beta2.powi(state.step as i32);

                    // Corrected exponential moving average
                    let corrected_exp_avg = &state.exp_avg / bias_correction1;

                    // Denominator for update
                    let denom = if amsgrad {
                        let max_exp_avg_sq = state.max_exp_avg_sq.as_mut().unwrap();
                        *max_exp_avg_sq = Tensor::max_tensor(max_exp_avg_sq, &state.exp_avg_sq);
                        (max_exp_avg_sq / bias_correction2).sqrt() + eps
                    } else {
                        (&state.exp_avg_sq / bias_correction2).sqrt() + eps
                    };

                    // Update parameters
                    let update = &corrected_exp_avg / denom * lr;
                    let _ = param.sub_(&update);

                    // AdamW: Decoupled weight decay (applied directly to parameters)
                    if weight_decay > 0.0 {
                        let _ = param.mul_(1.0 - lr * weight_decay);
                    }
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

    fn state_dict(&self) -> HashMap<String, crate::nn::optimizer::OptimizerValue> {
        use crate::nn::optimizer::OptimizerValue;

        let mut state_dict = HashMap::new();

        state_dict.insert("lr".to_string(), OptimizerValue::Float(self.defaults.lr));
        state_dict.insert("beta1".to_string(), OptimizerValue::Float(self.defaults.betas.0));
        state_dict.insert("beta2".to_string(), OptimizerValue::Float(self.defaults.betas.1));
        state_dict.insert("eps".to_string(), OptimizerValue::Float(self.defaults.eps));
        state_dict.insert("weight_decay".to_string(), OptimizerValue::Float(self.defaults.weight_decay));
        state_dict.insert("amsgrad".to_string(), OptimizerValue::Bool(self.defaults.amsgrad));

        for (param_id, state) in &self.state {
            let prefix = format!("state.{}", param_id);
            state_dict.insert(format!("{}.step", prefix), OptimizerValue::Int(state.step));
            state_dict.insert(format!("{}.exp_avg", prefix),
                OptimizerValue::Tensor(state.exp_avg.shallow_clone()));
            state_dict.insert(format!("{}.exp_avg_sq", prefix),
                OptimizerValue::Tensor(state.exp_avg_sq.shallow_clone()));

            if let Some(ref max_exp_avg_sq) = state.max_exp_avg_sq {
                state_dict.insert(format!("{}.max_exp_avg_sq", prefix),
                    OptimizerValue::Tensor(max_exp_avg_sq.shallow_clone()));
            }
        }

        state_dict
    }

    fn load_state_dict(&mut self, state_dict: &HashMap<String, crate::nn::optimizer::OptimizerValue>)
        -> Result<(), OptimizerError> {
        use crate::nn::optimizer::OptimizerValue;

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

        // Load state for each parameter
        self.state.clear();
        let mut param_states: HashMap<usize, AdamWState> = HashMap::new();

        for (key, value) in state_dict {
            if let Some(state_key) = key.strip_prefix("state.") {
                if let Some(dot_pos) = state_key.find('.') {
                    let param_id: usize = state_key[..dot_pos].parse()
                        .map_err(|_| OptimizerError::InvalidStateDict("Invalid parameter ID".to_string()))?;
                    let field = &state_key[dot_pos + 1..];

                    let state = param_states.entry(param_id).or_insert_with(|| {
                        AdamWState {
                            step: 0,
                            exp_avg: Tensor::new(),
                            exp_avg_sq: Tensor::new(),
                            max_exp_avg_sq: if self.defaults.amsgrad { Some(Tensor::new()) } else { None },
                        }
                    });

                    match (field, value) {
                        ("step", OptimizerValue::Int(step)) => state.step = *step,
                        ("exp_avg", OptimizerValue::Tensor(tensor)) => {
                            state.exp_avg = tensor.shallow_clone();
                        }
                        ("exp_avg_sq", OptimizerValue::Tensor(tensor)) => {
                            state.exp_avg_sq = tensor.shallow_clone();
                        }
                        ("max_exp_avg_sq", OptimizerValue::Tensor(tensor)) => {
                            state.max_exp_avg_sq = Some(tensor.shallow_clone());
                        }
                        _ => {}
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
}