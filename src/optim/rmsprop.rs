//! RMSprop optimizer implementation
//!
//! RMSprop uses an exponential moving average of squared gradients to normalize gradients

use super::phoenix_optimizer::{PhoenixOptimizer, OptimizerError, ParameterGroup};
use crate::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RMSpropConfig {
    pub lr: f64,
    pub alpha: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub momentum: f64,
    pub centered: bool,
}

impl Default for RMSpropConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
        }
    }
}

#[derive(Debug)]
pub struct RMSprop {
    parameter_groups: Vec<ParameterGroup>,
    state: HashMap<usize, RMSpropState>,
    defaults: RMSpropConfig,
}

#[derive(Debug, Clone)]
struct RMSpropState {
    step: i64,
    square_avg: Tensor,
    momentum_buffer: Option<Tensor>,
    grad_avg: Option<Tensor>,
}

impl RMSprop {
    pub fn new<I>(params: I, config: RMSpropConfig) -> Result<Self, OptimizerError>
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
        Self::new(params, RMSpropConfig::default())
    }

    pub fn with_lr<I>(params: I, lr: f64) -> Result<Self, OptimizerError>
    where
        I: IntoIterator<Item = *mut Tensor>,
    {
        Self::new(params, RMSpropConfig { lr, ..Default::default() })
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

    fn get_lr(&self, group: &ParameterGroup) -> f64 {
        group.get_option("lr").unwrap_or(self.defaults.lr.into()).into()
    }

    fn get_alpha(&self, group: &ParameterGroup) -> f64 {
        group.get_option("alpha").unwrap_or(self.defaults.alpha.into()).into()
    }

    fn get_eps(&self, group: &ParameterGroup) -> f64 {
        group.get_option("eps").unwrap_or(self.defaults.eps.into()).into()
    }

    fn get_weight_decay(&self, group: &ParameterGroup) -> f64 {
        group.get_option("weight_decay").unwrap_or(self.defaults.weight_decay.into()).into()
    }

    fn get_momentum(&self, group: &ParameterGroup) -> f64 {
        group.get_option("momentum").unwrap_or(self.defaults.momentum.into()).into()
    }

    fn get_centered(&self, group: &ParameterGroup) -> bool {
        let value: f64 = group.get_option("centered")
            .unwrap_or((if self.defaults.centered { 1.0 } else { 0.0 }).into())
            .into();
        value > 0.5
    }
}

impl PhoenixOptimizer for RMSprop {
    fn step(&mut self) -> Result<(), OptimizerError> {
        for group in &mut self.parameter_groups {
            let lr = self.get_lr(group);
            let alpha = self.get_alpha(group);
            let eps = self.get_eps(group);
            let weight_decay = self.get_weight_decay(group);
            let momentum = self.get_momentum(group);
            let centered = self.get_centered(group);

            for (param_id, param_ptr) in group.parameters().iter().enumerate() {
                let param = unsafe { &mut **param_ptr };

                if let Some(grad) = param.grad() {
                    // Apply weight decay
                    let grad = if weight_decay > 0.0 {
                        &grad + param * weight_decay
                    } else {
                        grad.shallow_clone()
                    };

                    let state = self.state.entry(param_id).or_insert_with(|| {
                        RMSpropState {
                            step: 0,
                            square_avg: Tensor::zeros_like(&param),
                            momentum_buffer: if momentum > 0.0 {
                                Some(Tensor::zeros_like(&param))
                            } else {
                                None
                            },
                            grad_avg: if centered {
                                Some(Tensor::zeros_like(&param))
                            } else {
                                None
                            },
                        }
                    });

                    state.step += 1;

                    // Update square average
                    state.square_avg = &state.square_avg * alpha + (&grad * &grad) * (1.0 - alpha);

                    let avg = if centered {
                        // Update grad average for centered RMSprop
                        let grad_avg = state.grad_avg.as_mut().unwrap();
                        *grad_avg = grad_avg * alpha + &grad * (1.0 - alpha);

                        // Compute centered variance
                        &state.square_avg - (grad_avg * grad_avg) + eps
                    } else {
                        state.square_avg.shallow_clone() + eps
                    };

                    let avg = avg.sqrt();

                    if momentum > 0.0 {
                        let buf = state.momentum_buffer.as_mut().unwrap();
                        *buf = buf * momentum + &grad / &avg;
                        let _ = param.sub_(buf * lr);
                    } else {
                        let _ = param.sub_(&grad / &avg * lr);
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
        state_dict.insert("alpha".to_string(), OptimizerValue::Float(self.defaults.alpha));
        state_dict.insert("eps".to_string(), OptimizerValue::Float(self.defaults.eps));
        state_dict.insert("weight_decay".to_string(), OptimizerValue::Float(self.defaults.weight_decay));
        state_dict.insert("momentum".to_string(), OptimizerValue::Float(self.defaults.momentum));
        state_dict.insert("centered".to_string(), OptimizerValue::Bool(self.defaults.centered));

        for (param_id, state) in &self.state {
            let prefix = format!("state.{}", param_id);
            state_dict.insert(format!("{}.step", prefix), OptimizerValue::Int(state.step));
            state_dict.insert(format!("{}.square_avg", prefix),
                OptimizerValue::Tensor(state.square_avg.shallow_clone()));

            if let Some(ref momentum_buffer) = state.momentum_buffer {
                state_dict.insert(format!("{}.momentum_buffer", prefix),
                    OptimizerValue::Tensor(momentum_buffer.shallow_clone()));
            }

            if let Some(ref grad_avg) = state.grad_avg {
                state_dict.insert(format!("{}.grad_avg", prefix),
                    OptimizerValue::Tensor(grad_avg.shallow_clone()));
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
        if let Some(OptimizerValue::Float(alpha)) = state_dict.get("alpha") {
            self.defaults.alpha = *alpha;
        }
        if let Some(OptimizerValue::Float(eps)) = state_dict.get("eps") {
            self.defaults.eps = *eps;
        }
        if let Some(OptimizerValue::Float(weight_decay)) = state_dict.get("weight_decay") {
            self.defaults.weight_decay = *weight_decay;
        }
        if let Some(OptimizerValue::Float(momentum)) = state_dict.get("momentum") {
            self.defaults.momentum = *momentum;
        }
        if let Some(OptimizerValue::Bool(centered)) = state_dict.get("centered") {
            self.defaults.centered = *centered;
        }

        // Load state for each parameter
        self.state.clear();
        let mut param_states: HashMap<usize, RMSpropState> = HashMap::new();

        for (key, value) in state_dict {
            if let Some(state_key) = key.strip_prefix("state.") {
                if let Some(dot_pos) = state_key.find('.') {
                    let param_id: usize = state_key[..dot_pos].parse()
                        .map_err(|_| OptimizerError::InvalidStateDict("Invalid parameter ID".to_string()))?;
                    let field = &state_key[dot_pos + 1..];

                    let state = param_states.entry(param_id).or_insert_with(|| {
                        RMSpropState {
                            step: 0,
                            square_avg: Tensor::new(),
                            momentum_buffer: if self.defaults.momentum > 0.0 { Some(Tensor::new()) } else { None },
                            grad_avg: if self.defaults.centered { Some(Tensor::new()) } else { None },
                        }
                    });

                    match (field, value) {
                        ("step", OptimizerValue::Int(step)) => state.step = *step,
                        ("square_avg", OptimizerValue::Tensor(tensor)) => {
                            state.square_avg = tensor.shallow_clone();
                        }
                        ("momentum_buffer", OptimizerValue::Tensor(tensor)) => {
                            state.momentum_buffer = Some(tensor.shallow_clone());
                        }
                        ("grad_avg", OptimizerValue::Tensor(tensor)) => {
                            state.grad_avg = Some(tensor.shallow_clone());
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