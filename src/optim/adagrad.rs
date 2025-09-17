//! Adagrad optimizer implementation
//!
//! Adagrad adapts the learning rate per parameter using accumulated squared gradients

use crate::optim::optimizer::{PhoenixOptimizer, OptimizerError, ParameterGroup};
use crate::Tensor;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AdagradConfig {
    pub lr: f64,
    pub lr_decay: f64,
    pub weight_decay: f64,
    pub initial_accumulator_value: f64,
    pub eps: f64,
}

impl Default for AdagradConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
        }
    }
}

#[derive(Debug)]
pub struct Adagrad {
    parameter_groups: Vec<ParameterGroup>,
    state: HashMap<usize, AdagradState>,
    defaults: AdagradConfig,
}

#[derive(Debug, Clone)]
struct AdagradState {
    step: i64,
    sum: Tensor,
}

impl Adagrad {
    pub fn new<I>(params: I, config: AdagradConfig) -> Result<Self, OptimizerError>
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
        Self::new(params, AdagradConfig::default())
    }

    pub fn with_lr<I>(params: I, lr: f64) -> Result<Self, OptimizerError>
    where
        I: IntoIterator<Item = *mut Tensor>,
    {
        Self::new(params, AdagradConfig { lr, ..Default::default() })
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

    fn get_lr_decay(&self, group: &ParameterGroup) -> f64 {
        group.get_option("lr_decay").unwrap_or(self.defaults.lr_decay.into()).into()
    }

    fn get_weight_decay(&self, group: &ParameterGroup) -> f64 {
        group.get_option("weight_decay").unwrap_or(self.defaults.weight_decay.into()).into()
    }

    fn get_initial_accumulator_value(&self, group: &ParameterGroup) -> f64 {
        group.get_option("initial_accumulator_value")
            .unwrap_or(self.defaults.initial_accumulator_value.into()).into()
    }

    fn get_eps(&self, group: &ParameterGroup) -> f64 {
        group.get_option("eps").unwrap_or(self.defaults.eps.into()).into()
    }
}

impl PhoenixOptimizer for Adagrad {
    fn step(&mut self) -> Result<(), OptimizerError> {
        for group in &mut self.parameter_groups {
            let lr = self.get_lr(group);
            let lr_decay = self.get_lr_decay(group);
            let weight_decay = self.get_weight_decay(group);
            let initial_accumulator_value = self.get_initial_accumulator_value(group);
            let eps = self.get_eps(group);

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
                        let sum = if initial_accumulator_value != 0.0 {
                            Tensor::full_like(&param, initial_accumulator_value)
                        } else {
                            Tensor::zeros_like(&param)
                        };
                        AdagradState {
                            step: 0,
                            sum,
                        }
                    });

                    state.step += 1;

                    // Accumulate squared gradients
                    state.sum = &state.sum + (&grad * &grad);

                    // Apply learning rate decay
                    let clr = lr / (1.0 + (state.step as f64 - 1.0) * lr_decay);

                    // Compute denominator
                    let std = state.sum.sqrt() + eps;

                    // Update parameters
                    let _ = param.sub_(&grad / &std * clr);
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
        state_dict.insert("lr_decay".to_string(), OptimizerValue::Float(self.defaults.lr_decay));
        state_dict.insert("weight_decay".to_string(), OptimizerValue::Float(self.defaults.weight_decay));
        state_dict.insert("initial_accumulator_value".to_string(), 
            OptimizerValue::Float(self.defaults.initial_accumulator_value));
        state_dict.insert("eps".to_string(), OptimizerValue::Float(self.defaults.eps));

        for (param_id, state) in &self.state {
            let prefix = format!("state.{}", param_id);
            state_dict.insert(format!("{}.step", prefix), OptimizerValue::Int(state.step));
            state_dict.insert(format!("{}.sum", prefix),
                OptimizerValue::Tensor(state.sum.shallow_clone()));
        }

        state_dict
    }

    fn load_state_dict(&mut self, state_dict: &HashMap<String, crate::nn::optimizer::OptimizerValue>)
        -> Result<(), OptimizerError> {
        use crate::nn::optimizer::OptimizerValue;

        if let Some(OptimizerValue::Float(lr)) = state_dict.get("lr") {
            self.defaults.lr = *lr;
        }
        if let Some(OptimizerValue::Float(lr_decay)) = state_dict.get("lr_decay") {
            self.defaults.lr_decay = *lr_decay;
        }
        if let Some(OptimizerValue::Float(weight_decay)) = state_dict.get("weight_decay") {
            self.defaults.weight_decay = *weight_decay;
        }
        if let Some(OptimizerValue::Float(initial_accumulator_value)) = state_dict.get("initial_accumulator_value") {
            self.defaults.initial_accumulator_value = *initial_accumulator_value;
        }
        if let Some(OptimizerValue::Float(eps)) = state_dict.get("eps") {
            self.defaults.eps = *eps;
        }

        // Load state for each parameter
        self.state.clear();
        let mut param_states: HashMap<usize, AdagradState> = HashMap::new();

        for (key, value) in state_dict {
            if let Some(state_key) = key.strip_prefix("state.") {
                if let Some(dot_pos) = state_key.find('.') {
                    let param_id: usize = state_key[..dot_pos].parse()
                        .map_err(|_| OptimizerError::InvalidStateDict("Invalid parameter ID".to_string()))?;
                    let field = &state_key[dot_pos + 1..];

                    let state = param_states.entry(param_id).or_insert_with(|| {
                        AdagradState {
                            step: 0,
                            sum: Tensor::new(),
                        }
                    });

                    match (field, value) {
                        ("step", OptimizerValue::Int(step)) => state.step = *step,
                        ("sum", OptimizerValue::Tensor(tensor)) => {
                            state.sum = tensor.shallow_clone();
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