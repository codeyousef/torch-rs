//! Transformer encoder layer implementation

use crate::nn::{Module, phoenix::{PhoenixModule, PhoenixModuleError, Linear, Dropout}};
use crate::nn::phoenix_layers::MultiheadAttention;
use crate::{Device, Kind, Tensor};
use std::collections::HashMap;

#[derive(Debug)]
pub struct TransformerEncoderLayer {
    pub d_model: i64,
    pub nhead: i64,
    pub dim_feedforward: i64,
    pub dropout: f64,
    pub activation: Activation,
    pub layer_norm_eps: f64,
    pub batch_first: bool,
    pub norm_first: bool,

    // Sub-layers
    pub self_attn: MultiheadAttention,
    pub linear1: Linear,
    pub linear2: Linear,
    pub dropout1: Dropout,
    pub dropout2: Dropout,
    pub dropout3: Dropout,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,

    training: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    GELU,
    Swish,
}

// Simple LayerNorm implementation
#[derive(Debug)]
pub struct LayerNorm {
    pub normalized_shape: Vec<i64>,
    pub eps: f64,
    pub weight: Tensor,
    pub bias: Tensor,
}

impl LayerNorm {
    pub fn new(normalized_shape: &[i64], eps: f64) -> Self {
        let weight = Tensor::ones(normalized_shape, (Kind::Float, Device::Cpu));
        let bias = Tensor::zeros(normalized_shape, (Kind::Float, Device::Cpu));
        Self {
            normalized_shape: normalized_shape.to_vec(),
            eps,
            weight,
            bias,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.layer_norm(
            &self.normalized_shape,
            Some(&self.weight),
            Some(&self.bias),
            self.eps,
            true,
        )
    }
}

impl PhoenixModule for LayerNorm {
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn num_parameters(&self) -> usize {
        (self.weight.numel() + self.bias.numel()) as usize
    }

    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
        self.weight = self.weight.to_device(device);
        self.bias = self.bias.to_device(device);
        Ok(())
    }

    fn set_training(&mut self, _training: bool) {}

    fn is_training(&self) -> bool {
        false
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        state.insert("weight".to_string(), self.weight.shallow_clone());
        state.insert("bias".to_string(), self.bias.shallow_clone());
        state
    }

    fn load_state_dict(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
        if let Some(weight) = state_dict.get("weight") {
            self.weight = weight.shallow_clone();
        }
        if let Some(bias) = state_dict.get("bias") {
            self.bias = bias.shallow_clone();
        }
        Ok(())
    }
}

impl TransformerEncoderLayer {
    pub fn new(d_model: i64, nhead: i64) -> Self {
        Self::builder(d_model, nhead).build()
    }

    pub fn builder(d_model: i64, nhead: i64) -> TransformerEncoderLayerBuilder {
        TransformerEncoderLayerBuilder {
            d_model,
            nhead,
            dim_feedforward: 2048,
            dropout: 0.1,
            activation: Activation::ReLU,
            layer_norm_eps: 1e-5,
            batch_first: false,
            norm_first: false,
        }
    }

    pub fn forward_with_mask(
        &self,
        src: &Tensor,
        src_mask: Option<&Tensor>,
        src_key_padding_mask: Option<&Tensor>,
    ) -> Tensor {
        let x = if self.norm_first {
            let x = self.norm1.forward(src);
            let (attn_out, _) = self.self_attn.forward_with_mask(
                &x, &x, &x,
                src_key_padding_mask,
                false,
                src_mask,
            );
            src + self.dropout1.forward(&attn_out)
        } else {
            let (attn_out, _) = self.self_attn.forward_with_mask(
                src, src, src,
                src_key_padding_mask,
                false,
                src_mask,
            );
            let x = src + self.dropout1.forward(&attn_out);
            self.norm1.forward(&x)
        };

        if self.norm_first {
            let x2 = self.norm2.forward(&x);
            let x2 = self.linear1.forward(&x2);
            let x2 = self.apply_activation(&x2);
            let x2 = self.dropout2.forward(&x2);
            let x2 = self.linear2.forward(&x2);
            x + self.dropout3.forward(&x2)
        } else {
            let x2 = self.linear1.forward(&x);
            let x2 = self.apply_activation(&x2);
            let x2 = self.dropout2.forward(&x2);
            let x2 = self.linear2.forward(&x2);
            let x2 = x + self.dropout3.forward(&x2);
            self.norm2.forward(&x2)
        }
    }

    fn apply_activation(&self, x: &Tensor) -> Tensor {
        match self.activation {
            Activation::ReLU => x.relu(),
            Activation::GELU => x.gelu("none"),
            Activation::Swish => x * x.sigmoid(),
        }
    }
}

impl Module for TransformerEncoderLayer {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward_with_mask(xs, None, None)
    }
}

impl PhoenixModule for TransformerEncoderLayer {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.linear1.parameters_mut());
        params.extend(self.linear2.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params
    }

    fn num_parameters(&self) -> usize {
        self.self_attn.num_parameters() +
        self.linear1.num_parameters() +
        self.linear2.num_parameters() +
        self.norm1.num_parameters() +
        self.norm2.num_parameters()
    }

    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
        self.self_attn.to_device(device)?;
        self.linear1.to_device(device)?;
        self.linear2.to_device(device)?;
        self.norm1.to_device(device)?;
        self.norm2.to_device(device)?;
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.self_attn.set_training(training);
        self.linear1.set_training(training);
        self.linear2.set_training(training);
        self.dropout1.set_training(training);
        self.dropout2.set_training(training);
        self.dropout3.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();

        for (name, param) in self.self_attn.state_dict() {
            state.insert(format!("self_attn.{}", name), param);
        }
        for (name, param) in self.linear1.state_dict() {
            state.insert(format!("linear1.{}", name), param);
        }
        for (name, param) in self.linear2.state_dict() {
            state.insert(format!("linear2.{}", name), param);
        }
        for (name, param) in self.norm1.state_dict() {
            state.insert(format!("norm1.{}", name), param);
        }
        for (name, param) in self.norm2.state_dict() {
            state.insert(format!("norm2.{}", name), param);
        }

        state
    }

    fn load_state_dict(&mut self, _state_dict: &HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
        Ok(())
    }
}

pub struct TransformerEncoderLayerBuilder {
    d_model: i64,
    nhead: i64,
    dim_feedforward: i64,
    dropout: f64,
    activation: Activation,
    layer_norm_eps: f64,
    batch_first: bool,
    norm_first: bool,
}

impl TransformerEncoderLayerBuilder {
    pub fn dim_feedforward(mut self, dim: i64) -> Self {
        self.dim_feedforward = dim;
        self
    }

    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    pub fn layer_norm_eps(mut self, eps: f64) -> Self {
        self.layer_norm_eps = eps;
        self
    }

    pub fn batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    pub fn norm_first(mut self, norm_first: bool) -> Self {
        self.norm_first = norm_first;
        self
    }

    pub fn build(self) -> TransformerEncoderLayer {
        TransformerEncoderLayer {
            d_model: self.d_model,
            nhead: self.nhead,
            dim_feedforward: self.dim_feedforward,
            dropout: self.dropout,
            activation: self.activation,
            layer_norm_eps: self.layer_norm_eps,
            batch_first: self.batch_first,
            norm_first: self.norm_first,
            self_attn: MultiheadAttention::builder(self.d_model, self.nhead)
                .dropout(self.dropout)
                .batch_first(self.batch_first)
                .build(),
            linear1: Linear::new(self.d_model, self.dim_feedforward),
            linear2: Linear::new(self.dim_feedforward, self.d_model),
            dropout1: Dropout::new(self.dropout),
            dropout2: Dropout::new(self.dropout),
            dropout3: Dropout::new(self.dropout),
            norm1: LayerNorm::new(&[self.d_model], self.layer_norm_eps),
            norm2: LayerNorm::new(&[self.d_model], self.layer_norm_eps),
            training: true,
        }
    }
}