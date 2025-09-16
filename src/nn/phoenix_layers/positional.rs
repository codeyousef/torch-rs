//! Positional encoding for Transformer models

use crate::nn::{Module, phoenix::{PhoenixModule, PhoenixModuleError, Dropout}};
use crate::{Device, Kind, Tensor};
use std::collections::HashMap;

#[derive(Debug)]
pub struct PositionalEncoding {
    pub d_model: i64,
    pub dropout: f64,
    pub max_len: i64,
    pub dropout_layer: Dropout,
    pub pe: Tensor,
    training: bool,
}

impl PositionalEncoding {
    pub fn new(d_model: i64, dropout: f64, max_len: i64) -> Self {
        let mut pe = Tensor::zeros(&[max_len, d_model], (Kind::Float, Device::Cpu));

        let position = Tensor::arange(max_len, (Kind::Float, Device::Cpu))
            .unsqueeze(1);

        let div_term = (Tensor::arange_step(0, d_model, 2, (Kind::Float, Device::Cpu))
            * -(10000_f64.ln() / d_model as f64)).exp();

        // Sin for even indices
        let sin_values = &position * &div_term;
        for i in 0..div_term.size()[0] {
            pe.slice(1, 2 * i, 2 * i + 1, 1)
                .copy_(&sin_values.slice(1, i, i + 1, 1).sin());
        }

        // Cos for odd indices
        let cos_values = &position * &div_term;
        for i in 0..div_term.size()[0] {
            if 2 * i + 1 < d_model {
                pe.slice(1, 2 * i + 1, 2 * i + 2, 1)
                    .copy_(&cos_values.slice(1, i, i + 1, 1).cos());
            }
        }

        // Add batch dimension
        let pe = pe.unsqueeze(0);

        Self {
            d_model,
            dropout,
            max_len,
            dropout_layer: Dropout::new(dropout),
            pe: pe.set_requires_grad(false),
            training: true,
        }
    }

    pub fn builder(d_model: i64) -> PositionalEncodingBuilder {
        PositionalEncodingBuilder {
            d_model,
            dropout: 0.1,
            max_len: 5000,
        }
    }
}

impl Module for PositionalEncoding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // xs shape: [batch_size, seq_len, d_model] or [seq_len, batch_size, d_model]
        let seq_len = xs.size()[xs.dim() - 2];

        let pe_slice = self.pe.slice(1, 0, seq_len, 1);
        let output = xs + pe_slice;

        self.dropout_layer.forward(&output)
    }
}

impl PhoenixModule for PositionalEncoding {
    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No trainable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn buffers(&self) -> Vec<&Tensor> {
        vec![&self.pe]
    }

    fn buffers_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.pe]
    }

    fn num_parameters(&self) -> usize {
        0
    }

    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
        self.pe = self.pe.to_device(device);
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.dropout_layer.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        state.insert("pe".to_string(), self.pe.shallow_clone());
        state
    }

    fn load_state_dict(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
        if let Some(pe) = state_dict.get("pe") {
            self.pe = pe.shallow_clone();
        }
        Ok(())
    }
}

pub struct PositionalEncodingBuilder {
    d_model: i64,
    dropout: f64,
    max_len: i64,
}

impl PositionalEncodingBuilder {
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn max_len(mut self, max_len: i64) -> Self {
        self.max_len = max_len;
        self
    }

    pub fn build(self) -> PositionalEncoding {
        PositionalEncoding::new(self.d_model, self.dropout, self.max_len)
    }
}

/// Sinusoidal positional encoding (learnable version)
#[derive(Debug)]
pub struct LearnedPositionalEncoding {
    pub d_model: i64,
    pub max_len: i64,
    pub dropout: f64,
    pub embeddings: Tensor,
    pub dropout_layer: Dropout,
    training: bool,
}

impl LearnedPositionalEncoding {
    pub fn new(d_model: i64, max_len: i64, dropout: f64) -> Self {
        let embeddings = Tensor::randn(&[max_len, d_model], (Kind::Float, Device::Cpu))
            / (d_model as f64).sqrt();

        Self {
            d_model,
            max_len,
            dropout,
            embeddings,
            dropout_layer: Dropout::new(dropout),
            training: true,
        }
    }
}

impl Module for LearnedPositionalEncoding {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let seq_len = xs.size()[xs.dim() - 2];
        let positions = Tensor::arange(seq_len, (Kind::Int64, xs.device()));

        let pos_embeddings = self.embeddings.index_select(0, &positions);
        let output = xs + pos_embeddings;

        self.dropout_layer.forward(&output)
    }
}

impl PhoenixModule for LearnedPositionalEncoding {
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.embeddings]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.embeddings]
    }

    fn num_parameters(&self) -> usize {
        self.embeddings.numel() as usize
    }

    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
        self.embeddings = self.embeddings.to_device(device);
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.dropout_layer.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        state.insert("embeddings".to_string(), self.embeddings.shallow_clone());
        state
    }

    fn load_state_dict(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
        if let Some(embeddings) = state_dict.get("embeddings") {
            self.embeddings = embeddings.shallow_clone();
        }
        Ok(())
    }
}