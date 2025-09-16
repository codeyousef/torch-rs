//! Multi-head attention mechanism for Transformer models

use crate::nn::{Module, phoenix::{PhoenixModule, PhoenixModuleError, Linear}};
use crate::{Device, Kind, Tensor};
use std::collections::HashMap;

#[derive(Debug)]
pub struct MultiheadAttention {
    pub embed_dim: i64,
    pub num_heads: i64,
    pub dropout: f64,
    pub bias: bool,
    pub add_bias_kv: bool,
    pub add_zero_attn: bool,
    pub kdim: i64,
    pub vdim: i64,
    pub batch_first: bool,

    // Projection layers
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,

    // Optional biases
    pub bias_k: Option<Tensor>,
    pub bias_v: Option<Tensor>,

    training: bool,
}

impl MultiheadAttention {
    pub fn new(embed_dim: i64, num_heads: i64) -> Self {
        Self::builder(embed_dim, num_heads).build()
    }

    pub fn builder(embed_dim: i64, num_heads: i64) -> MultiheadAttentionBuilder {
        MultiheadAttentionBuilder {
            embed_dim,
            num_heads,
            dropout: 0.0,
            bias: true,
            add_bias_kv: false,
            add_zero_attn: false,
            kdim: None,
            vdim: None,
            batch_first: false,
        }
    }

    pub fn forward_with_mask(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        key_padding_mask: Option<&Tensor>,
        need_weights: bool,
        attn_mask: Option<&Tensor>,
    ) -> (Tensor, Option<Tensor>) {
        let batch_size = if self.batch_first {
            query.size()[0]
        } else {
            query.size()[1]
        };
        let target_len = if self.batch_first {
            query.size()[1]
        } else {
            query.size()[0]
        };
        let source_len = if self.batch_first {
            key.size()[1]
        } else {
            key.size()[0]
        };

        // Project Q, K, V
        let q = self.q_proj.forward(query);
        let k = self.k_proj.forward(key);
        let v = self.v_proj.forward(value);

        // Reshape for multi-head attention
        let head_dim = self.embed_dim / self.num_heads;

        let q = if self.batch_first {
            q.reshape(&[batch_size, target_len, self.num_heads, head_dim])
                .transpose(1, 2)
        } else {
            q.reshape(&[target_len, batch_size, self.num_heads, head_dim])
                .permute(&[1, 2, 0, 3])
        };

        let k = if self.batch_first {
            k.reshape(&[batch_size, source_len, self.num_heads, head_dim])
                .transpose(1, 2)
        } else {
            k.reshape(&[source_len, batch_size, self.num_heads, head_dim])
                .permute(&[1, 2, 0, 3])
        };

        let v = if self.batch_first {
            v.reshape(&[batch_size, source_len, self.num_heads, head_dim])
                .transpose(1, 2)
        } else {
            v.reshape(&[source_len, batch_size, self.num_heads, head_dim])
                .permute(&[1, 2, 0, 3])
        };

        // Scaled dot-product attention
        let scaling = (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)) / scaling;

        // Apply attention mask if provided
        let scores = if let Some(mask) = attn_mask {
            scores.masked_fill(&mask.logical_not(), f64::NEG_INFINITY)
        } else {
            scores
        };

        // Apply key padding mask if provided
        let scores = if let Some(mask) = key_padding_mask {
            let mask = mask.unsqueeze(1).unsqueeze(2);
            scores.masked_fill(&mask, f64::NEG_INFINITY)
        } else {
            scores
        };

        // Compute attention weights
        let attn_weights = scores.softmax(-1, Kind::Float);
        let attn_weights = if self.training && self.dropout > 0.0 {
            attn_weights.dropout(self.dropout, self.training)
        } else {
            attn_weights
        };

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v);

        // Reshape back
        let attn_output = if self.batch_first {
            attn_output.transpose(1, 2)
                .reshape(&[batch_size, target_len, self.embed_dim])
        } else {
            attn_output.permute(&[2, 0, 1, 3])
                .reshape(&[target_len, batch_size, self.embed_dim])
        };

        // Final projection
        let output = self.out_proj.forward(&attn_output);

        if need_weights {
            // Average attention weights across heads
            let avg_weights = attn_weights.mean_dim(&[1], false, Kind::Float);
            (output, Some(avg_weights))
        } else {
            (output, None)
        }
    }
}

impl Module for MultiheadAttention {
    fn forward(&self, xs: &Tensor) -> Tensor {
        // Self-attention: query = key = value
        let (output, _) = self.forward_with_mask(xs, xs, xs, None, false, None);
        output
    }
}

impl PhoenixModule for MultiheadAttention {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        if let Some(ref bias_k) = self.bias_k {
            params.push(bias_k);
        }
        if let Some(ref bias_v) = self.bias_v {
            params.push(bias_v);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters_mut());
        params.extend(self.k_proj.parameters_mut());
        params.extend(self.v_proj.parameters_mut());
        params.extend(self.out_proj.parameters_mut());
        if let Some(ref mut bias_k) = self.bias_k {
            params.push(bias_k);
        }
        if let Some(ref mut bias_v) = self.bias_v {
            params.push(bias_v);
        }
        params
    }

    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel() as usize).sum()
    }

    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
        self.q_proj.to_device(device)?;
        self.k_proj.to_device(device)?;
        self.v_proj.to_device(device)?;
        self.out_proj.to_device(device)?;
        if let Some(ref mut bias_k) = self.bias_k {
            *bias_k = bias_k.to_device(device);
        }
        if let Some(ref mut bias_v) = self.bias_v {
            *bias_v = bias_v.to_device(device);
        }
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.q_proj.set_training(training);
        self.k_proj.set_training(training);
        self.v_proj.set_training(training);
        self.out_proj.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();

        // Add projection layer weights
        for (name, param) in self.q_proj.state_dict() {
            state.insert(format!("q_proj.{}", name), param);
        }
        for (name, param) in self.k_proj.state_dict() {
            state.insert(format!("k_proj.{}", name), param);
        }
        for (name, param) in self.v_proj.state_dict() {
            state.insert(format!("v_proj.{}", name), param);
        }
        for (name, param) in self.out_proj.state_dict() {
            state.insert(format!("out_proj.{}", name), param);
        }

        if let Some(ref bias_k) = self.bias_k {
            state.insert("bias_k".to_string(), bias_k.shallow_clone());
        }
        if let Some(ref bias_v) = self.bias_v {
            state.insert("bias_v".to_string(), bias_v.shallow_clone());
        }

        state
    }

    fn load_state_dict(&mut self, _state_dict: &HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
        // Implementation would load projection weights and biases
        Ok(())
    }
}

pub struct MultiheadAttentionBuilder {
    embed_dim: i64,
    num_heads: i64,
    dropout: f64,
    bias: bool,
    add_bias_kv: bool,
    add_zero_attn: bool,
    kdim: Option<i64>,
    vdim: Option<i64>,
    batch_first: bool,
}

impl MultiheadAttentionBuilder {
    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn add_bias_kv(mut self, add_bias_kv: bool) -> Self {
        self.add_bias_kv = add_bias_kv;
        self
    }

    pub fn add_zero_attn(mut self, add_zero_attn: bool) -> Self {
        self.add_zero_attn = add_zero_attn;
        self
    }

    pub fn kdim(mut self, kdim: i64) -> Self {
        self.kdim = Some(kdim);
        self
    }

    pub fn vdim(mut self, vdim: i64) -> Self {
        self.vdim = Some(vdim);
        self
    }

    pub fn batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    pub fn build(self) -> MultiheadAttention {
        let kdim = self.kdim.unwrap_or(self.embed_dim);
        let vdim = self.vdim.unwrap_or(self.embed_dim);

        let mut attn = MultiheadAttention {
            embed_dim: self.embed_dim,
            num_heads: self.num_heads,
            dropout: self.dropout,
            bias: self.bias,
            add_bias_kv: self.add_bias_kv,
            add_zero_attn: self.add_zero_attn,
            kdim,
            vdim,
            batch_first: self.batch_first,
            q_proj: Linear::builder(self.embed_dim, self.embed_dim).bias(self.bias).build(),
            k_proj: Linear::builder(kdim, self.embed_dim).bias(self.bias).build(),
            v_proj: Linear::builder(vdim, self.embed_dim).bias(self.bias).build(),
            out_proj: Linear::builder(self.embed_dim, self.embed_dim).bias(self.bias).build(),
            bias_k: None,
            bias_v: None,
            training: true,
        };

        if self.add_bias_kv {
            let head_dim = self.embed_dim / self.num_heads;
            attn.bias_k = Some(Tensor::zeros(&[1, 1, self.embed_dim], (Kind::Float, Device::Cpu)));
            attn.bias_v = Some(Tensor::zeros(&[1, 1, self.embed_dim], (Kind::Float, Device::Cpu)));
        }

        attn
    }
}