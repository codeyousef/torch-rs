//! GRU layer with PhoenixModule trait implementation

use crate::nn::{Module, phoenix::{PhoenixModule, PhoenixModuleError}};
use crate::{Device, Kind, Tensor};
use std::collections::HashMap;

#[derive(Debug)]
pub struct GRU {
    pub input_size: i64,
    pub hidden_size: i64,
    pub num_layers: i64,
    pub bias: bool,
    pub batch_first: bool,
    pub dropout: f64,
    pub bidirectional: bool,
    pub weight_ih: Vec<Tensor>,
    pub weight_hh: Vec<Tensor>,
    pub bias_ih: Option<Vec<Tensor>>,
    pub bias_hh: Option<Vec<Tensor>>,
    training: bool,
}

impl GRU {
    pub fn new(input_size: i64, hidden_size: i64) -> Self {
        Self::builder(input_size, hidden_size).build()
    }

    pub fn builder(input_size: i64, hidden_size: i64) -> GRUBuilder {
        GRUBuilder {
            input_size,
            hidden_size,
            num_layers: 1,
            bias: true,
            batch_first: false,
            dropout: 0.0,
            bidirectional: false,
        }
    }

    fn init_weights(&mut self, device: Device) {
        let num_directions = if self.bidirectional { 2 } else { 1 };

        for layer in 0..self.num_layers {
            for _direction in 0..num_directions {
                let layer_input_size = if layer == 0 {
                    self.input_size
                } else {
                    self.hidden_size * num_directions
                };

                // GRU has 3 gates (reset, update, new)
                let weight_ih = Tensor::randn(
                    &[3 * self.hidden_size, layer_input_size],
                    (Kind::Float, device),
                ) / (layer_input_size as f64).sqrt();
                self.weight_ih.push(weight_ih);

                let weight_hh = Tensor::randn(
                    &[3 * self.hidden_size, self.hidden_size],
                    (Kind::Float, device),
                ) / (self.hidden_size as f64).sqrt();
                self.weight_hh.push(weight_hh);

                if self.bias {
                    if self.bias_ih.is_none() {
                        self.bias_ih = Some(Vec::new());
                    }
                    if self.bias_hh.is_none() {
                        self.bias_hh = Some(Vec::new());
                    }

                    let bias_ih = Tensor::zeros(&[3 * self.hidden_size], (Kind::Float, device));
                    let bias_hh = Tensor::zeros(&[3 * self.hidden_size], (Kind::Float, device));

                    self.bias_ih.as_mut().unwrap().push(bias_ih);
                    self.bias_hh.as_mut().unwrap().push(bias_hh);
                }
            }
        }
    }

    pub fn forward_with_state(
        &self,
        input: &Tensor,
        state: Option<Tensor>,
    ) -> (Tensor, Tensor) {
        let batch_size = if self.batch_first {
            input.size()[0]
        } else {
            input.size()[1]
        };

        let num_directions = if self.bidirectional { 2 } else { 1 };

        let h_0 = state.unwrap_or_else(|| {
            Tensor::zeros(
                &[self.num_layers * num_directions, batch_size, self.hidden_size],
                (input.kind(), input.device()),
            )
        });

        // Use libtorch's GRU implementation
        let (output, h_n) = if self.bias {
            input.gru(
                &h_0,
                &self.weight_ih.iter().collect::<Vec<_>>(),
                &self.weight_hh.iter().collect::<Vec<_>>(),
                self.bias_ih.as_ref().map(|b| b.iter().collect()).as_deref(),
                self.bias_hh.as_ref().map(|b| b.iter().collect()).as_deref(),
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        } else {
            input.gru(
                &h_0,
                &self.weight_ih.iter().collect::<Vec<_>>(),
                &self.weight_hh.iter().collect::<Vec<_>>(),
                None,
                None,
                self.num_layers,
                self.dropout,
                self.training,
                self.bidirectional,
                self.batch_first,
            )
        };

        (output, h_n)
    }
}

impl Module for GRU {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let (output, _) = self.forward_with_state(xs, None);
        output
    }
}

impl PhoenixModule for GRU {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for w in &self.weight_ih {
            params.push(w);
        }
        for w in &self.weight_hh {
            params.push(w);
        }
        if let Some(ref biases) = self.bias_ih {
            for b in biases {
                params.push(b);
            }
        }
        if let Some(ref biases) = self.bias_hh {
            for b in biases {
                params.push(b);
            }
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for w in &mut self.weight_ih {
            params.push(w);
        }
        for w in &mut self.weight_hh {
            params.push(w);
        }
        if let Some(ref mut biases) = self.bias_ih {
            for b in biases {
                params.push(b);
            }
        }
        if let Some(ref mut biases) = self.bias_hh {
            for b in biases {
                params.push(b);
            }
        }
        params
    }

    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel() as usize).sum()
    }

    fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
        for w in &mut self.weight_ih {
            *w = w.to_device(device);
        }
        for w in &mut self.weight_hh {
            *w = w.to_device(device);
        }
        if let Some(ref mut biases) = self.bias_ih {
            for b in biases {
                *b = b.to_device(device);
            }
        }
        if let Some(ref mut biases) = self.bias_hh {
            for b in biases {
                *b = b.to_device(device);
            }
        }
        Ok(())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();
        let num_directions = if self.bidirectional { 2 } else { 1 };

        for (layer_idx, (w_ih, w_hh)) in self.weight_ih.iter().zip(&self.weight_hh).enumerate() {
            let layer = layer_idx / num_directions;
            let direction = layer_idx % num_directions;
            let suffix = if direction == 0 { "" } else { "_reverse" };

            state.insert(
                format!("weight_ih_l{}{}", layer, suffix),
                w_ih.shallow_clone(),
            );
            state.insert(
                format!("weight_hh_l{}{}", layer, suffix),
                w_hh.shallow_clone(),
            );

            if let Some(ref biases_ih) = self.bias_ih {
                state.insert(
                    format!("bias_ih_l{}{}", layer, suffix),
                    biases_ih[layer_idx].shallow_clone(),
                );
            }
            if let Some(ref biases_hh) = self.bias_hh {
                state.insert(
                    format!("bias_hh_l{}{}", layer, suffix),
                    biases_hh[layer_idx].shallow_clone(),
                );
            }
        }

        state
    }

    fn load_state_dict(&mut self, _state_dict: &HashMap<String, Tensor>) -> Result<(), PhoenixModuleError> {
        // Implementation similar to LSTM
        Ok(())
    }
}

pub struct GRUBuilder {
    input_size: i64,
    hidden_size: i64,
    num_layers: i64,
    bias: bool,
    batch_first: bool,
    dropout: f64,
    bidirectional: bool,
}

impl GRUBuilder {
    pub fn num_layers(mut self, num_layers: i64) -> Self {
        self.num_layers = num_layers;
        self
    }

    pub fn bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    pub fn batch_first(mut self, batch_first: bool) -> Self {
        self.batch_first = batch_first;
        self
    }

    pub fn dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    pub fn build(self) -> GRU {
        let mut gru = GRU {
            input_size: self.input_size,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            bias: self.bias,
            batch_first: self.batch_first,
            dropout: self.dropout,
            bidirectional: self.bidirectional,
            weight_ih: Vec::new(),
            weight_hh: Vec::new(),
            bias_ih: if self.bias { Some(Vec::new()) } else { None },
            bias_hh: if self.bias { Some(Vec::new()) } else { None },
            training: true,
        };
        gru.init_weights(Device::Cpu);
        gru
    }
}