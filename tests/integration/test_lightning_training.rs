//! Integration test: Train model with Trainer → validate convergence
#![cfg(feature = "torch-rs")]

use tch::nn::phoenix::{LightningModule, PhoenixModule, Linear};
use tch::training::{Trainer, TrainerConfig, StepOutput, OptimizerConfig};
use tch::data::{Dataset, DataLoader};
use tch::{nn, Device, Kind, Tensor};
use std::collections::HashMap;

// Simple XOR problem for testing convergence
#[derive(Debug)]
struct XORModel {
    fc1: Linear,
    fc2: Linear,
    training: bool,
}

impl XORModel {
    fn new() -> Self {
        Self {
            fc1: Linear::new(2, 4),
            fc2: Linear::new(4, 1),
            training: true,
        }
    }
}

impl nn::Module for XORModel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.fc1).tanh().apply(&self.fc2)
    }
}

impl LightningModule for XORModel {
    fn training_step(&mut self, batch: (Tensor, Tensor), _batch_idx: usize) -> StepOutput {
        let (x, y) = batch;
        let y_hat = self.forward(&x);
        let loss = y_hat.mse_loss(&y, tch::Reduction::Mean);

        StepOutput {
            loss: loss.shallow_clone(),
            logs: {
                let mut logs = HashMap::new();
                logs.insert("train_loss".to_string(), f64::try_from(&loss).unwrap());
                logs
            },
        }
    }

    fn validation_step(&mut self, batch: (Tensor, Tensor), _batch_idx: usize) -> StepOutput {
        let (x, y) = batch;
        let y_hat = self.forward(&x);
        let loss = y_hat.mse_loss(&y, tch::Reduction::Mean);

        // Calculate accuracy for XOR
        let predictions = y_hat.ge(0.0).to_kind(Kind::Float);
        let correct = predictions.eq_tensor(&y).to_kind(Kind::Float).sum(Kind::Float);
        let accuracy = f64::try_from(correct).unwrap() / y.size()[0] as f64;

        StepOutput {
            loss: loss.shallow_clone(),
            logs: {
                let mut logs = HashMap::new();
                logs.insert("val_loss".to_string(), f64::try_from(&loss).unwrap());
                logs.insert("val_accuracy".to_string(), accuracy);
                logs
            },
        }
    }

    fn configure_optimizers(&self) -> OptimizerConfig {
        OptimizerConfig::adam(self.parameters_mut(), 0.01)
    }
}

tch::impl_phoenix_module!(XORModel {
    fc1: Linear,
    fc2: Linear,
});

struct XORDataset;

impl Dataset for XORDataset {
    type Item = (Tensor, Tensor);

    fn len(&self) -> usize {
        4
    }

    fn get(&self, index: usize) -> Result<Self::Item, tch::data::DatasetError> {
        let data = vec![
            (vec![0.0, 0.0], 0.0),
            (vec![0.0, 1.0], 1.0),
            (vec![1.0, 0.0], 1.0),
            (vec![1.0, 1.0], 0.0),
        ];

        if index >= 4 {
            return Err(tch::data::DatasetError::IndexOutOfBounds {
                index,
                size: 4,
            });
        }

        let (input, target) = &data[index];
        let x = Tensor::from_slice(input).to_kind(Kind::Float);
        let y = Tensor::from_slice(&[*target]).to_kind(Kind::Float);

        Ok((x, y))
    }

    fn download(&self) -> Result<(), tch::data::DatasetError> {
        Ok(())
    }

    fn is_downloaded(&self) -> bool {
        true
    }

    fn root(&self) -> &std::path::PathBuf {
        static ROOT: std::path::PathBuf = std::path::PathBuf::new();
        &ROOT
    }
}

#[test]
fn test_training_convergence() {
    let mut model = XORModel::new();
    let dataset = XORDataset;
    let dataloader = DataLoader::with_batch_size(dataset, 4);

    let config = TrainerConfig {
        max_epochs: 100,
        accelerator: tch::training::Accelerator::Cpu,
        ..Default::default()
    };

    let mut trainer = Trainer::new(config);
    let result = trainer.fit(&mut model, dataloader.clone(), Some(dataloader));

    assert!(result.is_ok());

    // Check that model learned XOR
    model.set_training(false);

    let test_cases = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ];

    let mut correct = 0;
    for (input, expected) in test_cases {
        let x = Tensor::from_slice(&input);
        let y_hat = model.forward(&x);
        let prediction = if f64::try_from(y_hat).unwrap() > 0.0 {
            1.0
        } else {
            0.0
        };

        if (prediction - expected).abs() < 0.5 {
            correct += 1;
        }
    }

    assert!(
        correct >= 3,
        "Model should learn XOR pattern (got {}/4 correct)",
        correct
    );
}

#[test]
fn test_validation_metrics() {
    let mut model = XORModel::new();
    let dataset = XORDataset;
    let train_loader = DataLoader::with_batch_size(dataset, 2);
    let val_loader = DataLoader::with_batch_size(XORDataset, 4);

    let config = TrainerConfig {
        max_epochs: 50,
        val_check_interval: 1.0,
        ..Default::default()
    };

    let mut trainer = Trainer::new(config);
    let result = trainer.fit(&mut model, train_loader, Some(val_loader));

    assert!(result.is_ok());

    // Check that validation metrics were logged
    let metrics = trainer.get_metrics();
    assert!(metrics.contains_key("val_loss"));
    assert!(metrics.contains_key("val_accuracy"));

    // Validation accuracy should improve during training
    let final_val_acc = metrics["val_accuracy"];
    assert!(final_val_acc > 0.5, "Validation accuracy should be > 50%");
}