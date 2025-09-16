//! Contract test for Trainer.fit() API
#![cfg(feature = "torch-rs")]

use tch::nn::phoenix::{LightningModule, PhoenixModule};
use tch::training::{Trainer, TrainerConfig, StepOutput, OptimizerConfig};
use tch::data::{Dataset, DataLoader};
use tch::{nn, Device, Kind, Tensor};
use std::collections::HashMap;

// Mock model for testing
#[derive(Debug)]
struct TestModel {
    linear: nn::phoenix::Linear,
    training: bool,
}

impl TestModel {
    fn new() -> Self {
        Self {
            linear: nn::phoenix::Linear::new(10, 2),
            training: true,
        }
    }
}

impl nn::Module for TestModel {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.linear.forward(xs)
    }
}

impl LightningModule for TestModel {
    fn training_step(&mut self, batch: (Tensor, Tensor), _batch_idx: usize) -> StepOutput {
        let (x, y) = batch;
        let logits = self.forward(&x);
        let loss = logits.cross_entropy_for_logits(&y);

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
        let logits = self.forward(&x);
        let loss = logits.cross_entropy_for_logits(&y);

        StepOutput {
            loss: loss.shallow_clone(),
            logs: {
                let mut logs = HashMap::new();
                logs.insert("val_loss".to_string(), f64::try_from(&loss).unwrap());
                logs
            },
        }
    }

    fn configure_optimizers(&self) -> OptimizerConfig {
        OptimizerConfig::adam(self.parameters_mut(), 1e-3)
    }
}

// Implement PhoenixModule manually for test
tch::impl_phoenix_module!(TestModel {
    linear: nn::phoenix::Linear,
});

// Mock dataset
struct MockDataset {
    size: usize,
}

impl Dataset for MockDataset {
    type Item = (Tensor, i64);

    fn len(&self) -> usize {
        self.size
    }

    fn get(&self, index: usize) -> Result<Self::Item, tch::data::DatasetError> {
        if index >= self.size {
            return Err(tch::data::DatasetError::IndexOutOfBounds {
                index,
                size: self.size,
            });
        }
        let x = Tensor::randn(&[10], (Kind::Float, Device::Cpu));
        let y = (index % 2) as i64;
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
fn test_trainer_fit_basic() {
    let mut model = TestModel::new();
    let train_dataset = MockDataset { size: 100 };
    let val_dataset = MockDataset { size: 20 };

    let train_loader = DataLoader::with_batch_size(train_dataset, 10);
    let val_loader = DataLoader::with_batch_size(val_dataset, 10);

    let config = TrainerConfig {
        max_epochs: 2,
        accelerator: tch::training::Accelerator::Cpu,
        ..Default::default()
    };

    let mut trainer = Trainer::new(config);
    let result = trainer.fit(&mut model, train_loader, Some(val_loader));

    assert!(result.is_ok());
    assert_eq!(trainer.current_epoch(), 2);
}

#[test]
fn test_trainer_early_stopping() {
    let mut model = TestModel::new();
    let train_dataset = MockDataset { size: 100 };
    let val_dataset = MockDataset { size: 20 };

    let train_loader = DataLoader::with_batch_size(train_dataset, 10);
    let val_loader = DataLoader::with_batch_size(val_dataset, 10);

    let config = TrainerConfig {
        max_epochs: 100,
        early_stopping: Some(tch::training::EarlyStoppingConfig {
            monitor: "val_loss".to_string(),
            patience: 3,
            mode: tch::training::Mode::Min,
            min_delta: 0.0001,
        }),
        ..Default::default()
    };

    let mut trainer = Trainer::new(config);
    let result = trainer.fit(&mut model, train_loader, Some(val_loader));

    assert!(result.is_ok());
    // Should stop early, not reach 100 epochs
    assert!(trainer.current_epoch() < 100);
}

#[test]
fn test_trainer_checkpointing() {
    let mut model = TestModel::new();
    let train_dataset = MockDataset { size: 50 };
    let train_loader = DataLoader::with_batch_size(train_dataset, 10);

    let checkpoint_dir = std::env::temp_dir().join("test_checkpoints");
    std::fs::create_dir_all(&checkpoint_dir).ok();

    let config = TrainerConfig {
        max_epochs: 3,
        checkpoint_callback: Some(tch::training::CheckpointConfig {
            save_dir: checkpoint_dir.clone(),
            save_top_k: 2,
            monitor: "train_loss".to_string(),
            mode: tch::training::Mode::Min,
            save_last: true,
            save_weights_only: false,
            ..Default::default()
        }),
        ..Default::default()
    };

    let mut trainer = Trainer::new(config);
    let result = trainer.fit(&mut model, train_loader, None);

    assert!(result.is_ok());

    // Check that checkpoints were saved
    let checkpoints = std::fs::read_dir(&checkpoint_dir)
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map_or(false, |ext| ext == "ckpt")
        })
        .count();

    assert!(checkpoints > 0, "Should have saved checkpoints");

    // Cleanup
    std::fs::remove_dir_all(&checkpoint_dir).ok();
}