//! Lightning-style trainer for Project Phoenix
//!
//! Provides high-level training abstractions similar to PyTorch Lightning

use crate::nn::phoenix::{PhoenixModule, PhoenixModuleError};
use crate::optim::{PhoenixOptimizer, LRScheduler};
use crate::{Device, Tensor};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub max_epochs: i64,
    pub val_check_interval: f64,
    pub gradient_clip_val: Option<f64>,
    pub accumulate_grad_batches: i64,
    pub enable_progress_bar: bool,
    pub enable_checkpointing: bool,
    pub checkpoint_dir: String,
    pub early_stopping_patience: Option<i64>,
    pub early_stopping_metric: String,
    pub early_stopping_mode: EarlyStoppingMode,
    pub devices: Vec<Device>,
    pub precision: Precision,
    pub log_every_n_steps: i64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            max_epochs: 10,
            val_check_interval: 1.0,
            gradient_clip_val: None,
            accumulate_grad_batches: 1,
            enable_progress_bar: true,
            enable_checkpointing: true,
            checkpoint_dir: "checkpoints".to_string(),
            early_stopping_patience: None,
            early_stopping_metric: "val_loss".to_string(),
            early_stopping_mode: EarlyStoppingMode::Min,
            devices: vec![Device::cuda_if_available()],
            precision: Precision::F32,
            log_every_n_steps: 50,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EarlyStoppingMode {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy)]
pub enum Precision {
    F16,
    F32,
    BF16,
}

/// Lightning-style module trait
pub trait LightningModule: PhoenixModule {
    /// Configure optimizers and schedulers
    fn configure_optimizers(&self) -> (Box<dyn PhoenixOptimizer>, Option<Box<dyn LRScheduler>>);
    
    /// Training step
    fn training_step(&mut self, batch: &Batch, batch_idx: i64) -> TrainingStepOutput;
    
    /// Validation step
    fn validation_step(&mut self, batch: &Batch, batch_idx: i64) -> ValidationStepOutput;
    
    /// Test step
    fn test_step(&mut self, batch: &Batch, batch_idx: i64) -> TestStepOutput;
    
    /// Called at the end of training epoch
    fn on_training_epoch_end(&mut self) {}
    
    /// Called at the end of validation epoch
    fn on_validation_epoch_end(&mut self) {}
    
    /// Called at the beginning of training
    fn on_train_start(&mut self) {}
    
    /// Called at the end of training
    fn on_train_end(&mut self) {}
}

/// Batch data container
#[derive(Debug)]
pub struct Batch {
    pub inputs: Tensor,
    pub targets: Tensor,
}

/// Training step output
#[derive(Debug)]
pub struct TrainingStepOutput {
    pub loss: Tensor,
    pub log: HashMap<String, f64>,
}

/// Validation step output
#[derive(Debug)]
pub struct ValidationStepOutput {
    pub loss: Tensor,
    pub log: HashMap<String, f64>,
}

/// Test step output
#[derive(Debug)]
pub struct TestStepOutput {
    pub loss: Tensor,
    pub log: HashMap<String, f64>,
}

/// DataLoader trait
pub trait DataLoader {
    fn len(&self) -> usize;
    fn batch(&self, idx: usize) -> Batch;
    fn shuffle(&mut self);
}

/// Trainer for Lightning-style training
pub struct Trainer {
    config: TrainerConfig,
    current_epoch: i64,
    global_step: i64,
    best_metric: f64,
    patience_counter: i64,
    metrics_history: Vec<HashMap<String, f64>>,
}

impl Trainer {
    /// Create a new trainer
    pub fn new(config: TrainerConfig) -> Self {
        Self {
            config,
            current_epoch: 0,
            global_step: 0,
            best_metric: match config.early_stopping_mode {
                EarlyStoppingMode::Min => f64::INFINITY,
                EarlyStoppingMode::Max => f64::NEG_INFINITY,
            },
            patience_counter: 0,
            metrics_history: Vec::new(),
        }
    }

    /// Fit the model
    pub fn fit<M>(
        &mut self,
        model: &mut M,
        train_dataloader: &mut dyn DataLoader,
        val_dataloader: Option<&mut dyn DataLoader>,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        M: LightningModule,
    {
        // Setup
        let (mut optimizer, lr_scheduler) = model.configure_optimizers();
        model.on_train_start();
        
        // Move model to device
        let device = self.config.devices[0];
        model.to_device(device)?;
        
        // Training loop
        for epoch in 0..self.config.max_epochs {
            self.current_epoch = epoch;
            
            // Training epoch
            self.training_epoch(
                model,
                train_dataloader,
                &mut optimizer,
                device,
            )?;
            
            // Validation epoch
            if let Some(val_dl) = val_dataloader {
                let val_metrics = self.validation_epoch(model, val_dl, device)?;
                
                // Early stopping
                if let Some(patience) = self.config.early_stopping_patience {
                    if self.check_early_stopping(&val_metrics, patience) {
                        println!("Early stopping triggered at epoch {}", epoch);
                        break;
                    }
                }
            }
            
            // Learning rate scheduling
            if let Some(ref mut scheduler) = lr_scheduler.as_mut() {
                scheduler.step();
            }
            
            // Checkpointing
            if self.config.enable_checkpointing {
                self.save_checkpoint(model, &optimizer, epoch)?;
            }
            
            model.on_training_epoch_end();
        }
        
        model.on_train_end();
        Ok(())
    }

    /// Run a training epoch
    fn training_epoch<M>(
        &mut self,
        model: &mut M,
        dataloader: &mut dyn DataLoader,
        optimizer: &mut Box<dyn PhoenixOptimizer>,
        device: Device,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        M: LightningModule,
    {
        model.set_training(true);
        dataloader.shuffle();
        
        let progress_bar = if self.config.enable_progress_bar {
            let pb = ProgressBar::new(dataloader.len() as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            Some(pb)
        } else {
            None
        };
        
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        
        for batch_idx in 0..dataloader.len() {
            let mut batch = dataloader.batch(batch_idx);
            batch.inputs = batch.inputs.to_device(device);
            batch.targets = batch.targets.to_device(device);
            
            // Forward pass
            let output = model.training_step(&batch, batch_idx as i64);
            let loss = output.loss;
            
            // Backward pass
            loss.backward();
            
            // Gradient accumulation
            if (batch_idx + 1) % self.config.accumulate_grad_batches as usize == 0 {
                // Gradient clipping
                if let Some(clip_val) = self.config.gradient_clip_val {
                    self.clip_gradients(model, clip_val);
                }
                
                optimizer.step()?;
                optimizer.zero_grad();
            }
            
            epoch_loss += loss.double_value(&[]);
            num_batches += 1;
            self.global_step += 1;
            
            // Logging
            if self.global_step % self.config.log_every_n_steps == 0 {
                for (key, value) in output.log {
                    println!("Step {}: {} = {:.4}", self.global_step, key, value);
                }
            }
            
            if let Some(ref pb) = progress_bar {
                pb.inc(1);
            }
        }
        
        if let Some(pb) = progress_bar {
            pb.finish_with_message(format!("Epoch {} - Loss: {:.4}", 
                self.current_epoch, epoch_loss / num_batches as f64));
        }
        
        Ok(())
    }

    /// Run a validation epoch
    fn validation_epoch<M>(
        &mut self,
        model: &mut M,
        dataloader: &mut dyn DataLoader,
        device: Device,
    ) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>>
    where
        M: LightningModule,
    {
        model.set_training(false);
        
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        let mut all_metrics = HashMap::new();
        
        for batch_idx in 0..dataloader.len() {
            let mut batch = dataloader.batch(batch_idx);
            batch.inputs = batch.inputs.to_device(device);
            batch.targets = batch.targets.to_device(device);
            
            let output = model.validation_step(&batch, batch_idx as i64);
            
            total_loss += output.loss.double_value(&[]);
            num_batches += 1;
            
            // Aggregate metrics
            for (key, value) in output.log {
                *all_metrics.entry(key).or_insert(0.0) += value;
            }
        }
        
        // Average metrics
        let avg_loss = total_loss / num_batches as f64;
        all_metrics.insert("val_loss".to_string(), avg_loss);
        
        for value in all_metrics.values_mut() {
            *value /= num_batches as f64;
        }
        
        println!("\nValidation - Epoch {}: Loss = {:.4}", self.current_epoch, avg_loss);
        
        model.on_validation_epoch_end();
        self.metrics_history.push(all_metrics.clone());
        
        Ok(all_metrics)
    }

    /// Check early stopping condition
    fn check_early_stopping(&mut self, metrics: &HashMap<String, f64>, patience: i64) -> bool {
        let metric_value = metrics.get(&self.config.early_stopping_metric)
            .copied()
            .unwrap_or(0.0);
        
        let is_better = match self.config.early_stopping_mode {
            EarlyStoppingMode::Min => metric_value < self.best_metric,
            EarlyStoppingMode::Max => metric_value > self.best_metric,
        };
        
        if is_better {
            self.best_metric = metric_value;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
        }
        
        self.patience_counter >= patience
    }

    /// Clip gradients
    fn clip_gradients<M: PhoenixModule>(&self, model: &M, max_norm: f64) {
        let params = model.parameters();
        let mut total_norm = 0.0;
        
        // Calculate total norm
        for param in &params {
            if let Some(grad) = param.grad() {
                let param_norm = grad.norm().double_value(&[]);
                total_norm += param_norm * param_norm;
            }
        }
        
        total_norm = total_norm.sqrt();
        
        // Clip if necessary
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for param in params {
                if let Some(grad) = param.grad() {
                    let _ = grad.mul_(scale);
                }
            }
        }
    }

    /// Save checkpoint
    fn save_checkpoint<M: PhoenixModule>(
        &self,
        model: &M,
        optimizer: &Box<dyn PhoenixOptimizer>,
        epoch: i64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        std::fs::create_dir_all(&self.config.checkpoint_dir)?;
        
        let checkpoint_path = Path::new(&self.config.checkpoint_dir)
            .join(format!("checkpoint_epoch_{}.pt", epoch));
        
        // Save model state dict
        let model_state = model.state_dict();
        let optimizer_state = optimizer.state_dict();
        
        // In a real implementation, we would serialize these to a file
        // For now, just return Ok
        Ok(())
    }

    /// Test the model
    pub fn test<M>(
        &mut self,
        model: &mut M,
        test_dataloader: &mut dyn DataLoader,
    ) -> Result<HashMap<String, f64>, Box<dyn std::error::Error>>
    where
        M: LightningModule,
    {
        model.set_training(false);
        let device = self.config.devices[0];
        model.to_device(device)?;
        
        let mut total_loss = 0.0;
        let mut num_batches = 0;
        let mut all_metrics = HashMap::new();
        
        let progress_bar = if self.config.enable_progress_bar {
            Some(ProgressBar::new(test_dataloader.len() as u64))
        } else {
            None
        };
        
        for batch_idx in 0..test_dataloader.len() {
            let mut batch = test_dataloader.batch(batch_idx);
            batch.inputs = batch.inputs.to_device(device);
            batch.targets = batch.targets.to_device(device);
            
            let output = model.test_step(&batch, batch_idx as i64);
            
            total_loss += output.loss.double_value(&[]);
            num_batches += 1;
            
            for (key, value) in output.log {
                *all_metrics.entry(key).or_insert(0.0) += value;
            }
            
            if let Some(ref pb) = progress_bar {
                pb.inc(1);
            }
        }
        
        // Average metrics
        for value in all_metrics.values_mut() {
            *value /= num_batches as f64;
        }
        
        all_metrics.insert("test_loss".to_string(), total_loss / num_batches as f64);
        
        if let Some(pb) = progress_bar {
            pb.finish();
        }
        
        println!("\nTest Results:");
        for (key, value) in &all_metrics {
            println!("{}: {:.4}", key, value);
        }
        
        Ok(all_metrics)
    }
}