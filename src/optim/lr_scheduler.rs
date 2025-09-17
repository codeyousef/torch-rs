//! Learning rate scheduler implementations for Project Phoenix
//!
//! Provides various strategies for adjusting learning rates during training

use crate::optim::PhoenixOptimizer;
use std::f64::consts::PI;

/// Base trait for learning rate schedulers
pub trait LRScheduler {
    /// Update learning rates for the optimizer
    fn step(&mut self);

    /// Get the current learning rate(s)
    fn get_last_lr(&self) -> Vec<f64>;

    /// Get the current epoch
    fn get_epoch(&self) -> i64;
}

/// Step learning rate scheduler
/// Decays the learning rate by gamma every step_size epochs
#[derive(Debug)]
pub struct StepLR<O: PhoenixOptimizer> {
    optimizer: *mut O,
    step_size: i64,
    gamma: f64,
    base_lrs: Vec<f64>,
    epoch: i64,
    last_epoch: i64,
}

impl<O: PhoenixOptimizer> StepLR<O> {
    pub fn new(optimizer: &mut O, step_size: i64, gamma: f64) -> Self {
        let base_lrs = optimizer
            .parameter_groups()
            .iter()
            .map(|group| group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3))
            .collect();

        Self {
            optimizer: optimizer as *mut O,
            step_size,
            gamma,
            base_lrs,
            epoch: 0,
            last_epoch: -1,
        }
    }

    fn update_lr(&mut self) {
        let optimizer = unsafe { &mut *self.optimizer };

        for (group, &base_lr) in optimizer.parameter_groups_mut().iter_mut().zip(&self.base_lrs) {
            let lr = if self.epoch == 0 {
                base_lr
            } else {
                base_lr * self.gamma.powi((self.epoch / self.step_size) as i32)
            };
            group.set_option("lr", lr.into());
        }
    }
}

impl<O: PhoenixOptimizer> LRScheduler for StepLR<O> {
    fn step(&mut self) {
        self.epoch += 1;
        self.update_lr();
        self.last_epoch = self.epoch;
    }

    fn get_last_lr(&self) -> Vec<f64> {
        let optimizer = unsafe { &*self.optimizer };
        optimizer
            .parameter_groups()
            .iter()
            .map(|group| group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3))
            .collect()
    }

    fn get_epoch(&self) -> i64 {
        self.epoch
    }
}

/// Exponential learning rate scheduler
/// Decays the learning rate by gamma every epoch
#[derive(Debug)]
pub struct ExponentialLR<O: PhoenixOptimizer> {
    optimizer: *mut O,
    gamma: f64,
    base_lrs: Vec<f64>,
    epoch: i64,
    last_epoch: i64,
}

impl<O: PhoenixOptimizer> ExponentialLR<O> {
    pub fn new(optimizer: &mut O, gamma: f64) -> Self {
        let base_lrs = optimizer
            .parameter_groups()
            .iter()
            .map(|group| group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3))
            .collect();

        Self { optimizer: optimizer as *mut O, gamma, base_lrs, epoch: 0, last_epoch: -1 }
    }

    fn update_lr(&mut self) {
        let optimizer = unsafe { &mut *self.optimizer };

        for (group, &base_lr) in optimizer.parameter_groups_mut().iter_mut().zip(&self.base_lrs) {
            let lr = base_lr * self.gamma.powi(self.epoch as i32);
            group.set_option("lr", lr.into());
        }
    }
}

impl<O: PhoenixOptimizer> LRScheduler for ExponentialLR<O> {
    fn step(&mut self) {
        self.epoch += 1;
        self.update_lr();
        self.last_epoch = self.epoch;
    }

    fn get_last_lr(&self) -> Vec<f64> {
        let optimizer = unsafe { &*self.optimizer };
        optimizer
            .parameter_groups()
            .iter()
            .map(|group| group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3))
            .collect()
    }

    fn get_epoch(&self) -> i64 {
        self.epoch
    }
}

/// Cosine annealing learning rate scheduler
/// Anneals learning rate using cosine schedule
#[derive(Debug)]
pub struct CosineAnnealingLR<O: PhoenixOptimizer> {
    optimizer: *mut O,
    t_max: i64,
    eta_min: f64,
    base_lrs: Vec<f64>,
    epoch: i64,
    last_epoch: i64,
}

impl<O: PhoenixOptimizer> CosineAnnealingLR<O> {
    pub fn new(optimizer: &mut O, t_max: i64, eta_min: f64) -> Self {
        let base_lrs = optimizer
            .parameter_groups()
            .iter()
            .map(|group| group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3))
            .collect();

        Self { optimizer: optimizer as *mut O, t_max, eta_min, base_lrs, epoch: 0, last_epoch: -1 }
    }

    fn update_lr(&mut self) {
        let optimizer = unsafe { &mut *self.optimizer };

        for (group, &base_lr) in optimizer.parameter_groups_mut().iter_mut().zip(&self.base_lrs) {
            let lr = if self.epoch == 0 {
                base_lr
            } else if self.epoch % (2 * self.t_max) < self.t_max {
                self.eta_min
                    + (base_lr - self.eta_min)
                        * (1.0 + (PI * (self.epoch % self.t_max) as f64 / self.t_max as f64).cos())
                        / 2.0
            } else {
                self.eta_min
                    + (base_lr - self.eta_min)
                        * (1.0 + (PI * (self.t_max - 1) as f64 / self.t_max as f64).cos())
                        / 2.0
            };
            group.set_option("lr", lr.into());
        }
    }
}

impl<O: PhoenixOptimizer> LRScheduler for CosineAnnealingLR<O> {
    fn step(&mut self) {
        self.epoch += 1;
        self.update_lr();
        self.last_epoch = self.epoch;
    }

    fn get_last_lr(&self) -> Vec<f64> {
        let optimizer = unsafe { &*self.optimizer };
        optimizer
            .parameter_groups()
            .iter()
            .map(|group| group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3))
            .collect()
    }

    fn get_epoch(&self) -> i64 {
        self.epoch
    }
}

/// Reduce learning rate on plateau scheduler
/// Reduces learning rate when a metric has stopped improving
#[derive(Debug)]
pub struct ReduceLROnPlateau<O: PhoenixOptimizer> {
    optimizer: *mut O,
    mode: PlateauMode,
    factor: f64,
    patience: i64,
    threshold: f64,
    threshold_mode: ThresholdMode,
    cooldown: i64,
    min_lr: f64,
    eps: f64,
    best: f64,
    num_bad_epochs: i64,
    cooldown_counter: i64,
    last_epoch: i64,
}

#[derive(Debug, Clone, Copy)]
pub enum PlateauMode {
    Min,
    Max,
}

#[derive(Debug, Clone, Copy)]
pub enum ThresholdMode {
    Rel,
    Abs,
}

impl<O: PhoenixOptimizer> ReduceLROnPlateau<O> {
    pub fn new(optimizer: &mut O) -> Self {
        Self {
            optimizer: optimizer as *mut O,
            mode: PlateauMode::Min,
            factor: 0.1,
            patience: 10,
            threshold: 1e-4,
            threshold_mode: ThresholdMode::Rel,
            cooldown: 0,
            min_lr: 0.0,
            eps: 1e-8,
            best: f64::INFINITY,
            num_bad_epochs: 0,
            cooldown_counter: 0,
            last_epoch: 0,
        }
    }

    pub fn mode(mut self, mode: PlateauMode) -> Self {
        self.mode = mode;
        self.best = match mode {
            PlateauMode::Min => f64::INFINITY,
            PlateauMode::Max => f64::NEG_INFINITY,
        };
        self
    }

    pub fn factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    pub fn patience(mut self, patience: i64) -> Self {
        self.patience = patience;
        self
    }

    pub fn min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    pub fn step_with_metric(&mut self, metric: f64) {
        let is_better = self.is_better(metric);

        if is_better {
            self.best = metric;
            self.num_bad_epochs = 0;
        } else {
            self.num_bad_epochs += 1;
        }

        if self.in_cooldown() {
            self.cooldown_counter -= 1;
            self.num_bad_epochs = 0;
        }

        if self.num_bad_epochs > self.patience {
            self.reduce_lr();
            self.cooldown_counter = self.cooldown;
            self.num_bad_epochs = 0;
        }

        self.last_epoch += 1;
    }

    fn is_better(&self, metric: f64) -> bool {
        let threshold = match self.threshold_mode {
            ThresholdMode::Rel => match self.mode {
                PlateauMode::Min => self.best * (1.0 - self.threshold),
                PlateauMode::Max => self.best * (1.0 + self.threshold),
            },
            ThresholdMode::Abs => match self.mode {
                PlateauMode::Min => self.best - self.threshold,
                PlateauMode::Max => self.best + self.threshold,
            },
        };

        match self.mode {
            PlateauMode::Min => metric < threshold,
            PlateauMode::Max => metric > threshold,
        }
    }

    fn in_cooldown(&self) -> bool {
        self.cooldown_counter > 0
    }

    fn reduce_lr(&mut self) {
        let optimizer = unsafe { &mut *self.optimizer };

        for group in optimizer.parameter_groups_mut() {
            let old_lr: f64 = group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3);
            let new_lr = (old_lr * self.factor).max(self.min_lr);

            if old_lr - new_lr > self.eps {
                group.set_option("lr", new_lr.into());
            }
        }
    }
}

impl<O: PhoenixOptimizer> LRScheduler for ReduceLROnPlateau<O> {
    fn step(&mut self) {
        // ReduceLROnPlateau requires metric, use step_with_metric instead
        panic!("ReduceLROnPlateau requires a metric. Use step_with_metric() instead.");
    }

    fn get_last_lr(&self) -> Vec<f64> {
        let optimizer = unsafe { &*self.optimizer };
        optimizer
            .parameter_groups()
            .iter()
            .map(|group| group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3))
            .collect()
    }

    fn get_epoch(&self) -> i64 {
        self.last_epoch
    }
}

/// One cycle learning rate scheduler
/// Implements the 1cycle learning rate policy
#[derive(Debug)]
pub struct OneCycleLR<O: PhoenixOptimizer> {
    optimizer: *mut O,
    max_lr: Vec<f64>,
    total_steps: i64,
    epochs: Option<i64>,
    steps_per_epoch: Option<i64>,
    pct_start: f64,
    anneal_strategy: AnnealStrategy,
    cycle_momentum: bool,
    base_momentum: f64,
    max_momentum: f64,
    div_factor: f64,
    final_div_factor: f64,
    last_epoch: i64,
    step_count: i64,
}

#[derive(Debug, Clone, Copy)]
pub enum AnnealStrategy {
    Cos,
    Linear,
}

impl<O: PhoenixOptimizer> OneCycleLR<O> {
    pub fn new(optimizer: &mut O, max_lr: Vec<f64>, total_steps: i64) -> Self {
        Self {
            optimizer: optimizer as *mut O,
            max_lr,
            total_steps,
            epochs: None,
            steps_per_epoch: None,
            pct_start: 0.3,
            anneal_strategy: AnnealStrategy::Cos,
            cycle_momentum: true,
            base_momentum: 0.85,
            max_momentum: 0.95,
            div_factor: 25.0,
            final_div_factor: 1e4,
            last_epoch: -1,
            step_count: 0,
        }
    }

    pub fn pct_start(mut self, pct_start: f64) -> Self {
        self.pct_start = pct_start;
        self
    }

    pub fn anneal_strategy(mut self, strategy: AnnealStrategy) -> Self {
        self.anneal_strategy = strategy;
        self
    }

    pub fn cycle_momentum(mut self, cycle: bool) -> Self {
        self.cycle_momentum = cycle;
        self
    }

    fn update_lr(&mut self) {
        let optimizer = unsafe { &mut *self.optimizer };
        let step_num = self.step_count;

        let step_size_up = (self.pct_start * self.total_steps as f64) as i64;
        let step_size_down = self.total_steps - step_size_up;

        for (group, &max_lr) in optimizer.parameter_groups_mut().iter_mut().zip(&self.max_lr) {
            let initial_lr = max_lr / self.div_factor;
            let min_lr = initial_lr / self.final_div_factor;

            let lr = if step_num <= step_size_up {
                // Warmup phase
                let pct = step_num as f64 / step_size_up as f64;
                initial_lr + (max_lr - initial_lr) * pct
            } else {
                // Annealing phase
                let pct = (step_num - step_size_up) as f64 / step_size_down as f64;
                let lr = match self.anneal_strategy {
                    AnnealStrategy::Cos => {
                        min_lr + (max_lr - min_lr) * (1.0 + (PI * pct).cos()) / 2.0
                    }
                    AnnealStrategy::Linear => max_lr + (min_lr - max_lr) * pct,
                };
                lr
            };

            group.set_option("lr", lr.into());

            // Update momentum if applicable
            if self.cycle_momentum {
                let momentum = if step_num <= step_size_up {
                    let pct = step_num as f64 / step_size_up as f64;
                    self.max_momentum - (self.max_momentum - self.base_momentum) * pct
                } else {
                    let pct = (step_num - step_size_up) as f64 / step_size_down as f64;
                    match self.anneal_strategy {
                        AnnealStrategy::Cos => {
                            self.base_momentum
                                + (self.max_momentum - self.base_momentum)
                                    * (1.0 + (PI * pct).cos())
                                    / 2.0
                        }
                        AnnealStrategy::Linear => {
                            self.base_momentum + (self.max_momentum - self.base_momentum) * pct
                        }
                    }
                };

                // Set momentum for SGD or beta1 for Adam-like optimizers
                group.set_option("momentum", momentum.into());
                group.set_option("beta1", momentum.into());
            }
        }
    }
}

impl<O: PhoenixOptimizer> LRScheduler for OneCycleLR<O> {
    fn step(&mut self) {
        self.step_count += 1;
        self.update_lr();
        self.last_epoch = self.step_count / self.steps_per_epoch.unwrap_or(1);
    }

    fn get_last_lr(&self) -> Vec<f64> {
        let optimizer = unsafe { &*self.optimizer };
        optimizer
            .parameter_groups()
            .iter()
            .map(|group| group.get_option("lr").map(|v| v.into()).unwrap_or(1e-3))
            .collect()
    }

    fn get_epoch(&self) -> i64 {
        self.last_epoch
    }
}
