//! Metrics for model evaluation
//!
//! Provides common metrics for classification and regression tasks

use crate::Tensor;
use std::collections::HashMap;

/// Base trait for all metrics
pub trait Metric {
    /// Update the metric with new predictions and targets
    fn update(&mut self, preds: &Tensor, target: &Tensor);
    
    /// Compute the metric value
    fn compute(&self) -> f64;
    
    /// Reset the metric state
    fn reset(&mut self);
    
    /// Get metric name
    fn name(&self) -> &str;
}

/// Accuracy metric for classification
pub struct Accuracy {
    correct: i64,
    total: i64,
    top_k: i64,
}

impl Accuracy {
    pub fn new(top_k: i64) -> Self {
        Self {
            correct: 0,
            total: 0,
            top_k,
        }
    }
    
    pub fn top1() -> Self {
        Self::new(1)
    }
    
    pub fn top5() -> Self {
        Self::new(5)
    }
}

impl Metric for Accuracy {
    fn update(&mut self, preds: &Tensor, target: &Tensor) {
        let batch_size = preds.size()[0];
        
        if self.top_k == 1 {
            let pred_classes = preds.argmax(-1, false);
            let correct = pred_classes.eq_tensor(target).sum(crate::Kind::Int64);
            self.correct += correct.int64_value(&[]);
        } else {
            // Top-k accuracy
            let (_, top_indices) = preds.topk(self.top_k, -1, true, true);
            let target_expanded = target.unsqueeze(-1).expand_as(&top_indices);
            let correct = top_indices.eq_tensor(&target_expanded)
                .sum_dim_intlist(&[-1], false, crate::Kind::Int64)
                .sum(crate::Kind::Int64);
            self.correct += correct.int64_value(&[]);
        }
        
        self.total += batch_size;
    }
    
    fn compute(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }
    
    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }
    
    fn name(&self) -> &str {
        if self.top_k == 1 {
            "accuracy"
        } else {
            "top_k_accuracy"
        }
    }
}

/// Mean Squared Error metric
pub struct MeanSquaredError {
    sum_squared_error: f64,
    count: i64,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        Self {
            sum_squared_error: 0.0,
            count: 0,
        }
    }
}

impl Metric for MeanSquaredError {
    fn update(&mut self, preds: &Tensor, target: &Tensor) {
        let squared_error = (preds - target).pow_tensor_scalar(2).sum(crate::Kind::Float);
        self.sum_squared_error += f64::from(squared_error);
        self.count += preds.numel() as i64;
    }
    
    fn compute(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_squared_error / self.count as f64
        }
    }
    
    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.count = 0;
    }
    
    fn name(&self) -> &str {
        "mse"
    }
}

/// Mean Absolute Error metric
pub struct MeanAbsoluteError {
    sum_absolute_error: f64,
    count: i64,
}

impl MeanAbsoluteError {
    pub fn new() -> Self {
        Self {
            sum_absolute_error: 0.0,
            count: 0,
        }
    }
}

impl Metric for MeanAbsoluteError {
    fn update(&mut self, preds: &Tensor, target: &Tensor) {
        let absolute_error = (preds - target).abs().sum(crate::Kind::Float);
        self.sum_absolute_error += f64::from(absolute_error);
        self.count += preds.numel() as i64;
    }
    
    fn compute(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_absolute_error / self.count as f64
        }
    }
    
    fn reset(&mut self) {
        self.sum_absolute_error = 0.0;
        self.count = 0;
    }
    
    fn name(&self) -> &str {
        "mae"
    }
}

/// F1 Score metric for binary classification
pub struct F1Score {
    true_positives: i64,
    false_positives: i64,
    false_negatives: i64,
    threshold: f64,
}

impl F1Score {
    pub fn new(threshold: f64) -> Self {
        Self {
            true_positives: 0,
            false_positives: 0,
            false_negatives: 0,
            threshold,
        }
    }
}

impl Metric for F1Score {
    fn update(&mut self, preds: &Tensor, target: &Tensor) {
        let pred_binary = preds.ge(self.threshold);
        let target_binary = target.ge(0.5);
        
        let tp = (&pred_binary * &target_binary).sum(crate::Kind::Int64);
        let fp = (&pred_binary * &target_binary.logical_not()).sum(crate::Kind::Int64);
        let fn_val = (&pred_binary.logical_not() * &target_binary).sum(crate::Kind::Int64);
        
        self.true_positives += tp.int64_value(&[]);
        self.false_positives += fp.int64_value(&[]);
        self.false_negatives += fn_val.int64_value(&[]);
    }
    
    fn compute(&self) -> f64 {
        let precision = if self.true_positives + self.false_positives == 0 {
            0.0
        } else {
            self.true_positives as f64 / (self.true_positives + self.false_positives) as f64
        };
        
        let recall = if self.true_positives + self.false_negatives == 0 {
            0.0
        } else {
            self.true_positives as f64 / (self.true_positives + self.false_negatives) as f64
        };
        
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
    
    fn reset(&mut self) {
        self.true_positives = 0;
        self.false_positives = 0;
        self.false_negatives = 0;
    }
    
    fn name(&self) -> &str {
        "f1_score"
    }
}

/// Confusion Matrix for multi-class classification
pub struct ConfusionMatrix {
    num_classes: i64,
    matrix: Tensor,
}

impl ConfusionMatrix {
    pub fn new(num_classes: i64) -> Self {
        Self {
            num_classes,
            matrix: Tensor::zeros(&[num_classes, num_classes], (crate::Kind::Int64, crate::Device::Cpu)),
        }
    }
    
    pub fn update(&mut self, preds: &Tensor, target: &Tensor) {
        let pred_classes = preds.argmax(-1, false);
        
        for i in 0..pred_classes.size()[0] {
            let pred_idx = pred_classes.get(i).int64_value(&[]);
            let target_idx = target.get(i).int64_value(&[]);
            
            if pred_idx < self.num_classes && target_idx < self.num_classes {
                let current = self.matrix.get(target_idx).get(pred_idx);
                self.matrix.get(target_idx).get(pred_idx).copy_(&(current + 1));
            }
        }
    }
    
    pub fn compute(&self) -> Tensor {
        self.matrix.shallow_clone()
    }
    
    pub fn reset(&mut self) {
        self.matrix = Tensor::zeros(&[self.num_classes, self.num_classes], 
                                    (crate::Kind::Int64, crate::Device::Cpu));
    }
    
    pub fn precision_per_class(&self) -> Tensor {
        let tp_fp_sum = self.matrix.sum_dim_intlist(&[0], false, crate::Kind::Float);
        let tp = self.matrix.diag(0);
        tp / (tp_fp_sum + 1e-10)
    }
    
    pub fn recall_per_class(&self) -> Tensor {
        let tp_fn_sum = self.matrix.sum_dim_intlist(&[1], false, crate::Kind::Float);
        let tp = self.matrix.diag(0);
        tp / (tp_fn_sum + 1e-10)
    }
    
    pub fn f1_per_class(&self) -> Tensor {
        let precision = self.precision_per_class();
        let recall = self.recall_per_class();
        &precision * &recall * 2.0 / (&precision + &recall + 1e-10)
    }
}

/// Collection of metrics
pub struct MetricCollection {
    metrics: HashMap<String, Box<dyn Metric>>,
}

impl MetricCollection {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }
    
    pub fn add_metric(&mut self, name: String, metric: Box<dyn Metric>) {
        self.metrics.insert(name, metric);
    }
    
    pub fn update(&mut self, preds: &Tensor, target: &Tensor) {
        for metric in self.metrics.values_mut() {
            metric.update(preds, target);
        }
    }
    
    pub fn compute(&self) -> HashMap<String, f64> {
        self.metrics
            .iter()
            .map(|(name, metric)| (name.clone(), metric.compute()))
            .collect()
    }
    
    pub fn reset(&mut self) {
        for metric in self.metrics.values_mut() {
            metric.reset();
        }
    }
}

/// AUROC (Area Under ROC Curve) metric
pub struct AUROC {
    predictions: Vec<f64>,
    targets: Vec<i64>,
}

impl AUROC {
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            targets: Vec::new(),
        }
    }
    
    fn compute_auroc(&self) -> f64 {
        if self.predictions.is_empty() {
            return 0.0;
        }
        
        // Sort by predictions
        let mut paired: Vec<(f64, i64)> = self.predictions
            .iter()
            .zip(self.targets.iter())
            .map(|(&p, &t)| (p, t))
            .collect();
        paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        // Calculate AUROC using trapezoidal rule
        let mut tpr_prev = 0.0;
        let mut fpr_prev = 0.0;
        let mut auc = 0.0;
        let mut tp = 0;
        let mut fp = 0;
        
        let total_positives = self.targets.iter().filter(|&&t| t == 1).count() as f64;
        let total_negatives = self.targets.iter().filter(|&&t| t == 0).count() as f64;
        
        for (_, target) in paired {
            if target == 1 {
                tp += 1;
            } else {
                fp += 1;
            }
            
            let tpr = tp as f64 / total_positives;
            let fpr = fp as f64 / total_negatives;
            
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0;
            
            tpr_prev = tpr;
            fpr_prev = fpr;
        }
        
        auc
    }
}

impl Metric for AUROC {
    fn update(&mut self, preds: &Tensor, target: &Tensor) {
        let preds_vec: Vec<f64> = Vec::from(preds);
        let target_vec: Vec<i64> = Vec::from(target);
        
        self.predictions.extend(preds_vec);
        self.targets.extend(target_vec);
    }
    
    fn compute(&self) -> f64 {
        self.compute_auroc()
    }
    
    fn reset(&mut self) {
        self.predictions.clear();
        self.targets.clear();
    }
    
    fn name(&self) -> &str {
        "auroc"
    }
}