//! Contract tests for performance baseline API
//!
//! Validates that the baseline API matches the OpenAPI specification

use tch::test_utils::{PerformanceBaseline, BaselineTracker};
use std::collections::HashMap;

#[test]
fn test_baseline_api_get_baseline() {
    // This test should fail until BaselineTracker is implemented
    let tracker = BaselineTracker::new();

    let baseline = tracker.get_baseline("tensor_matmul");

    assert!(baseline.is_ok(), "Should retrieve baseline for operation");
    let perf = baseline.unwrap();

    assert_eq!(perf.operation, "tensor_matmul", "Operation name should match");
    assert!(perf.baseline_ms > 0.0, "Baseline time must be positive");
    assert!(perf.threshold_percent >= 5.0 && perf.threshold_percent <= 50.0,
            "Threshold should be between 5% and 50%");
    assert!(perf.samples >= 10, "Should have at least 10 samples for valid baseline");
}

#[test]
fn test_baseline_api_update_baseline() {
    let tracker = BaselineTracker::new();

    let new_baseline = PerformanceBaseline {
        operation: "conv2d_forward".to_string(),
        baseline_ms: 15.5,
        variance_ms: 2.1,
        threshold_percent: 10.0,
        samples: 100,
        percentiles: {
            let mut p = HashMap::new();
            p.insert("p50".to_string(), 14.8);
            p.insert("p95".to_string(), 18.2);
            p.insert("p99".to_string(), 22.1);
            p
        },
        established_date: chrono::Utc::now(),
        commit_hash: "abc123def".to_string(),
    };

    let result = tracker.update_baseline("conv2d_forward", new_baseline);

    assert!(result.is_ok(), "Should update baseline successfully");

    // Verify update
    let updated = tracker.get_baseline("conv2d_forward").unwrap();
    assert_eq!(updated.baseline_ms, 15.5, "Baseline should be updated");
    assert_eq!(updated.samples, 100, "Sample count should be updated");
}

#[test]
fn test_baseline_api_percentiles() {
    let tracker = BaselineTracker::new();

    let baseline = tracker.get_baseline("batch_norm").unwrap();

    assert!(baseline.percentiles.contains_key("p50"), "Should have P50");
    assert!(baseline.percentiles.contains_key("p95"), "Should have P95");
    assert!(baseline.percentiles.contains_key("p99"), "Should have P99");

    let p50 = baseline.percentiles["p50"];
    let p95 = baseline.percentiles["p95"];
    let p99 = baseline.percentiles["p99"];

    assert!(p50 <= p95 && p95 <= p99, "Percentiles should be ordered: P50 <= P95 <= P99");
    assert!(p50 <= baseline.baseline_ms, "P50 should be less than or equal to mean");
}

#[test]
fn test_baseline_api_regression_detection() {
    let tracker = BaselineTracker::new();

    // Get baseline
    let baseline = tracker.get_baseline("lstm_forward").unwrap();

    // Test with current performance
    let current_time = baseline.baseline_ms * 1.05; // 5% slower
    let is_regression = tracker.check_regression("lstm_forward", current_time);

    assert!(!is_regression, "5% slowdown should not trigger regression if threshold is 10%");

    // Test with significant regression
    let slow_time = baseline.baseline_ms * 1.25; // 25% slower
    let is_regression = tracker.check_regression("lstm_forward", slow_time);

    assert!(is_regression, "25% slowdown should trigger regression");
}

#[test]
fn test_baseline_api_variance_validation() {
    let tracker = BaselineTracker::new();

    let invalid_baseline = PerformanceBaseline {
        operation: "invalid_op".to_string(),
        baseline_ms: 10.0,
        variance_ms: 8.0, // Variance too high (80% of baseline)
        threshold_percent: 10.0,
        samples: 20,
        percentiles: HashMap::new(),
        established_date: chrono::Utc::now(),
        commit_hash: "xyz789".to_string(),
    };

    let result = tracker.update_baseline("invalid_op", invalid_baseline);

    assert!(result.is_err(), "Should reject baseline with variance > 50% of mean");
}

#[test]
fn test_baseline_api_list_operations() {
    let tracker = BaselineTracker::new();

    let operations = tracker.list_operations();

    assert!(operations.is_ok(), "Should list all operations with baselines");
    let ops = operations.unwrap();

    assert!(!ops.is_empty(), "Should have baseline operations");

    // Check common operations exist
    let expected_ops = ["tensor_matmul", "conv2d_forward", "batch_norm", "lstm_forward"];
    for op in expected_ops {
        assert!(ops.contains(&op.to_string()), "Should have baseline for {}", op);
    }
}

#[test]
fn test_baseline_api_export_import() {
    let tracker = BaselineTracker::new();

    // Export all baselines
    let export = tracker.export_baselines();
    assert!(export.is_ok(), "Should export baselines");

    let json_data = export.unwrap();
    assert!(!json_data.is_empty(), "Export should contain data");

    // Create new tracker and import
    let new_tracker = BaselineTracker::new();
    let import_result = new_tracker.import_baselines(&json_data);

    assert!(import_result.is_ok(), "Should import baselines");

    // Verify imported data
    let ops = new_tracker.list_operations().unwrap();
    let original_ops = tracker.list_operations().unwrap();

    assert_eq!(ops.len(), original_ops.len(), "Should have same number of operations");
}

#[test]
fn test_baseline_api_historical_tracking() {
    let tracker = BaselineTracker::new();

    // Get historical data for an operation
    let history = tracker.get_history("tensor_add", 10);

    assert!(history.is_ok(), "Should retrieve historical baselines");
    let hist_data = history.unwrap();

    assert!(!hist_data.is_empty(), "Should have historical data");
    assert!(hist_data.len() <= 10, "Should return at most requested number of entries");

    // Check that history is ordered by date
    for i in 1..hist_data.len() {
        assert!(hist_data[i-1].established_date <= hist_data[i].established_date,
                "History should be chronologically ordered");
    }
}