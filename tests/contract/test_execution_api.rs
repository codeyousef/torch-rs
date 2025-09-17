//! Contract tests for test execution API
//!
//! Validates that the test execution API matches the OpenAPI specification

use tch::test_utils::{TestRunner, TestRunConfig};
use std::time::Duration;

#[test]
fn test_execution_api_run_tests() {
    // This test should fail until TestRunner is implemented
    let config = TestRunConfig {
        suites: vec!["unit".to_string(), "integration".to_string()],
        parallel: true,
        timeout: Duration::from_secs(300),
    };

    let runner = TestRunner::new();
    let result = runner.run(config);

    // Validate response matches contract
    assert!(result.is_ok(), "Test execution should succeed");
    let run = result.unwrap();

    assert!(!run.id.is_empty(), "Run ID must not be empty");
    assert!(!run.commit_hash.is_empty(), "Commit hash must be present");
    assert!(!run.branch.is_empty(), "Branch name must be present");
    assert!(run.total_tests > 0, "Must execute at least one test");
    assert!(run.start_time <= run.end_time.unwrap(), "End time must be after start time");
}

#[test]
fn test_execution_api_get_results() {
    let runner = TestRunner::new();

    // Run tests first
    let config = TestRunConfig::default();
    let run = runner.run(config).expect("Test run should succeed");

    // Get results by run ID
    let results = runner.get_results(&run.id, None);

    assert!(results.is_ok(), "Should retrieve test results");
    let results = results.unwrap();

    assert!(!results.is_empty(), "Should have test results");
    for result in &results {
        assert!(!result.test_id.is_empty(), "Test ID must not be empty");
        assert!(!result.execution_id.is_empty(), "Execution ID must not be empty");
        assert!(result.duration_ms > 0, "Test duration must be positive");
    }
}

#[test]
fn test_execution_api_filter_by_status() {
    let runner = TestRunner::new();

    // Run tests
    let config = TestRunConfig::default();
    let run = runner.run(config).expect("Test run should succeed");

    // Filter by status
    let failed_results = runner.get_results(&run.id, Some("fail".to_string()));

    assert!(failed_results.is_ok(), "Should filter results by status");
    let failed = failed_results.unwrap();

    for result in &failed {
        assert_eq!(result.status, "fail", "Should only return failed tests");
    }
}

#[test]
fn test_execution_api_parallel_execution() {
    let config = TestRunConfig {
        suites: vec!["unit".to_string(), "integration".to_string(), "e2e".to_string()],
        parallel: true,
        timeout: Duration::from_secs(300),
    };

    let runner = TestRunner::new();
    let start = std::time::Instant::now();
    let result = runner.run(config);
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Parallel execution should succeed");
    let run = result.unwrap();

    // Parallel execution should be faster than sequential
    let sequential_estimate = run.total_duration_ms;
    let actual_duration = elapsed.as_millis() as u64;

    assert!(
        actual_duration < sequential_estimate * 0.7,
        "Parallel execution should be significantly faster than sequential"
    );
}

#[test]
fn test_execution_api_timeout_handling() {
    let config = TestRunConfig {
        suites: vec!["slow_tests".to_string()],
        parallel: false,
        timeout: Duration::from_millis(100), // Very short timeout
    };

    let runner = TestRunner::new();
    let result = runner.run(config);

    assert!(result.is_ok(), "Should handle timeouts gracefully");
    let run = result.unwrap();

    // Check that some tests timed out
    let results = runner.get_results(&run.id, Some("timeout".to_string())).unwrap();
    assert!(!results.is_empty(), "Should have timed out tests");
}