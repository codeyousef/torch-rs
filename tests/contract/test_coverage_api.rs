//! Contract tests for coverage reporting API
//!
//! Validates that the coverage API matches the OpenAPI specification

use tch::test_utils::{CoverageCollector, TestRunner, TestRunConfig};

#[test]
fn test_coverage_api_get_report() {
    // This test should fail until CoverageCollector is implemented
    let runner = TestRunner::new();
    let config = TestRunConfig::default();

    // Run tests to generate coverage
    let run = runner.run(config).expect("Test run should succeed");

    // Get coverage report
    let collector = CoverageCollector::new();
    let report = collector.get_report(&run.id);

    assert!(report.is_ok(), "Should retrieve coverage report");
    let coverage = report.unwrap();

    // Validate coverage report structure
    assert_eq!(coverage.test_run_id, run.id, "Report should match run ID");
    assert!(coverage.total_lines > 0, "Should have lines to cover");
    assert!(coverage.covered_lines <= coverage.total_lines, "Covered lines cannot exceed total");
    assert!(coverage.line_coverage_percent >= 0.0 && coverage.line_coverage_percent <= 100.0,
            "Coverage percentage must be between 0 and 100");
}

#[test]
fn test_coverage_api_module_breakdown() {
    let runner = TestRunner::new();
    let collector = CoverageCollector::new();

    // Run tests
    let run = runner.run(TestRunConfig::default()).unwrap();
    let report = collector.get_report(&run.id).unwrap();

    // Check module-level coverage
    assert!(!report.module_coverage.is_empty(), "Should have module coverage breakdown");

    for (module, coverage) in &report.module_coverage {
        assert!(!module.is_empty(), "Module name must not be empty");
        assert!(coverage.lines_covered <= coverage.lines_total,
                "Module covered lines cannot exceed total");

        let module_percent = (coverage.lines_covered as f32 / coverage.lines_total as f32) * 100.0;
        assert!(module_percent >= 0.0 && module_percent <= 100.0,
                "Module coverage must be valid percentage");
    }
}

#[test]
fn test_coverage_api_critical_paths() {
    let collector = CoverageCollector::new();
    let runner = TestRunner::new();

    // Run tests
    let run = runner.run(TestRunConfig::default()).unwrap();
    let report = collector.get_report(&run.id).unwrap();

    // Check critical path coverage
    assert!(report.critical_path_coverage >= 0.0 && report.critical_path_coverage <= 100.0,
            "Critical path coverage must be valid percentage");

    // Critical paths should have higher coverage requirements
    let critical_modules = ["tensor", "nn", "optim", "torch_data"];
    for module_name in &critical_modules {
        if let Some(coverage) = report.module_coverage.get(*module_name) {
            let module_percent = (coverage.lines_covered as f32 / coverage.lines_total as f32) * 100.0;
            assert!(module_percent >= 85.0,
                    "Critical module {} should have at least 85% coverage, got {:.1}%",
                    module_name, module_percent);
        }
    }
}

#[test]
fn test_coverage_api_branch_coverage() {
    let collector = CoverageCollector::new();
    let runner = TestRunner::new();

    let run = runner.run(TestRunConfig::default()).unwrap();
    let report = collector.get_report(&run.id).unwrap();

    // Validate branch coverage
    if report.total_branches > 0 {
        assert!(report.covered_branches <= report.total_branches,
                "Covered branches cannot exceed total");
        assert!(report.branch_coverage_percent >= 0.0 && report.branch_coverage_percent <= 100.0,
                "Branch coverage percentage must be valid");
    }
}

#[test]
fn test_coverage_api_function_coverage() {
    let collector = CoverageCollector::new();
    let runner = TestRunner::new();

    let run = runner.run(TestRunConfig::default()).unwrap();
    let report = collector.get_report(&run.id).unwrap();

    // Validate function coverage
    assert!(report.total_functions > 0, "Should have functions to cover");
    assert!(report.covered_functions <= report.total_functions,
            "Covered functions cannot exceed total");
    assert!(report.function_coverage_percent >= 0.0 && report.function_coverage_percent <= 100.0,
            "Function coverage percentage must be valid");
}

#[test]
fn test_coverage_api_threshold_check() {
    let collector = CoverageCollector::new();
    let runner = TestRunner::new();

    let run = runner.run(TestRunConfig::default()).unwrap();
    let report = collector.get_report(&run.id).unwrap();

    // Check against thresholds
    const OVERALL_THRESHOLD: f32 = 85.0;
    const CRITICAL_THRESHOLD: f32 = 95.0;

    let meets_overall = report.line_coverage_percent >= OVERALL_THRESHOLD;
    let meets_critical = report.critical_path_coverage >= CRITICAL_THRESHOLD;

    assert!(meets_overall || !meets_critical,
            "If overall coverage is below {:.0}%, critical paths cannot meet {:.0}%",
            OVERALL_THRESHOLD, CRITICAL_THRESHOLD);
}