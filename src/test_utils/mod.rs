//! Test utilities for torch-rs
//!
//! Provides testing infrastructure including assertions, fixtures, and harness

pub mod assertions;
pub mod fixtures;
pub mod harness;
pub mod memory;
pub mod runner;
pub mod coverage;
pub mod baseline;

// Re-export commonly used items
pub use assertions::{assert_tensor_eq, assert_tensor_approx_eq, assert_shape_eq};
pub use fixtures::{TestFixture, FixtureBuilder, load_mnist_sample, create_simple_cnn};
pub use harness::{E2ETestHarness, TestContext};
pub use memory::{track_memory, detect_memory_leak, MemoryTracker};
pub use runner::{TestRunner, TestRunConfig};
pub use coverage::{CoverageCollector, CoverageReport};
pub use baseline::{PerformanceBaseline, BaselineTracker};