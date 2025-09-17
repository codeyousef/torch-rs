//! Test utilities for torch-rs
//!
//! Provides testing infrastructure including assertions, fixtures, and harness

pub mod assertions;
pub mod baseline;
pub mod coverage;
pub mod fixtures;
pub mod harness;
pub mod memory;
pub mod runner;

// Re-export commonly used items
pub use assertions::{assert_shape_eq, assert_tensor_approx_eq, assert_tensor_eq};
pub use baseline::{BaselineTracker, PerformanceBaseline};
pub use coverage::{CoverageCollector, CoverageReport};
pub use fixtures::{create_simple_cnn, load_mnist_sample, FixtureBuilder, TestFixture};
pub use harness::{E2ETestHarness, TestContext};
pub use memory::{detect_memory_leak, track_memory, MemoryTracker};
pub use runner::{TestRunConfig, TestRunner};
