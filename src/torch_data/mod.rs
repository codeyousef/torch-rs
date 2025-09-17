//! Data module for torch-rs
//!
//! Provides dataset loading, transforms, and data utilities

// Phoenix-specific data functionality
#[cfg(feature = "torch-rs")]
mod cifar10;
#[cfg(feature = "torch-rs")]
mod dataloader;
#[cfg(feature = "torch-rs")]
mod dataset;
#[cfg(feature = "torch-rs")]
mod mnist;
#[cfg(feature = "torch-rs")]
mod transforms;

#[cfg(feature = "torch-rs")]
pub use cifar10::*;
#[cfg(feature = "torch-rs")]
pub use dataloader::*;
#[cfg(feature = "torch-rs")]
pub use dataset::*;
#[cfg(feature = "torch-rs")]
pub use mnist::*;
#[cfg(feature = "torch-rs")]
pub use transforms::*;
