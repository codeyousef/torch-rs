//! Optimizer module for Project Phoenix
//!
//! Enhanced optimizers with automatic parameter discovery and state management

#[cfg(feature = "torch-rs")]
mod phoenix_optimizer;
#[cfg(feature = "torch-rs")]
mod sgd;
#[cfg(feature = "torch-rs")]
mod adam;
#[cfg(feature = "torch-rs")]
mod adamw;
#[cfg(feature = "torch-rs")]
mod rmsprop;
#[cfg(feature = "torch-rs")]
mod adagrad;
#[cfg(feature = "torch-rs")]
mod lr_scheduler;

#[cfg(feature = "torch-rs")]
pub use phoenix_optimizer::*;
#[cfg(feature = "torch-rs")]
pub use sgd::*;
#[cfg(feature = "torch-rs")]
pub use adam::*;
#[cfg(feature = "torch-rs")]
pub use adamw::*;
#[cfg(feature = "torch-rs")]
pub use rmsprop::*;
#[cfg(feature = "torch-rs")]
pub use adagrad::*;
#[cfg(feature = "torch-rs")]
pub use lr_scheduler::*;