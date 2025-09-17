//! A small neural-network library based on Torch.
//!
//! This library tries to stay as close as possible to the original
//! Python and C++ implementations.
pub mod init;
pub use init::{f_init, init, Init};

mod var_store;
pub use var_store::{Path, VarStore, Variables};

mod module;
pub use module::{Module, ModuleT};

// Project Phoenix module system
#[cfg(feature = "torch-rs")]
pub mod derive;
#[cfg(feature = "torch-rs")]
pub mod phoenix;
#[cfg(feature = "torch-rs")]
pub mod phoenix_batch_norm;
#[cfg(feature = "torch-rs")]
pub mod phoenix_conv;
#[cfg(feature = "torch-rs")]
pub mod phoenix_dropout;
#[cfg(feature = "torch-rs")]
pub mod phoenix_linear;
#[cfg(feature = "torch-rs")]
pub mod phoenix_sequential;

// Re-export Phoenix types with renamed exports to avoid conflicts
#[cfg(feature = "torch-rs")]
pub use phoenix::{PhoenixModule, PhoenixModuleError};
#[cfg(feature = "torch-rs")]
pub use phoenix_batch_norm::batch_norm::{
    BatchNorm1d as PhoenixBatchNorm1d, BatchNorm2d as PhoenixBatchNorm2d, PhoenixBatchNormConfig,
};
#[cfg(feature = "torch-rs")]
pub use phoenix_conv::conv2d::{Conv2d as PhoenixConv2d, PhoenixConvConfig};
#[cfg(feature = "torch-rs")]
pub use phoenix_dropout::Dropout as PhoenixDropout;
#[cfg(feature = "torch-rs")]
pub use phoenix_linear::linear::Linear as PhoenixLinear;
#[cfg(feature = "torch-rs")]
pub use phoenix_sequential::Sequential as PhoenixSequential;

#[cfg(feature = "torch-rs")]
mod trainer;
#[cfg(feature = "torch-rs")]
pub use trainer::*;

#[cfg(feature = "torch-rs")]
mod metrics;
#[cfg(feature = "torch-rs")]
pub use metrics::*;

mod linear;
pub use linear::*;

mod conv;
pub use conv::*;

mod conv_transpose;
pub use conv_transpose::*;

mod batch_norm;
pub use batch_norm::*;

mod group_norm;
pub use group_norm::*;

mod layer_norm;
pub use layer_norm::*;

mod sparse;
pub use sparse::*;

mod rnn;
pub use rnn::*;

mod func;
pub use func::*;

mod sequential;
pub use sequential::*;

mod optimizer;
pub use optimizer::{
    adam, adamw, rms_prop, sgd, Adam, AdamW, Optimizer, OptimizerConfig, OptimizerValue, RmsProp,
    Sgd,
};

/// An identity layer. This just propagates its tensor input as output.
#[derive(Debug)]
pub struct Id();

impl ModuleT for Id {
    fn forward_t(&self, xs: &crate::Tensor, _train: bool) -> crate::Tensor {
        xs.shallow_clone()
    }
}

impl Module for crate::CModule {
    fn forward(&self, xs: &crate::Tensor) -> crate::Tensor {
        self.forward_ts(&[xs]).unwrap()
    }
}

impl ModuleT for crate::TrainableCModule {
    fn forward_t(&self, xs: &crate::Tensor, _train: bool) -> crate::Tensor {
        self.inner.forward_ts(&[xs]).unwrap()
    }
}
