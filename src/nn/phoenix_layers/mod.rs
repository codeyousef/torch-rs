//! Phoenix layers module - advanced neural network layers

#[cfg(feature = "torch-rs")]
pub mod lstm;
#[cfg(feature = "torch-rs")]
pub mod gru;
#[cfg(feature = "torch-rs")]
pub mod attention;
#[cfg(feature = "torch-rs")]
pub mod transformer;
#[cfg(feature = "torch-rs")]
pub mod positional;

#[cfg(feature = "torch-rs")]
pub use lstm::LSTM;
#[cfg(feature = "torch-rs")]
pub use gru::GRU;
#[cfg(feature = "torch-rs")]
pub use attention::MultiheadAttention;
#[cfg(feature = "torch-rs")]
pub use transformer::TransformerEncoderLayer;
#[cfg(feature = "torch-rs")]
pub use positional::PositionalEncoding;