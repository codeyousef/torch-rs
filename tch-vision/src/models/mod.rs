//! Pre-trained model implementations for computer vision
//!
//! This module provides implementations of popular computer vision models
//! with support for loading pre-trained weights.

pub mod resnet;
pub mod vgg;
pub mod vit;

pub use resnet::*;
pub use vgg::*;
pub use vit::*;