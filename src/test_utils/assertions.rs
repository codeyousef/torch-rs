//! Test assertion utilities for tensor comparisons

use crate::{Kind, Tensor};

/// Assert two tensors are exactly equal
pub fn assert_tensor_eq(actual: &Tensor, expected: &Tensor) {
    assert_eq!(
        actual.size(),
        expected.size(),
        "Tensor shapes don't match: {:?} vs {:?}",
        actual.size(),
        expected.size()
    );

    let diff = (actual - expected).abs().max();
    assert_eq!(
        f64::try_from(diff).unwrap(),
        0.0,
        "Tensors are not equal"
    );
}

/// Assert two tensors are approximately equal within tolerance
pub fn assert_tensor_approx_eq(actual: &Tensor, expected: &Tensor, tolerance: f64) {
    assert_eq!(
        actual.size(),
        expected.size(),
        "Tensor shapes don't match: {:?} vs {:?}",
        actual.size(),
        expected.size()
    );

    let diff = (actual - expected).abs().max();
    let max_diff = f64::try_from(diff).unwrap();

    assert!(
        max_diff <= tolerance,
        "Tensors differ by {} (tolerance: {})",
        max_diff,
        tolerance
    );
}

/// Assert tensor shapes are equal
pub fn assert_shape_eq(actual: &[i64], expected: &[i64]) {
    assert_eq!(
        actual,
        expected,
        "Shapes don't match: {:?} vs {:?}",
        actual,
        expected
    );
}