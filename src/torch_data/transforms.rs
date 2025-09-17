//! Transform System for Project Phoenix
//!
//! Provides data preprocessing, augmentation, and transformation capabilities

#[cfg(feature = "torch-rs")]
pub mod transforms {
    use crate::Tensor;
    use super::super::dataset::TransformError;

    /// Core trait for all data transforms
    ///
    /// Transforms are composable operations that modify tensors, typically used
    /// for data preprocessing and augmentation in machine learning pipelines.
    pub trait Transform: Send + Sync {
        /// Apply transformation to input data
        ///
        /// # Arguments
        /// * `input` - Input data (typically a tensor)
        ///
        /// # Returns
        /// * `Result<Tensor, TransformError>` - Transformed data or error
        ///
        /// # Contract
        /// - Must preserve data type consistency where expected
        /// - Must handle edge cases (empty tensors, unusual shapes)
        /// - Must be deterministic for same input (unless specifically random)
        fn apply(&self, input: Tensor) -> Result<Tensor, TransformError>;

        /// Get transform name for debugging/logging
        fn name(&self) -> &str;

        /// Check if transform is deterministic
        fn is_deterministic(&self) -> bool;

        /// Check if transform supports in-place operation
        fn supports_inplace(&self) -> bool {
            false
        }

        /// Apply transform in-place if supported
        fn apply_inplace(&self, input: &mut Tensor) -> Result<(), TransformError> {
            if !self.supports_inplace() {
                return Err(TransformError::UnsupportedOperation {
                    operation: "in-place transform".to_string(),
                });
            }
            let result = self.apply(input.copy())?;
            input.copy_(&result);
            Ok(())
        }

        /// Get expected input shape requirements
        fn input_requirements(&self) -> Option<ShapeRequirement> {
            None
        }

        /// Get expected output shape
        fn output_shape(&self, input_shape: &[i64]) -> Result<Vec<i64>, TransformError> {
            // Default: preserve input shape
            Ok(input_shape.to_vec())
        }
    }

    /// Shape requirements for transforms
    #[derive(Debug, Clone)]
    pub enum ShapeRequirement {
        /// Exact shape required
        Exact(Vec<i64>),
        /// Minimum number of dimensions
        MinDims(usize),
        /// Maximum number of dimensions
        MaxDims(usize),
        /// Range of dimensions
        DimRange(usize, usize),
        /// Custom validation function
        Custom(fn(&[i64]) -> bool),
    }

    impl ShapeRequirement {
        pub fn validate(&self, shape: &[i64]) -> bool {
            match self {
                ShapeRequirement::Exact(expected) => shape == expected.as_slice(),
                ShapeRequirement::MinDims(min) => shape.len() >= *min,
                ShapeRequirement::MaxDims(max) => shape.len() <= *max,
                ShapeRequirement::DimRange(min, max) => shape.len() >= *min && shape.len() <= *max,
                ShapeRequirement::Custom(validator) => validator(shape),
            }
        }
    }

    /// Compose multiple transforms into a pipeline
    pub struct Compose {
        transforms: Vec<Box<dyn Transform>>,
        name: String,
    }

    impl Compose {
        pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
            let names: Vec<_> = transforms.iter().map(|t| t.name()).collect();
            let name = format!("Compose({})", names.join(" -> "));

            Self { transforms, name }
        }

        pub fn add(&mut self, transform: Box<dyn Transform>) {
            self.transforms.push(transform);
            // Update name
            let names: Vec<_> = self.transforms.iter().map(|t| t.name()).collect();
            self.name = format!("Compose({})", names.join(" -> "));
        }

        pub fn len(&self) -> usize {
            self.transforms.len()
        }

        pub fn is_empty(&self) -> bool {
            self.transforms.is_empty()
        }
    }

    impl Transform for Compose {
        fn apply(&self, mut input: Tensor) -> Result<Tensor, TransformError> {
            for transform in &self.transforms {
                input = transform.apply(input)?;
            }
            Ok(input)
        }

        fn name(&self) -> &str {
            &self.name
        }

        fn is_deterministic(&self) -> bool {
            self.transforms.iter().all(|t| t.is_deterministic())
        }

        fn output_shape(&self, input_shape: &[i64]) -> Result<Vec<i64>, TransformError> {
            let mut shape = input_shape.to_vec();
            for transform in &self.transforms {
                shape = transform.output_shape(&shape)?;
            }
            Ok(shape)
        }
    }

    /// Convert data to tensor format
    pub struct ToTensor {
        dtype: Option<crate::Kind>,
    }

    impl ToTensor {
        pub fn new() -> Self {
            Self { dtype: None }
        }

        pub fn with_dtype(dtype: crate::Kind) -> Self {
            Self { dtype: Some(dtype) }
        }
    }

    impl Default for ToTensor {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Transform for ToTensor {
        fn apply(&self, input: Tensor) -> Result<Tensor, TransformError> {
            let result = if let Some(dtype) = self.dtype {
                input.to_kind(dtype)
            } else {
                input.to_kind(crate::Kind::Float)
            };
            Ok(result)
        }

        fn name(&self) -> &str {
            "ToTensor"
        }

        fn is_deterministic(&self) -> bool {
            true
        }
    }

    /// Normalize tensor with mean and standard deviation
    pub struct Normalize {
        mean: Vec<f64>,
        std: Vec<f64>,
    }

    impl Normalize {
        pub fn new(mean: Vec<f64>, std: Vec<f64>) -> Result<Self, TransformError> {
            if mean.len() != std.len() {
                return Err(TransformError::ParameterOutOfRange {
                    param: "mean/std length".to_string(),
                    value: mean.len() as f64,
                    range: format!("must match std length: {}", std.len()),
                });
            }

            if std.iter().any(|&s| s <= 0.0) {
                return Err(TransformError::ParameterOutOfRange {
                    param: "std".to_string(),
                    value: 0.0,
                    range: "must be positive".to_string(),
                });
            }

            Ok(Self { mean, std })
        }

        pub fn imagenet() -> Result<Self, TransformError> {
            Self::new(
                vec![0.485, 0.456, 0.406],
                vec![0.229, 0.224, 0.225],
            )
        }
    }

    impl Transform for Normalize {
        fn apply(&self, input: Tensor) -> Result<Tensor, TransformError> {
            let shape = input.size();

            if shape.is_empty() {
                return Err(TransformError::ShapeMismatch {
                    expected: vec![-1], // At least 1D
                    actual: shape,
                });
            }

            // Handle different tensor layouts
            let normalized = if shape.len() == 3 && shape[0] as usize == self.mean.len() {
                // CHW format: apply per-channel normalization
                let mut result = input.shallow_clone();
                for (c, (&mean, &std)) in self.mean.iter().zip(self.std.iter()).enumerate() {
                    let channel = result.select(0, c as i64);
                    let normalized_channel = (channel - mean) / std;
                    // Use narrow and copy_ to set the slice
                    let _ = result.narrow(0, c as i64, 1).copy_(&normalized_channel.unsqueeze(0));
                }
                result
            } else if shape.len() == 4 && shape[1] as usize == self.mean.len() {
                // NCHW format: apply per-channel normalization
                let mut result = input.shallow_clone();
                for (c, (&mean, &std)) in self.mean.iter().zip(self.std.iter()).enumerate() {
                    let channel = result.select(1, c as i64);
                    let normalized_channel = (channel - mean) / std;
                    // Use narrow and copy_ to set the slice
                    let _ = result.narrow(1, c as i64, 1).copy_(&normalized_channel.unsqueeze(1));
                }
                result
            } else {
                // Global normalization with first mean/std values
                (input - self.mean[0]) / self.std[0]
            };

            Ok(normalized)
        }

        fn name(&self) -> &str {
            "Normalize"
        }

        fn is_deterministic(&self) -> bool {
            true
        }

        fn input_requirements(&self) -> Option<ShapeRequirement> {
            Some(ShapeRequirement::MinDims(1))
        }
    }

    /// Resize tensor to specified dimensions
    pub struct Resize {
        size: Vec<i64>,
        interpolation: InterpolationMode,
    }

    #[derive(Debug, Clone, Copy)]
    pub enum InterpolationMode {
        Nearest,
        Linear,
        Bilinear,
        Trilinear,
        Area,
        Bicubic,
    }

    impl Resize {
        pub fn new(size: Vec<i64>) -> Self {
            Self {
                size,
                interpolation: InterpolationMode::Bilinear,
            }
        }

        pub fn with_interpolation(size: Vec<i64>, interpolation: InterpolationMode) -> Self {
            Self { size, interpolation }
        }

        pub fn square(size: i64) -> Self {
            Self::new(vec![size, size])
        }
    }

    impl Transform for Resize {
        fn apply(&self, input: Tensor) -> Result<Tensor, TransformError> {
            let input_shape = input.size();

            if input_shape.len() < 2 {
                return Err(TransformError::ShapeMismatch {
                    expected: vec![-1, -1], // At least 2D
                    actual: input_shape,
                });
            }

            // For now, just return a tensor with the target shape
            // Real implementation would use interpolation
            let mut target_shape = input_shape.clone();
            let spatial_dims = target_shape.len().min(self.size.len());

            for i in 0..spatial_dims {
                target_shape[target_shape.len() - spatial_dims + i] = self.size[i];
            }

            // Placeholder: return zeros with target shape
            // Real implementation would perform actual resizing
            Ok(Tensor::zeros(&target_shape, (input.kind(), input.device())))
        }

        fn name(&self) -> &str {
            "Resize"
        }

        fn is_deterministic(&self) -> bool {
            true
        }

        fn output_shape(&self, input_shape: &[i64]) -> Result<Vec<i64>, TransformError> {
            if input_shape.len() < 2 {
                return Err(TransformError::ShapeMismatch {
                    expected: vec![-1, -1],
                    actual: input_shape.to_vec(),
                });
            }

            let mut output_shape = input_shape.to_vec();
            let spatial_dims = output_shape.len().min(self.size.len());

            for i in 0..spatial_dims {
                output_shape[output_shape.len() - spatial_dims + i] = self.size[i];
            }

            Ok(output_shape)
        }
    }

    /// Center crop to specified size
    pub struct CenterCrop {
        size: Vec<i64>,
    }

    impl CenterCrop {
        pub fn new(size: Vec<i64>) -> Self {
            Self { size }
        }

        pub fn square(size: i64) -> Self {
            Self::new(vec![size, size])
        }
    }

    impl Transform for CenterCrop {
        fn apply(&self, input: Tensor) -> Result<Tensor, TransformError> {
            let input_shape = input.size();

            if input_shape.len() < 2 {
                return Err(TransformError::ShapeMismatch {
                    expected: vec![-1, -1],
                    actual: input_shape,
                });
            }

            // Calculate crop coordinates
            let spatial_dims = input_shape.len().min(self.size.len());
            let mut starts = vec![0i64; input_shape.len()];
            let mut ends = input_shape.clone();

            for i in 0..spatial_dims {
                let dim_idx = input_shape.len() - spatial_dims + i;
                let input_size = input_shape[dim_idx];
                let target_size = self.size[i];

                if target_size > input_size {
                    return Err(TransformError::ParameterOutOfRange {
                        param: format!("crop_size[{}]", i),
                        value: target_size as f64,
                        range: format!("must be <= input_size[{}] = {}", dim_idx, input_size),
                    });
                }

                let start = (input_size - target_size) / 2;
                starts[dim_idx] = start;
                ends[dim_idx] = start + target_size;
            }

            // Perform the crop using narrow operations
            let mut result = input;
            for (dim, (&start, &end)) in starts.iter().zip(ends.iter()).enumerate() {
                if end - start != input_shape[dim] {
                    result = result.narrow(dim as i64, start, end - start);
                }
            }

            Ok(result)
        }

        fn name(&self) -> &str {
            "CenterCrop"
        }

        fn is_deterministic(&self) -> bool {
            true
        }

        fn output_shape(&self, input_shape: &[i64]) -> Result<Vec<i64>, TransformError> {
            if input_shape.len() < 2 {
                return Err(TransformError::ShapeMismatch {
                    expected: vec![-1, -1],
                    actual: input_shape.to_vec(),
                });
            }

            let mut output_shape = input_shape.to_vec();
            let spatial_dims = output_shape.len().min(self.size.len());

            for i in 0..spatial_dims {
                let dim_idx = output_shape.len() - spatial_dims + i;
                output_shape[dim_idx] = self.size[i];
            }

            Ok(output_shape)
        }
    }

    /// Random horizontal flip
    pub struct RandomHorizontalFlip {
        probability: f64,
    }

    impl RandomHorizontalFlip {
        pub fn new(probability: f64) -> Result<Self, TransformError> {
            if !(0.0..=1.0).contains(&probability) {
                return Err(TransformError::ParameterOutOfRange {
                    param: "probability".to_string(),
                    value: probability,
                    range: "[0.0, 1.0]".to_string(),
                });
            }
            Ok(Self { probability })
        }
    }

    impl Transform for RandomHorizontalFlip {
        fn apply(&self, input: Tensor) -> Result<Tensor, TransformError> {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            if rng.gen::<f64>() < self.probability {
                // Flip horizontally (assume last dimension is width)
                let dims = input.size().len() as i64;
                if dims >= 1 {
                    Ok(input.flip(&[dims - 1]))
                } else {
                    Ok(input)
                }
            } else {
                Ok(input)
            }
        }

        fn name(&self) -> &str {
            "RandomHorizontalFlip"
        }

        fn is_deterministic(&self) -> bool {
            false
        }
    }

    /// Convert tensor values to specified range
    pub struct Rescale {
        scale: f64,
        offset: f64,
    }

    impl Rescale {
        /// Scale from [0, 1] to [0, 255] (uint8 range)
        pub fn to_uint8() -> Self {
            Self {
                scale: 255.0,
                offset: 0.0,
            }
        }

        /// Scale from [0, 255] to [0, 1] (float range)
        pub fn to_float() -> Self {
            Self {
                scale: 1.0 / 255.0,
                offset: 0.0,
            }
        }

        /// Custom scale and offset
        pub fn new(scale: f64, offset: f64) -> Self {
            Self { scale, offset }
        }
    }

    impl Transform for Rescale {
        fn apply(&self, input: Tensor) -> Result<Tensor, TransformError> {
            Ok(input * self.scale + self.offset)
        }

        fn name(&self) -> &str {
            "Rescale"
        }

        fn is_deterministic(&self) -> bool {
            true
        }
    }

    /// Transform builder for chaining operations
    pub struct TransformBuilder {
        transforms: Vec<Box<dyn Transform>>,
    }

    impl TransformBuilder {
        pub fn new() -> Self {
            Self {
                transforms: Vec::new(),
            }
        }

        pub fn resize(mut self, size: Vec<i64>) -> Self {
            self.transforms.push(Box::new(Resize::new(size)));
            self
        }

        pub fn center_crop(mut self, size: Vec<i64>) -> Self {
            self.transforms.push(Box::new(CenterCrop::new(size)));
            self
        }

        pub fn to_tensor(mut self) -> Self {
            self.transforms.push(Box::new(ToTensor::new()));
            self
        }

        pub fn normalize(mut self, mean: Vec<f64>, std: Vec<f64>) -> Result<Self, TransformError> {
            self.transforms.push(Box::new(Normalize::new(mean, std)?));
            Ok(self)
        }

        pub fn random_horizontal_flip(mut self, p: f64) -> Result<Self, TransformError> {
            self.transforms.push(Box::new(RandomHorizontalFlip::new(p)?));
            Ok(self)
        }

        pub fn rescale(mut self, scale: f64, offset: f64) -> Self {
            self.transforms.push(Box::new(Rescale::new(scale, offset)));
            self
        }

        pub fn add_custom(mut self, transform: Box<dyn Transform>) -> Self {
            self.transforms.push(transform);
            self
        }

        pub fn build(self) -> Compose {
            Compose::new(self.transforms)
        }
    }

    impl Default for TransformBuilder {
        fn default() -> Self {
            Self::new()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Kind, Device};

        #[test]
        fn test_to_tensor() {
            let transform = ToTensor::new();
            let input = Tensor::randn(&[3, 32, 32], (Kind::Float, Device::Cpu));

            let result = transform.apply(input.copy());
            assert!(result.is_ok());

            let output = result.unwrap();
            assert_eq!(output.kind(), Kind::Float);
            assert_eq!(output.size(), &[3, 32, 32]);
        }

        #[test]
        fn test_normalize() {
            let mean = vec![0.5, 0.5, 0.5];
            let std = vec![0.2, 0.2, 0.2];
            let transform = Normalize::new(mean, std).unwrap();

            let input = Tensor::ones(&[3, 32, 32], (Kind::Float, Device::Cpu));
            let result = transform.apply(input);
            assert!(result.is_ok());

            // Normalized value should be (1.0 - 0.5) / 0.2 = 2.5 for all channels
            let output = result.unwrap();
            assert_eq!(output.size(), &[3, 32, 32]);
        }

        #[test]
        fn test_normalize_invalid_params() {
            // Different lengths
            let result = Normalize::new(vec![0.5], vec![0.2, 0.3]);
            assert!(result.is_err());

            // Zero std
            let result = Normalize::new(vec![0.5], vec![0.0]);
            assert!(result.is_err());
        }

        #[test]
        fn test_resize() {
            let transform = Resize::new(vec![64, 64]);
            let input = Tensor::randn(&[3, 32, 32], (Kind::Float, Device::Cpu));

            let output_shape = transform.output_shape(&input.size());
            assert!(output_shape.is_ok());
            assert_eq!(output_shape.unwrap(), vec![3, 64, 64]);
        }

        #[test]
        fn test_center_crop() {
            let transform = CenterCrop::new(vec![16, 16]);
            let input = Tensor::randn(&[3, 32, 32], (Kind::Float, Device::Cpu));

            let result = transform.apply(input);
            assert!(result.is_ok());

            let output = result.unwrap();
            assert_eq!(output.size(), &[3, 16, 16]);
        }

        #[test]
        fn test_center_crop_too_large() {
            let transform = CenterCrop::new(vec![64, 64]);
            let input = Tensor::randn(&[3, 32, 32], (Kind::Float, Device::Cpu));

            let result = transform.apply(input);
            assert!(result.is_err());
        }

        #[test]
        fn test_compose() {
            let transforms: Vec<Box<dyn Transform>> = vec![
                Box::new(Resize::new(vec![64, 64])),
                Box::new(CenterCrop::new(vec![32, 32])),
                Box::new(ToTensor::new()),
            ];

            let compose = Compose::new(transforms);
            let input = Tensor::randn(&[3, 128, 128], (Kind::Float, Device::Cpu));

            assert_eq!(compose.name(), "Compose(Resize -> CenterCrop -> ToTensor)");
            assert!(compose.is_deterministic());

            let output_shape = compose.output_shape(&input.size());
            assert!(output_shape.is_ok());
            assert_eq!(output_shape.unwrap(), vec![3, 32, 32]);
        }

        #[test]
        fn test_transform_builder() {
            let compose = TransformBuilder::new()
                .resize(vec![256, 256])
                .center_crop(vec![224, 224])
                .to_tensor()
                .normalize(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
                .unwrap()
                .build();

            assert_eq!(compose.len(), 4);
            assert!(compose.is_deterministic());
        }

        #[test]
        fn test_random_horizontal_flip() {
            let transform = RandomHorizontalFlip::new(0.5).unwrap();
            let input = Tensor::randn(&[3, 32, 32], (Kind::Float, Device::Cpu));

            // Test multiple times since it's random
            for _ in 0..10 {
                let result = transform.apply(input.copy());
                assert!(result.is_ok());
                let output = result.unwrap();
                assert_eq!(output.size(), input.size());
            }

            assert!(!transform.is_deterministic());
        }

        #[test]
        fn test_rescale() {
            let transform = Rescale::to_float();
            let input = Tensor::full(&[3, 32, 32], 255.0, (Kind::Float, Device::Cpu));

            let result = transform.apply(input);
            assert!(result.is_ok());

            let output = result.unwrap();
            let max_val = output.max().double_value(&[]);
            assert!((max_val - 1.0).abs() < 1e-6);
        }

        #[test]
        fn test_shape_requirement() {
            let req = ShapeRequirement::MinDims(2);
            assert!(req.validate(&[32, 32]));
            assert!(req.validate(&[3, 32, 32]));
            assert!(!req.validate(&[32]));

            let req = ShapeRequirement::Exact(vec![3, 224, 224]);
            assert!(req.validate(&[3, 224, 224]));
            assert!(!req.validate(&[1, 224, 224]));
        }
    }
}

// Re-export Phoenix transform functionality when feature is enabled
#[cfg(feature = "torch-rs")]
pub use transforms::*;