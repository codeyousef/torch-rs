//! Phoenix Conv2d Layer Implementation
//!
//! Enhanced 2D convolution layer with automatic parameter discovery

#[cfg(feature = "torch-rs")]
pub mod conv2d {
    use crate::{Device, Kind, Tensor, nn::phoenix::PhoenixModule, nn::Module};
    use crate::nn::phoenix::PhoenixModuleError;
    use std::collections::HashMap;

    /// Phoenix Conv2d Layer
    ///
    /// Applies a 2D convolution over an input signal composed of several input planes.
    ///
    /// # Example
    /// ```rust
    /// use tch::nn::phoenix::Conv2d;
    /// use tch::{Device, Kind, Tensor};
    ///
    /// let layer = Conv2d::new(3, 64, 3); // 3 input channels, 64 output channels, 3x3 kernel
    /// let input = Tensor::randn(&[1, 3, 32, 32], (Kind::Float, Device::Cpu));
    /// let output = layer.forward(&input).unwrap();
    /// assert_eq!(output.size(), &[1, 64, 30, 30]); // With default padding=0
    /// ```
    #[derive(Debug)]
    pub struct Conv2d {
        /// Weight tensor of shape (out_channels, in_channels/groups, kernel_height, kernel_width)
        pub weight: Tensor,
        /// Optional bias tensor of shape (out_channels,)
        pub bias: Option<Tensor>,
        /// Number of input channels
        pub in_channels: i64,
        /// Number of output channels
        pub out_channels: i64,
        /// Kernel size (height, width)
        pub kernel_size: (i64, i64),
        /// Stride (height, width)
        pub stride: (i64, i64),
        /// Padding (height, width)
        pub padding: (i64, i64),
        /// Dilation (height, width)
        pub dilation: (i64, i64),
        /// Number of groups for grouped convolution
        pub groups: i64,
        /// Training mode
        training: bool,
    }

    impl Conv2d {
        /// Create a new Conv2d layer
        ///
        /// # Arguments
        /// * `in_channels` - Number of input channels
        /// * `out_channels` - Number of output channels
        /// * `kernel_size` - Size of the convolving kernel
        pub fn new(in_channels: i64, out_channels: i64, kernel_size: i64) -> Self {
            Self::new_with_config(ConvConfig {
                in_channels,
                out_channels,
                kernel_size: (kernel_size, kernel_size),
                stride: (1, 1),
                padding: (0, 0),
                dilation: (1, 1),
                groups: 1,
                bias: true,
            })
        }

        /// Create Conv2d with custom configuration
        pub fn new_with_config(config: ConvConfig) -> Self {
            let ConvConfig {
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias,
            } = config;

            // Validate parameters
            assert!(in_channels > 0, "in_channels must be positive");
            assert!(out_channels > 0, "out_channels must be positive");
            assert!(groups > 0, "groups must be positive");
            assert!(in_channels % groups == 0, "in_channels must be divisible by groups");
            assert!(out_channels % groups == 0, "out_channels must be divisible by groups");

            let weight_shape = [out_channels, in_channels / groups, kernel_size.0, kernel_size.1];
            let weight = Tensor::empty(&weight_shape, (Kind::Float, Device::Cpu))
                .set_requires_grad(true);

            // Initialize with Kaiming uniform (He initialization)
            let fan_in = (in_channels / groups) * kernel_size.0 * kernel_size.1;
            let bound = (1.0 / fan_in as f64).sqrt();
            let _ = weight.uniform_(-bound, bound);

            let bias_tensor = if bias {
                let bias_tensor = Tensor::empty(&[out_channels], (Kind::Float, Device::Cpu))
                    .set_requires_grad(true);
                let _ = bias_tensor.uniform_(-bound, bound);
                Some(bias_tensor)
            } else {
                None
            };

            Self {
                weight,
                bias: bias_tensor,
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                training: true,
            }
        }

        /// Reset parameters using Kaiming uniform initialization
        pub fn reset_parameters(&mut self) {
            let fan_in = (self.in_channels / self.groups) * self.kernel_size.0 * self.kernel_size.1;
            let bound = (1.0 / fan_in as f64).sqrt();
            let _ = self.weight.uniform_(-bound, bound);

            if let Some(ref bias) = self.bias {
                let _ = bias.uniform_(-bound, bound);
            }
        }

        /// Calculate output size given input size
        pub fn output_size(&self, input_size: (i64, i64)) -> (i64, i64) {
            let (h_in, w_in) = input_size;
            let (kh, kw) = self.kernel_size;
            let (sh, sw) = self.stride;
            let (ph, pw) = self.padding;
            let (dh, dw) = self.dilation;

            let h_out = (h_in + 2 * ph - dh * (kh - 1) - 1) / sh + 1;
            let w_out = (w_in + 2 * pw - dw * (kw - 1) - 1) / sw + 1;

            (h_out, w_out)
        }

        /// Check if layer has bias
        pub fn has_bias(&self) -> bool {
            self.bias.is_some()
        }

        /// Get receptive field size
        pub fn receptive_field(&self) -> (i64, i64) {
            let (kh, kw) = self.kernel_size;
            let (dh, dw) = self.dilation;
            (dh * (kh - 1) + 1, dw * (kw - 1) + 1)
        }
    }

    impl Module for Conv2d {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let input_shape = xs.size();

            // Validate input dimensions
            if input_shape.len() != 4 {
                panic!("Conv2d expects 4D input (N, C, H, W), got shape: {:?}", input_shape);
            }

            let input_channels = input_shape[1];
            if input_channels != self.in_channels {
                panic!("Input channels mismatch: expected {}, got {}",
                       self.in_channels, input_channels);
            }

            // Perform 2D convolution
            let output = xs.conv2d(
                &self.weight,
                self.bias.as_ref(),
                &[self.stride.0, self.stride.1],
                &[self.padding.0, self.padding.1],
                &[self.dilation.0, self.dilation.1],
                self.groups,
            );

            output
        }
    }

    impl PhoenixModule for Conv2d {
        fn parameters(&self) -> Vec<&Tensor> {
            let mut params = vec![&self.weight];
            if let Some(ref bias) = self.bias {
                params.push(bias);
            }
            params
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            let mut params = vec![&mut self.weight];
            if let Some(ref mut bias) = self.bias {
                params.push(bias);
            }
            params
        }

        fn named_parameters(&self) -> HashMap<String, &Tensor> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), &self.weight);
            if let Some(ref bias) = self.bias {
                params.insert("bias".to_string(), bias);
            }
            params
        }

        fn named_parameters_mut(&mut self) -> HashMap<String, &mut Tensor> {
            let mut params = HashMap::new();
            params.insert("weight".to_string(), &mut self.weight);
            if let Some(ref mut bias) = self.bias {
                params.insert("bias".to_string(), bias);
            }
            params
        }

        fn to_device(&mut self, device: Device) -> Result<(), PhoenixModuleError> {
            self.weight = self.weight.to_device(device);
            if let Some(ref bias) = self.bias {
                self.bias = Some(bias.to_device(device));
            }
            Ok(())
        }

        fn set_training(&mut self, training: bool) {
            self.training = training;
        }

        fn is_training(&self) -> bool {
            self.training
        }

        fn zero_grad(&mut self) {
            if let Some(grad) = self.weight.grad() {
                let _ = grad.zero_();
            }
            if let Some(ref bias) = self.bias {
                if let Some(grad) = bias.grad() {
                    let _ = grad.zero_();
                }
            }
        }

        fn device(&self) -> Option<Device> {
            let weight_device = self.weight.device();
            if let Some(ref bias) = self.bias {
                let bias_device = bias.device();
                if weight_device == bias_device {
                    Some(weight_device)
                } else {
                    None
                }
            } else {
                Some(weight_device)
            }
        }
    }

    /// Configuration for Conv2d layer
    #[derive(Debug, Clone)]
    pub struct ConvConfig {
        pub in_channels: i64,
        pub out_channels: i64,
        pub kernel_size: (i64, i64),
        pub stride: (i64, i64),
        pub padding: (i64, i64),
        pub dilation: (i64, i64),
        pub groups: i64,
        pub bias: bool,
    }

    /// Builder for Conv2d layers
    pub struct Conv2dBuilder {
        config: ConvConfig,
        device: Device,
        dtype: Kind,
    }

    impl Conv2dBuilder {
        pub fn new(in_channels: i64, out_channels: i64, kernel_size: i64) -> Self {
            Self {
                config: ConvConfig {
                    in_channels,
                    out_channels,
                    kernel_size: (kernel_size, kernel_size),
                    stride: (1, 1),
                    padding: (0, 0),
                    dilation: (1, 1),
                    groups: 1,
                    bias: true,
                },
                device: Device::Cpu,
                dtype: Kind::Float,
            }
        }

        pub fn kernel_size(mut self, kernel_size: (i64, i64)) -> Self {
            self.config.kernel_size = kernel_size;
            self
        }

        pub fn stride(mut self, stride: (i64, i64)) -> Self {
            self.config.stride = stride;
            self
        }

        pub fn padding(mut self, padding: (i64, i64)) -> Self {
            self.config.padding = padding;
            self
        }

        pub fn same_padding(mut self) -> Self {
            // Calculate padding for "SAME" convolution
            let (kh, kw) = self.config.kernel_size;
            let ph = (kh - 1) / 2;
            let pw = (kw - 1) / 2;
            self.config.padding = (ph, pw);
            self
        }

        pub fn dilation(mut self, dilation: (i64, i64)) -> Self {
            self.config.dilation = dilation;
            self
        }

        pub fn groups(mut self, groups: i64) -> Self {
            self.config.groups = groups;
            self
        }

        pub fn bias(mut self, bias: bool) -> Self {
            self.config.bias = bias;
            self
        }

        pub fn device(mut self, device: Device) -> Self {
            self.device = device;
            self
        }

        pub fn dtype(mut self, dtype: Kind) -> Self {
            self.dtype = dtype;
            self
        }

        pub fn build(self) -> Conv2d {
            let mut conv = Conv2d::new_with_config(self.config);
            let _ = conv.to_device(self.device);
            conv
        }
    }

    /// Depthwise separable convolution
    pub struct DepthwiseSeparableConv2d {
        depthwise: Conv2d,
        pointwise: Conv2d,
        training: bool,
    }

    impl DepthwiseSeparableConv2d {
        pub fn new(in_channels: i64, out_channels: i64, kernel_size: i64) -> Self {
            let depthwise = Conv2dBuilder::new(in_channels, in_channels, kernel_size)
                .groups(in_channels)
                .build();

            let pointwise = Conv2dBuilder::new(in_channels, out_channels, 1)
                .build();

            Self {
                depthwise,
                pointwise,
                training: true,
            }
        }
    }

    impl Module for DepthwiseSeparableConv2d {
        fn forward(&self, xs: &Tensor) -> Tensor {
            let depthwise_out = self.depthwise.forward(xs);
            self.pointwise.forward(&depthwise_out)
        }
    }

    crate::impl_phoenix_module!(DepthwiseSeparableConv2d {
        depthwise: Conv2d,
        pointwise: Conv2d,
    });

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{Kind, Device, Tensor};

        #[test]
        fn test_conv2d_creation() {
            let conv = Conv2d::new(3, 64, 3);

            assert_eq!(conv.in_channels, 3);
            assert_eq!(conv.out_channels, 64);
            assert_eq!(conv.kernel_size, (3, 3));
            assert_eq!(conv.stride, (1, 1));
            assert_eq!(conv.padding, (0, 0));
            assert!(conv.has_bias());
            assert_eq!(conv.weight.size(), &[64, 3, 3, 3]);
        }

        #[test]
        fn test_conv2d_forward() {
            let conv = Conv2d::new(3, 64, 3);
            let input = Tensor::randn(&[1, 3, 32, 32], (Kind::Float, Device::Cpu));

            let output = conv.forward(&input);
            assert_eq!(output.size(), &[1, 64, 30, 30]); // 32-3+1 = 30
        }

        #[test]
        fn test_conv2d_with_padding() {
            let conv = Conv2dBuilder::new(3, 64, 3)
                .same_padding()
                .build();

            let input = Tensor::randn(&[1, 3, 32, 32], (Kind::Float, Device::Cpu));
            let output = conv.forward(&input);
            assert_eq!(output.size(), &[1, 64, 32, 32]); // Same size with padding
        }

        #[test]
        fn test_conv2d_with_stride() {
            let conv = Conv2dBuilder::new(3, 64, 3)
                .stride((2, 2))
                .build();

            let input = Tensor::randn(&[1, 3, 32, 32], (Kind::Float, Device::Cpu));
            let output = conv.forward(&input);
            assert_eq!(output.size(), &[1, 64, 15, 15]); // (32-3)/2 + 1 = 15
        }

        #[test]
        fn test_output_size_calculation() {
            let conv = Conv2d::new(3, 64, 3);
            let output_size = conv.output_size((32, 32));
            assert_eq!(output_size, (30, 30));

            let conv_with_padding = Conv2dBuilder::new(3, 64, 3)
                .padding((1, 1))
                .build();
            let output_size = conv_with_padding.output_size((32, 32));
            assert_eq!(output_size, (32, 32));
        }

        #[test]
        fn test_grouped_convolution() {
            let conv = Conv2dBuilder::new(8, 16, 3)
                .groups(2)
                .build();

            assert_eq!(conv.groups, 2);
            assert_eq!(conv.weight.size(), &[16, 4, 3, 3]); // 8/2 = 4 channels per group

            let input = Tensor::randn(&[1, 8, 32, 32], (Kind::Float, Device::Cpu));
            let output = conv.forward(&input);
            assert_eq!(output.size(), &[1, 16, 30, 30]);
        }

        #[test]
        fn test_depthwise_separable() {
            let conv = DepthwiseSeparableConv2d::new(32, 64, 3);

            // Check parameter count vs regular conv
            let regular_conv = Conv2d::new(32, 64, 3);
            let ds_params = conv.num_parameters();
            let regular_params = regular_conv.num_parameters();

            // Depthwise separable should have fewer parameters
            assert!(ds_params < regular_params);

            let input = Tensor::randn(&[1, 32, 32, 32], (Kind::Float, Device::Cpu));
            let output = conv.forward(&input);
            assert_eq!(output.size(), &[1, 64, 30, 30]);
        }

        #[test]
        fn test_phoenix_module_implementation() {
            let mut conv = Conv2d::new(3, 64, 3);

            // Test parameters
            let params = conv.parameters();
            assert_eq!(params.len(), 2); // weight + bias

            let named_params = conv.named_parameters();
            assert_eq!(named_params.len(), 2);
            assert!(named_params.contains_key("weight"));
            assert!(named_params.contains_key("bias"));

            // Test device
            assert_eq!(conv.device(), Some(Device::Cpu));

            // Test training mode
            assert!(conv.is_training());
            conv.set_training(false);
            assert!(!conv.is_training());

            // Test parameter count
            let expected_params = 64 * 3 * 3 * 3 + 64; // weights + bias
            assert_eq!(conv.num_parameters(), expected_params);
        }

        #[test]
        #[should_panic(expected = "Conv2d expects 4D input")]
        fn test_invalid_input_dimensions() {
            let conv = Conv2d::new(3, 64, 3);
            let input = Tensor::randn(&[3, 32, 32], (Kind::Float, Device::Cpu)); // 3D instead of 4D
            let _ = conv.forward(&input);
        }

        #[test]
        #[should_panic(expected = "Input channels mismatch")]
        fn test_input_channels_mismatch() {
            let conv = Conv2d::new(3, 64, 3);
            let input = Tensor::randn(&[1, 6, 32, 32], (Kind::Float, Device::Cpu)); // 6 channels instead of 3
            let _ = conv.forward(&input);
        }

        #[test]
        fn test_receptive_field() {
            let conv = Conv2d::new(3, 64, 3);
            assert_eq!(conv.receptive_field(), (3, 3));

            let conv_dilated = Conv2dBuilder::new(3, 64, 3)
                .dilation((2, 2))
                .build();
            assert_eq!(conv_dilated.receptive_field(), (5, 5)); // 2*(3-1)+1 = 5
        }

        #[test]
        fn test_state_dict() {
            let conv = Conv2d::new(3, 64, 3);
            let state_dict = conv.state_dict();

            assert_eq!(state_dict.len(), 2);
            assert!(state_dict.contains_key("weight"));
            assert!(state_dict.contains_key("bias"));
            assert_eq!(state_dict["weight"].size(), &[64, 3, 3, 3]);
            assert_eq!(state_dict["bias"].size(), &[64]);
        }
    }
}

// Re-export when Phoenix feature is enabled
#[cfg(feature = "torch-rs")]
pub use conv2d::*;