//! Image transformations for data preprocessing and augmentation
//!
//! Provides common computer vision transformations compatible with PyTorch

use tch::{Device, Kind, Tensor};
use std::f64::consts::PI;

/// Normalize a tensor with mean and standard deviation
pub struct Normalize {
    mean: Vec<f64>,
    std: Vec<f64>,
}

impl Normalize {
    pub fn new(mean: Vec<f64>, std: Vec<f64>) -> Self {
        assert_eq!(mean.len(), std.len(), "Mean and std must have same length");
        Self { mean, std }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let device = tensor.device();
        let dims = tensor.dim();
        
        // Assume tensor is CHW or NCHW format
        let channels_dim = if dims == 4 { 1 } else { 0 };
        
        let mean = Tensor::of_slice(&self.mean)
            .to_device(device)
            .view([-1, 1, 1]);
        let std = Tensor::of_slice(&self.std)
            .to_device(device)
            .view([-1, 1, 1]);
        
        (tensor - mean) / std
    }
}

/// Random horizontal flip transformation
pub struct RandomHorizontalFlip {
    p: f64,
}

impl RandomHorizontalFlip {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p <= 1.0, "Probability must be in [0, 1]");
        Self { p }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        if Tensor::rand(&[], (Kind::Float, tensor.device())).double_value(&[]) < self.p {
            tensor.flip(&[-1])  // Flip along width dimension
        } else {
            tensor.shallow_clone()
        }
    }
}

/// Random vertical flip transformation
pub struct RandomVerticalFlip {
    p: f64,
}

impl RandomVerticalFlip {
    pub fn new(p: f64) -> Self {
        assert!(p >= 0.0 && p <= 1.0, "Probability must be in [0, 1]");
        Self { p }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        if Tensor::rand(&[], (Kind::Float, tensor.device())).double_value(&[]) < self.p {
            tensor.flip(&[-2])  // Flip along height dimension
        } else {
            tensor.shallow_clone()
        }
    }
}

/// Random crop transformation
pub struct RandomCrop {
    size: (i64, i64),
    padding: Option<i64>,
}

impl RandomCrop {
    pub fn new(size: (i64, i64), padding: Option<i64>) -> Self {
        Self { size, padding }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let mut input = tensor.shallow_clone();
        
        // Apply padding if specified
        if let Some(pad) = self.padding {
            input = input.pad(&[pad, pad, pad, pad], "constant", 0.0);
        }
        
        let (_, _, h, w) = input.size4().unwrap();
        let (crop_h, crop_w) = self.size;
        
        if h < crop_h || w < crop_w {
            panic!("Input size is smaller than crop size");
        }
        
        // Random top-left corner
        let top = ((h - crop_h) as f64 * Tensor::rand(&[], (Kind::Float, Device::Cpu)).double_value(&[])) as i64;
        let left = ((w - crop_w) as f64 * Tensor::rand(&[], (Kind::Float, Device::Cpu)).double_value(&[])) as i64;
        
        input.narrow(2, top, crop_h).narrow(3, left, crop_w)
    }
}

/// Center crop transformation
pub struct CenterCrop {
    size: (i64, i64),
}

impl CenterCrop {
    pub fn new(size: (i64, i64)) -> Self {
        Self { size }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let (_, _, h, w) = tensor.size4().unwrap();
        let (crop_h, crop_w) = self.size;
        
        if h < crop_h || w < crop_w {
            panic!("Input size is smaller than crop size");
        }
        
        let top = (h - crop_h) / 2;
        let left = (w - crop_w) / 2;
        
        tensor.narrow(2, top, crop_h).narrow(3, left, crop_w)
    }
}

/// Resize transformation
pub struct Resize {
    size: (i64, i64),
    interpolation: InterpolationMode,
}

#[derive(Debug, Clone, Copy)]
pub enum InterpolationMode {
    Nearest,
    Linear,
    Bilinear,
    Bicubic,
}

impl Resize {
    pub fn new(size: (i64, i64), interpolation: InterpolationMode) -> Self {
        Self { size, interpolation }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let (h, w) = self.size;
        let mode = match self.interpolation {
            InterpolationMode::Nearest => "nearest",
            InterpolationMode::Linear => "linear",
            InterpolationMode::Bilinear => "bilinear",
            InterpolationMode::Bicubic => "bicubic",
        };
        
        tensor.upsample_bilinear2d(&[h, w], false, None, None)
    }
}

/// Random rotation transformation
pub struct RandomRotation {
    degrees: (f64, f64),
}

impl RandomRotation {
    pub fn new(degrees: (f64, f64)) -> Self {
        Self { degrees }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let (min_deg, max_deg) = self.degrees;
        let angle = min_deg + (max_deg - min_deg) * 
            Tensor::rand(&[], (Kind::Float, Device::Cpu)).double_value(&[]);
        
        // Convert to radians
        let theta = angle * PI / 180.0;
        
        // Create rotation matrix
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        
        // Affine transformation matrix for rotation
        let rotation_matrix = Tensor::of_slice(&[
            cos_theta, -sin_theta, 0.0,
            sin_theta, cos_theta, 0.0,
        ]).view([2, 3]);
        
        // Apply affine transformation
        let grid = tch::vision::image::affine_grid_generator(
            &rotation_matrix.unsqueeze(0),
            &tensor.size(),
            false,
        );
        
        tch::vision::image::grid_sampler(
            tensor,
            &grid,
            tch::vision::image::GridSamplerInterpolation::Bilinear,
            tch::vision::image::GridSamplerPadding::Zeros,
            false,
        )
    }
}

/// Color jitter transformation
pub struct ColorJitter {
    brightness: f64,
    contrast: f64,
    saturation: f64,
    hue: f64,
}

impl ColorJitter {
    pub fn new(brightness: f64, contrast: f64, saturation: f64, hue: f64) -> Self {
        Self {
            brightness,
            contrast,
            saturation,
            hue,
        }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let mut output = tensor.shallow_clone();
        
        // Brightness adjustment
        if self.brightness > 0.0 {
            let factor = 1.0 - self.brightness + 
                2.0 * self.brightness * Tensor::rand(&[], (Kind::Float, Device::Cpu)).double_value(&[]);
            output = &output * factor;
        }
        
        // Contrast adjustment
        if self.contrast > 0.0 {
            let factor = 1.0 - self.contrast + 
                2.0 * self.contrast * Tensor::rand(&[], (Kind::Float, Device::Cpu)).double_value(&[]);
            let mean = output.mean_dim(&[-3, -2, -1], true, Kind::Float);
            output = &mean + (&output - &mean) * factor;
        }
        
        // Saturation and hue would require RGB to HSV conversion
        // Simplified implementation here
        
        output.clamp(0.0, 1.0)
    }
}

/// Compose multiple transformations
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
    
    pub fn forward(&self, tensor: &Tensor) -> Tensor {
        let mut output = tensor.shallow_clone();
        for transform in &self.transforms {
            output = transform.forward(&output);
        }
        output
    }
}

/// Trait for all transformations
pub trait Transform {
    fn forward(&self, tensor: &Tensor) -> Tensor;
}

impl Transform for Normalize {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

impl Transform for RandomHorizontalFlip {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

impl Transform for RandomVerticalFlip {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

impl Transform for RandomCrop {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

impl Transform for CenterCrop {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

impl Transform for Resize {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

impl Transform for RandomRotation {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

impl Transform for ColorJitter {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

impl Transform for Compose {
    fn forward(&self, tensor: &Tensor) -> Tensor {
        self.forward(tensor)
    }
}

/// Convert image to tensor
pub fn to_tensor(image: &[u8], height: i64, width: i64, channels: i64) -> Tensor {
    let normalized: Vec<f32> = image.iter().map(|&x| x as f32 / 255.0).collect();
    Tensor::of_slice(&normalized)
        .reshape(&[height, width, channels])
        .permute(&[2, 0, 1])  // HWC to CHW
}

/// Convert tensor to image bytes
pub fn to_image(tensor: &Tensor) -> Vec<u8> {
    let tensor = tensor.permute(&[1, 2, 0]);  // CHW to HWC
    let tensor = tensor.clamp(0.0, 1.0) * 255.0;
    Vec::<u8>::from(tensor.to_kind(Kind::Uint8))
}