//! Project Phoenix Dataset Demo
//!
//! Demonstrates dataset loading and data pipeline with MNIST

#[cfg(feature = "torch-rs")]
use tch::{
    nn::phoenix::{PhoenixModule, Linear, Sequential, Dropout},
    optim::phoenix::{Adam, PhoenixOptimizer},
    data::{MnistDataset, DataLoader, DataLoaderConfig, Dataset, VisionDataset},
    Device, Kind, Tensor,
};

#[cfg(feature = "torch-rs")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Project Phoenix Dataset Demo");

    // Create MNIST dataset (training set)
    println!("\n📊 Loading MNIST dataset...");

    // For demo, we'll create a small mock dataset since MNIST requires download
    let mock_dataset = MockMnistDataset::new(1000);
    println!("✅ Mock MNIST dataset created with {} samples", mock_dataset.len());
    println!("   Classes: {}", mock_dataset.num_classes().unwrap());
    println!("   Image shape: {:?}", mock_dataset.image_shape());

    // Create data loader with batching and shuffling
    let config = DataLoaderConfig {
        batch_size: 32,
        shuffle: true,
        drop_last: false,
        ..Default::default()
    };

    let mut dataloader = DataLoader::new(mock_dataset, config);
    println!("✅ DataLoader created with {} batches", dataloader.len());

    // Create a CNN model for MNIST
    let mut model = MnistCNN::new();
    println!("✅ MNIST CNN model created with {} parameters", model.num_parameters());

    // Print model structure
    println!("\n📋 Model Structure:");
    for (name, param) in model.named_parameters() {
        println!("  {}: {:?}", name, param.size());
    }

    // Move model to device
    let device = Device::Cpu;
    model.to_device(device)?;
    println!("✅ Model moved to device: {:?}", model.device());

    // Create Adam optimizer
    let mut optimizer = Adam::with_lr(model.parameters_mut(), 1e-3)?;
    println!("✅ Adam optimizer created with learning rate: {}", optimizer.learning_rate());

    // Training loop with real data loading
    println!("\n🏋️  Training with DataLoader:");
    model.train();

    let mut batch_count = 0;
    for batch_result in dataloader.next() {
        if batch_count >= 3 { break; } // Just show first 3 batches for demo

        let (images, labels) = batch_result?;
        batch_count += 1;

        println!("  Batch {}: images={:?}, labels={:?}",
                 batch_count, images.size(), labels.size());

        // Forward pass
        let output = model.forward(&images);
        assert_eq!(output.size(), &[images.size()[0], 10]);

        // Compute loss
        let loss = output.cross_entropy_for_logits(&labels);
        let loss_value = f64::try_from(&loss)?;

        // Backward pass
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()?;

        println!("    Loss = {:.4}", loss_value);
    }

    // Test inference mode
    println!("\n🔮 Inference Mode:");
    model.eval();

    let test_image = Tensor::randn(&[1, 1, 28, 28], (Kind::Float, device));
    let test_output = model.forward(&test_image);
    let probabilities = test_output.softmax(-1, Kind::Float);
    let prediction = probabilities.argmax(-1, false);

    println!("  Test image shape: {:?}", test_image.size());
    println!("  Model output shape: {:?}", test_output.size());
    println!("  Predicted class: {}", i64::try_from(prediction)?);
    println!("  Confidence: {:.4}", f64::try_from(probabilities.max())?);

    // Dataset information
    println!("\n📈 Dataset Statistics:");
    println!("  Total samples: {}", 1000);
    println!("  Batch size: 32");
    println!("  Total batches: {}", (1000 + 31) / 32);

    println!("\n🎉 Phoenix Dataset Demo completed successfully!");
    println!("   ✨ PyTorch-like data loading achieved in Rust!");

    Ok(())
}

#[cfg(feature = "torch-rs")]
/// Mock MNIST dataset for demonstration
struct MockMnistDataset {
    size: usize,
}

#[cfg(feature = "torch-rs")]
impl MockMnistDataset {
    fn new(size: usize) -> Self {
        Self { size }
    }
}

#[cfg(feature = "torch-rs")]
impl Dataset for MockMnistDataset {
    type Item = (Tensor, i64);

    fn len(&self) -> usize {
        self.size
    }

    fn get(&self, index: usize) -> Result<Self::Item, tch::data::DatasetError> {
        if index >= self.size {
            return Err(tch::data::DatasetError::IndexOutOfBounds {
                index,
                size: self.size
            });
        }

        // Create fake MNIST-like image (28x28, single channel)
        let image = Tensor::randn(&[1, 28, 28], (Kind::Float, Device::Cpu));
        let label = (index % 10) as i64; // 10 classes for digits 0-9

        Ok((image, label))
    }

    fn download(&self) -> Result<(), tch::data::DatasetError> {
        Ok(()) // Mock dataset doesn't need downloading
    }

    fn is_downloaded(&self) -> bool {
        true
    }

    fn root(&self) -> &std::path::PathBuf {
        static ROOT: std::path::PathBuf = std::path::PathBuf::new();
        &ROOT
    }
}

#[cfg(feature = "torch-rs")]
impl VisionDataset for MockMnistDataset {
    type Image = Tensor;
    type Target = i64;

    fn get_item(&self, index: usize) -> Result<(Self::Image, Self::Target), tch::data::DatasetError> {
        self.get(index)
    }

    fn class_names(&self) -> Option<Vec<String>> {
        Some((0..10).map(|i| i.to_string()).collect())
    }

    fn num_classes(&self) -> Option<usize> {
        Some(10)
    }

    fn image_shape(&self) -> Option<(usize, usize, usize)> {
        Some((1, 28, 28))
    }
}

#[cfg(feature = "torch-rs")]
/// Simple CNN for MNIST classification
#[derive(Debug)]
struct MnistCNN {
    conv: Sequential,
    classifier: Sequential,
    training: bool,
}

#[cfg(feature = "torch-rs")]
impl MnistCNN {
    fn new() -> Self {
        use tch::nn::phoenix::{Conv2d, BatchNorm2d};

        let conv = Sequential::new()
            .add(Conv2d::new(1, 32, 3)) // 28x28 -> 26x26
            .add(BatchNorm2d::new(32))
            .add(Conv2d::new(32, 64, 3)) // 26x26 -> 24x24
            .add(BatchNorm2d::new(64))
            .add(Dropout::new(0.2));

        let classifier = Sequential::new()
            .add(Linear::new(64 * 24 * 24, 128))
            .add(Dropout::new(0.5))
            .add(Linear::new(128, 10));

        Self {
            conv,
            classifier,
            training: true,
        }
    }
}

#[cfg(feature = "torch-rs")]
impl tch::nn::Module for MnistCNN {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let conv_out = self.conv.forward(xs).relu();
        let flattened = conv_out.flatten(1, -1);
        self.classifier.forward(&flattened)
    }
}

#[cfg(feature = "torch-rs")]
tch::impl_phoenix_module!(MnistCNN {
    conv: Sequential,
    classifier: Sequential,
});

#[cfg(not(feature = "torch-rs"))]
fn main() {
    println!("Phoenix dataset demo requires the 'torch-rs' feature flag.");
    println!("Run with: cargo run --example phoenix_dataset_demo --features torch-rs");
}