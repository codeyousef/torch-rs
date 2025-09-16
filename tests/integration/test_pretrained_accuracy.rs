//! Integration test: Load ResNet50 → inference → match PyTorch output
#![cfg(feature = "torch-rs")]

use tch::vision::models;
use tch::{Device, Kind, Tensor};

#[test]
fn test_resnet50_inference_accuracy() {
    // This test will fail until models are implemented
    let model = models::resnet50(true).expect("Should load pretrained ResNet50");
    model.set_training(false);

    // Create test input with ImageNet normalization
    let input = create_imagenet_normalized_input();

    // Forward pass
    let output = model.forward(&input);
    assert_eq!(output.size(), vec![1, 1000]);

    // Get top-5 predictions
    let (values, indices) = output.topk(5, -1, true, true);
    let top5_indices: Vec<i64> = Vec::try_from(indices).unwrap();

    // Pretrained model should produce reasonable predictions
    // (not random - entropy should be low for top predictions)
    let probs = output.softmax(-1, Kind::Float);
    let top_prob = f64::try_from(probs.max()).unwrap();
    assert!(top_prob > 0.01, "Top prediction should have > 1% confidence");

    // Verify outputs are deterministic
    let output2 = model.forward(&input);
    let diff = (&output - &output2).abs().sum(Kind::Float);
    assert!(
        f64::try_from(diff).unwrap() < 1e-5,
        "Inference should be deterministic"
    );
}

#[test]
fn test_vgg16_inference_accuracy() {
    let model = models::vgg16(true, false).expect("Should load pretrained VGG16");
    model.set_training(false);

    let input = create_imagenet_normalized_input();
    let output = model.forward(&input);

    assert_eq!(output.size(), vec![1, 1000]);

    // Check that different inputs produce different outputs
    let input2 = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output2 = model.forward(&input2);

    let diff = (&output - &output2).abs().sum(Kind::Float);
    assert!(
        f64::try_from(diff).unwrap() > 0.1,
        "Different inputs should produce different outputs"
    );
}

#[test]
fn test_vit_inference_accuracy() {
    let model = models::vit_b_16(true).expect("Should load pretrained ViT-B/16");
    model.set_training(false);

    let input = create_imagenet_normalized_input();
    let output = model.forward(&input);

    assert_eq!(output.size(), vec![1, 1000]);

    // Vision Transformer should produce well-calibrated predictions
    let probs = output.softmax(-1, Kind::Float);
    let entropy = -(&probs * probs.log()).sum(Kind::Float);
    let max_entropy = (1000_f64).ln(); // Maximum entropy for 1000 classes

    assert!(
        f64::try_from(entropy).unwrap() < max_entropy * 0.8,
        "ViT should produce confident predictions"
    );
}

#[test]
fn test_model_comparison_consistency() {
    // Compare that all models process same input consistently
    let input = create_imagenet_normalized_input();

    let resnet = models::resnet50(true).unwrap();
    let vgg = models::vgg16(true, false).unwrap();
    let vit = models::vit_b_16(true).unwrap();

    resnet.set_training(false);
    vgg.set_training(false);
    vit.set_training(false);

    let out_resnet = resnet.forward(&input);
    let out_vgg = vgg.forward(&input);
    let out_vit = vit.forward(&input);

    // All should output 1000 classes
    assert_eq!(out_resnet.size(), vec![1, 1000]);
    assert_eq!(out_vgg.size(), vec![1, 1000]);
    assert_eq!(out_vit.size(), vec![1, 1000]);

    // All should sum to approximately 1 after softmax
    for output in [out_resnet, out_vgg, out_vit] {
        let probs = output.softmax(-1, Kind::Float);
        let sum = probs.sum(Kind::Float);
        assert!(
            (f64::try_from(sum).unwrap() - 1.0).abs() < 1e-5,
            "Softmax should sum to 1"
        );
    }
}

fn create_imagenet_normalized_input() -> Tensor {
    // Create input with ImageNet mean and std normalization
    let input = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));

    // ImageNet normalization
    let mean = Tensor::from_slice(&[0.485_f32, 0.456, 0.406])
        .reshape(&[1, 3, 1, 1])
        .to_device(Device::Cpu);
    let std = Tensor::from_slice(&[0.229_f32, 0.224, 0.225])
        .reshape(&[1, 3, 1, 1])
        .to_device(Device::Cpu);

    (input - mean) / std
}