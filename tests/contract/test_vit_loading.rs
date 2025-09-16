//! Contract test for Vision Transformer model loading API
#![cfg(feature = "torch-rs")]

use tch::vision::models;
use tch::{Device, Kind, Tensor};

#[test]
fn test_vit_b_16_loading() {
    let model = models::vit_b_16(false).expect("Should create ViT-B/16");
    assert!(model.num_parameters() > 80_000_000); // ~86M params

    let input = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);
    assert_eq!(output.size(), &[1, 1000]);
}

#[test]
fn test_vit_l_32_loading() {
    let model = models::vit_l_32(false).expect("Should create ViT-L/32");
    assert!(model.num_parameters() > 300_000_000); // ~304M params

    // ViT-L/32 uses 32x32 patches, so works with 224x224 images
    let input = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);
    assert_eq!(output.size(), &[1, 1000]);
}

#[test]
fn test_vit_variants() {
    let variants = vec![
        ("vit_b_16", 224, 86_567_656),  // ViT-Base, 16x16 patches
        ("vit_b_32", 224, 88_224_232),  // ViT-Base, 32x32 patches
        ("vit_l_16", 224, 304_326_632), // ViT-Large, 16x16 patches
        ("vit_l_32", 224, 306_535_400), // ViT-Large, 32x32 patches
    ];

    for (name, img_size, expected_params) in variants {
        let model = match name {
            "vit_b_16" => models::vit_b_16(false),
            "vit_b_32" => models::vit_b_32(false),
            "vit_l_16" => models::vit_l_16(false),
            "vit_l_32" => models::vit_l_32(false),
            _ => panic!("Unknown variant"),
        };

        let model = model.expect(&format!("Should create {}", name));
        assert_eq!(
            model.num_parameters(),
            expected_params,
            "Wrong param count for {}",
            name
        );

        // Test forward pass
        let input = Tensor::randn(&[1, 3, img_size, img_size], (Kind::Float, Device::Cpu));
        let output = model.forward(&input);
        assert_eq!(output.size(), &[1, 1000]);
    }
}

#[test]
fn test_vit_pretrained() {
    let model = models::vit_b_16(true).expect("Should load pretrained ViT-B/16");

    // Test with proper input size
    let input = Tensor::randn(&[2, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);

    assert_eq!(output.size(), &[2, 1000]);

    // Check attention mechanism is working (output should vary across batch)
    let out1 = output.get(0);
    let out2 = output.get(1);
    let diff = (&out1 - &out2).abs().sum(Kind::Float);
    assert!(f64::try_from(diff).unwrap() > 0.01);
}