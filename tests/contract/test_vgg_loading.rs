//! Contract test for VGG model loading API
#![cfg(feature = "torch-rs")]

use tch::vision::models;
use tch::{Device, Kind, Tensor};

#[test]
fn test_vgg11_loading() {
    let model = models::vgg11(false, false).expect("Should create VGG11");
    assert!(model.num_parameters() > 100_000_000); // VGG models are large

    let input = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);
    assert_eq!(output.size(), &[1, 1000]);
}

#[test]
fn test_vgg16_batch_norm() {
    let model = models::vgg16_bn(false).expect("Should create VGG16 with batch norm");

    let input = Tensor::randn(&[2, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);
    assert_eq!(output.size(), &[2, 1000]);
}

#[test]
fn test_vgg_variants() {
    let variants = vec![
        ("vgg11", false, 132_863_336),
        ("vgg13", false, 133_047_848),
        ("vgg16", false, 138_357_544),
        ("vgg19", false, 143_667_240),
        ("vgg11_bn", true, 132_868_840),
        ("vgg13_bn", true, 133_053_736),
        ("vgg16_bn", true, 138_365_992),
        ("vgg19_bn", true, 143_678_248),
    ];

    for (name, batch_norm, expected_params) in variants {
        let model = match name {
            "vgg11" => models::vgg11(false, batch_norm),
            "vgg13" => models::vgg13(false, batch_norm),
            "vgg16" => models::vgg16(false, batch_norm),
            "vgg19" => models::vgg19(false, batch_norm),
            "vgg11_bn" => models::vgg11_bn(false),
            "vgg13_bn" => models::vgg13_bn(false),
            "vgg16_bn" => models::vgg16_bn(false),
            "vgg19_bn" => models::vgg19_bn(false),
            _ => panic!("Unknown variant"),
        };

        let model = model.expect(&format!("Should create {}", name));
        assert_eq!(
            model.num_parameters(),
            expected_params,
            "Wrong param count for {}",
            name
        );
    }
}

#[test]
fn test_vgg_pretrained() {
    let model = models::vgg16(true, false).expect("Should load pretrained VGG16");

    // Test with ImageNet normalization
    let input = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);

    // Check output is valid classification logits
    assert_eq!(output.size(), &[1, 1000]);
    let max_val = f64::try_from(output.max()).unwrap();
    let min_val = f64::try_from(output.min()).unwrap();
    assert!(max_val > min_val);
}