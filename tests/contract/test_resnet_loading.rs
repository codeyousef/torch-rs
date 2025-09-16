//! Contract test for ResNet model loading API
#![cfg(feature = "torch-rs")]

use tch::vision::models;
use tch::{Device, Kind, Tensor};

#[test]
fn test_resnet18_loading() {
    let model = models::resnet18(false).expect("Should create ResNet18");
    assert_eq!(model.num_parameters(), 11_689_512);

    let input = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);
    assert_eq!(output.size(), &[1, 1000]);
}

#[test]
fn test_resnet50_pretrained() {
    let model = models::resnet50(true).expect("Should load pretrained ResNet50");
    assert_eq!(model.num_parameters(), 25_557_032);

    let input = Tensor::randn(&[1, 3, 224, 224], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);
    assert_eq!(output.size(), &[1, 1000]);

    // Output should be normalized (softmax sums to 1)
    let probs = output.softmax(-1, Kind::Float);
    let sum = probs.sum(Kind::Float);
    assert!((f64::try_from(sum).unwrap() - 1.0).abs() < 1e-5);
}

#[test]
fn test_resnet_variants() {
    let variants = vec![
        ("resnet18", 11_689_512),
        ("resnet34", 21_797_672),
        ("resnet50", 25_557_032),
        ("resnet101", 44_549_160),
        ("resnet152", 60_192_808),
    ];

    for (name, expected_params) in variants {
        let model = match name {
            "resnet18" => models::resnet18(false),
            "resnet34" => models::resnet34(false),
            "resnet50" => models::resnet50(false),
            "resnet101" => models::resnet101(false),
            "resnet152" => models::resnet152(false),
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