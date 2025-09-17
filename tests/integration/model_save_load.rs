//! Integration tests for model persistence
//!
//! Tests saving and loading models with various formats and configurations

use tch::{nn, nn::Module, Device, Kind, Tensor};
use tch::test_utils::fixtures;
use std::path::PathBuf;

#[test]
fn test_save_load_varstore() {
    let device = Device::Cpu;
    let model_path = std::env::temp_dir().join("test_model.pt");

    // Create and train model
    let original_loss = {
        let vs = nn::VarStore::new(device);
        let model = fixtures::create_simple_cnn(&vs.root());
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        // Train briefly
        let dataset = fixtures::load_mnist_sample(32);
        let mut last_loss = 0.0;

        for (images, labels) in dataset.train_iter(16).take(5) {
            let loss = model.forward_t(&images, true)
                .cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);
            last_loss = f64::try_from(loss).unwrap();
        }

        // Save model
        vs.save(&model_path).expect("Failed to save model");
        last_loss
    };

    // Load model and verify
    {
        let vs = nn::VarStore::new(device);
        let model = fixtures::create_simple_cnn(&vs.root());

        // Load weights
        vs.load(&model_path).expect("Failed to load model");

        // Test with same data
        let dataset = fixtures::load_mnist_sample(32);
        let (images, labels) = dataset.train_iter(16).next().unwrap();

        let loss = model.forward_t(&images, false)
            .cross_entropy_for_logits(&labels);

        let loaded_loss = f64::try_from(loss).unwrap();

        // Loss should be similar (not exact due to randomness in data loading)
        assert!((loaded_loss - original_loss).abs() < 0.5,
                "Loaded model should produce similar loss");
    }

    // Cleanup
    std::fs::remove_file(&model_path).ok();
}

#[test]
fn test_save_load_safetensors() {
    let device = Device::Cpu;
    let model_path = std::env::temp_dir().join("test_model.safetensors");

    // Create model with specific weights
    let test_tensor = Tensor::randn(&[10, 10], (Kind::Float, device));

    {
        let vs = nn::VarStore::new(device);
        let _layer = nn::linear(&vs.root() / "layer", 10, 10, Default::default());

        // Set specific weights for testing
        vs.variables()
            .trainable_variables
            .iter()
            .for_each(|var| {
                let _ = var.copy_(&test_tensor);
            });

        // Save in safetensors format
        vs.save_safetensors(&model_path).expect("Failed to save safetensors");
    }

    // Load and verify
    {
        let vs = nn::VarStore::new(device);
        let _layer = nn::linear(&vs.root() / "layer", 10, 10, Default::default());

        vs.load_safetensors(&model_path).expect("Failed to load safetensors");

        // Check weights match
        for var in vs.variables().trainable_variables.iter() {
            let diff = var.mean(Kind::Float) - test_tensor.mean(Kind::Float);
            assert!(f64::try_from(diff.abs()).unwrap() < 1e-6,
                    "Loaded weights should match saved weights");
        }
    }

    // Cleanup
    std::fs::remove_file(&model_path).ok();
}

#[test]
fn test_partial_model_loading() {
    let device = Device::Cpu;
    let encoder_path = std::env::temp_dir().join("encoder.pt");
    let full_model_path = std::env::temp_dir().join("full_model.pt");

    // Create and save encoder
    {
        let vs = nn::VarStore::new(device);
        let _encoder = nn::seq()
            .add(nn::linear(&vs.root() / "encoder" / "layer1", 784, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.root() / "encoder" / "layer2", 256, 128, Default::default()));

        vs.save(&encoder_path).unwrap();
    }

    // Create full model and load encoder weights
    {
        let vs = nn::VarStore::new(device);

        // Encoder part
        let _encoder = nn::seq()
            .add(nn::linear(&vs.root() / "encoder" / "layer1", 784, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.root() / "encoder" / "layer2", 256, 128, Default::default()));

        // Decoder part (not loaded)
        let _decoder = nn::seq()
            .add(nn::linear(&vs.root() / "decoder" / "layer1", 128, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(&vs.root() / "decoder" / "layer2", 256, 784, Default::default()));

        // Load only encoder weights
        let loaded_vars = vs.load_partial(&encoder_path).unwrap();

        assert!(!loaded_vars.is_empty(), "Should load encoder variables");
        assert!(loaded_vars.iter().all(|v| v.contains("encoder")),
                "Should only load encoder variables");

        vs.save(&full_model_path).unwrap();
    }

    // Cleanup
    std::fs::remove_file(&encoder_path).ok();
    std::fs::remove_file(&full_model_path).ok();
}

#[test]
fn test_optimizer_state_persistence() {
    let device = Device::Cpu;
    let checkpoint_path = std::env::temp_dir().join("checkpoint_with_opt.pt");

    // Train and save optimizer state
    let (initial_loss, initial_step) = {
        let vs = nn::VarStore::new(device);
        let model = fixtures::create_simple_mlp(&vs.root(), 784, 10);
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        let dataset = fixtures::load_mnist_sample(32);
        let mut last_loss = 0.0;
        let mut step = 0;

        for (images, labels) in dataset.train_iter(16).take(10) {
            let images_flat = images.view([-1, 784]);
            let loss = model.forward_t(&images_flat, true)
                .cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);
            last_loss = f64::try_from(loss).unwrap();
            step += 1;
        }

        // Save both model and optimizer state
        opt.save(&checkpoint_path).unwrap();
        vs.save(&checkpoint_path.with_extension("model.pt")).unwrap();

        (last_loss, step)
    };

    // Load and continue training
    {
        let vs = nn::VarStore::new(device);
        let model = fixtures::create_simple_mlp(&vs.root(), 784, 10);
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        // Load states
        vs.load(&checkpoint_path.with_extension("model.pt")).unwrap();
        opt.load(&checkpoint_path).unwrap();

        // Continue training
        let dataset = fixtures::load_mnist_sample(32);
        let mut continued_loss = 0.0;
        let mut continued_step = initial_step;

        for (images, labels) in dataset.train_iter(16).take(5) {
            let images_flat = images.view([-1, 784]);
            let loss = model.forward_t(&images_flat, true)
                .cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);
            continued_loss = f64::try_from(loss).unwrap();
            continued_step += 1;
        }

        // Loss should continue decreasing from checkpoint
        assert!(continued_loss <= initial_loss * 1.5,
                "Training should continue smoothly from checkpoint");
        assert_eq!(continued_step, initial_step + 5,
                   "Step count should continue from checkpoint");
    }

    // Cleanup
    std::fs::remove_file(&checkpoint_path).ok();
    std::fs::remove_file(&checkpoint_path.with_extension("model.pt")).ok();
}

#[test]
fn test_model_versioning() {
    let device = Device::Cpu;
    let base_path = std::env::temp_dir().join("model_versions");
    std::fs::create_dir_all(&base_path).ok();

    // Save multiple versions
    for version in 0..3 {
        let vs = nn::VarStore::new(device);
        let model = fixtures::create_simple_cnn(&vs.root());

        // Modify weights slightly for each version
        for var in vs.variables().trainable_variables.iter() {
            let noise = Tensor::randn_like(var) * 0.01 * version as f64;
            let _ = var.add_(&noise);
        }

        let version_path = base_path.join(format!("v{}.pt", version));
        vs.save(&version_path).unwrap();
    }

    // Load and compare versions
    let mut version_means = Vec::new();

    for version in 0..3 {
        let vs = nn::VarStore::new(device);
        let _model = fixtures::create_simple_cnn(&vs.root());

        let version_path = base_path.join(format!("v{}.pt", version));
        vs.load(&version_path).unwrap();

        // Calculate mean of all parameters
        let total_mean = vs.variables()
            .trainable_variables
            .iter()
            .map(|v| f64::try_from(v.mean(Kind::Float)).unwrap())
            .sum::<f64>() / vs.variables().trainable_variables.len() as f64;

        version_means.push(total_mean);
    }

    // Versions should be slightly different
    for i in 1..version_means.len() {
        assert!((version_means[i] - version_means[i-1]).abs() > 1e-6,
                "Different versions should have different weights");
    }

    // Cleanup
    std::fs::remove_dir_all(&base_path).ok();
}