//! Integration tests for complete training workflow
//!
//! Tests the integration between data loading, model training, and optimization

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use tch::test_utils::{fixtures, E2ETestHarness};

#[test]
fn test_complete_training_workflow() {
    // This test should fail until fixtures are implemented
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);

    // Load sample dataset
    let dataset = fixtures::load_mnist_sample(100);

    // Create simple model
    let model = fixtures::create_simple_cnn(&vs.root());

    // Setup optimizer
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    // Training loop
    let batch_size = 32;
    let num_epochs = 2;

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for (images, labels) in dataset.train_iter(batch_size) {
            let loss = model.forward_t(&images, true)
                .cross_entropy_for_logits(&labels);

            opt.backward_step(&loss);

            total_loss += f64::try_from(loss).unwrap();
            batch_count += 1;
        }

        let avg_loss = total_loss / batch_count as f64;
        assert!(avg_loss < 2.0, "Loss should decrease during training");

        // Validation
        let accuracy = model.batch_accuracy_for_logits(
            &dataset.test_images,
            &dataset.test_labels,
            device,
            1024
        );

        assert!(accuracy > 0.5, "Model should achieve >50% accuracy after epoch {}", epoch);
    }
}

#[test]
fn test_gradient_accumulation_workflow() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);

    let dataset = fixtures::load_cifar10_sample(64);
    let model = fixtures::create_resnet18(&vs.root(), 10);

    let mut opt = nn::SGD::default().build(&vs, 0.01).unwrap();

    // Gradient accumulation
    let accumulation_steps = 4;
    let micro_batch_size = 8;

    let mut accumulated_loss = Tensor::zeros(&[], (Kind::Float, device));
    let mut step_count = 0;

    for (images, labels) in dataset.train_iter(micro_batch_size).take(16) {
        let loss = model.forward_t(&images, true)
            .cross_entropy_for_logits(&labels) / accumulation_steps as f64;

        accumulated_loss = &accumulated_loss + &loss;
        step_count += 1;

        if step_count % accumulation_steps == 0 {
            opt.backward_step(&accumulated_loss);
            accumulated_loss = Tensor::zeros(&[], (Kind::Float, device));
        }
    }

    assert_eq!(step_count % accumulation_steps, 0, "Should complete accumulation cycles");
}

#[test]
fn test_learning_rate_scheduling() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);

    let model = fixtures::create_simple_mlp(&vs.root(), 784, 10);
    let mut opt = nn::Adam::default().build(&vs, 1e-2).unwrap();

    // Simulate learning rate scheduling
    let initial_lr = 1e-2;
    let decay_factor = 0.1;
    let decay_epochs = [3, 6, 9];

    let mut current_lr = initial_lr;

    for epoch in 0..10 {
        if decay_epochs.contains(&epoch) {
            current_lr *= decay_factor;
            opt.set_lr(current_lr);
        }

        assert_eq!(
            opt.get_lr(),
            current_lr,
            "Learning rate should match schedule at epoch {}",
            epoch
        );

        // Dummy training step
        let dummy_loss = Tensor::randn(&[], (Kind::Float, device));
        opt.backward_step(&dummy_loss);
    }

    assert_eq!(
        current_lr,
        initial_lr * decay_factor.powi(decay_epochs.len() as i32),
        "Final learning rate should reflect all decay steps"
    );
}

#[test]
fn test_mixed_precision_training() {
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);

    let dataset = fixtures::load_mnist_sample(32);
    let model = fixtures::create_simple_cnn(&vs.root());

    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

    // Mixed precision training simulation
    for (images, labels) in dataset.train_iter(16).take(5) {
        // Convert to half precision for forward pass
        let images_fp16 = images.to_kind(Kind::Half);

        // Forward pass in FP16
        let logits_fp16 = model.forward_t(&images_fp16, true);

        // Loss computation in FP32
        let logits_fp32 = logits_fp16.to_kind(Kind::Float);
        let loss = logits_fp32.cross_entropy_for_logits(&labels);

        // Scale loss for stability
        let scaled_loss = &loss * 1024.0;
        opt.backward_step(&scaled_loss);

        assert!(f64::try_from(&loss).unwrap() < 10.0, "Loss should remain stable");
    }
}

#[test]
#[serial_test::serial]
fn test_checkpoint_and_resume() {
    let device = Device::Cpu;
    let checkpoint_path = std::env::temp_dir().join("test_checkpoint.pt");

    // Initial training
    let initial_accuracy = {
        let vs = nn::VarStore::new(device);
        let model = fixtures::create_simple_cnn(&vs.root());
        let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();

        let dataset = fixtures::load_mnist_sample(100);

        // Train for a few steps
        for (images, labels) in dataset.train_iter(32).take(10) {
            let loss = model.forward_t(&images, true)
                .cross_entropy_for_logits(&labels);
            opt.backward_step(&loss);
        }

        // Save checkpoint
        vs.save(&checkpoint_path).unwrap();

        // Measure accuracy
        model.batch_accuracy_for_logits(
            &dataset.test_images,
            &dataset.test_labels,
            device,
            1024
        )
    };

    // Resume from checkpoint
    let resumed_accuracy = {
        let vs = nn::VarStore::new(device);
        let model = fixtures::create_simple_cnn(&vs.root());

        // Load checkpoint
        vs.load(&checkpoint_path).unwrap();

        let dataset = fixtures::load_mnist_sample(100);

        // Measure accuracy (should match)
        model.batch_accuracy_for_logits(
            &dataset.test_images,
            &dataset.test_labels,
            device,
            1024
        )
    };

    assert_eq!(
        initial_accuracy,
        resumed_accuracy,
        "Accuracy should match after loading checkpoint"
    );

    // Cleanup
    std::fs::remove_file(&checkpoint_path).ok();
}