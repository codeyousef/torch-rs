//! Project Phoenix Demo
//!
//! Demonstrates the enhanced tch-rs features with PyTorch-like ergonomics

#[cfg(feature = "torch-rs")]
use tch::{
    nn::phoenix::{Linear, PhoenixModule},
    optim::phoenix::{PhoenixOptimizer, SGD},
    Device, Kind, Tensor,
};

#[cfg(feature = "torch-rs")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Project Phoenix Demo - PyTorch Parity for tch-rs");

    // Create a simple MLP model using Phoenix Linear layers
    let mut model = SimpleMLP::new(784, 128, 10);
    println!("✅ Model created with {} parameters", model.num_parameters());

    // Print model structure
    println!("\n📊 Model Structure:");
    for (name, param) in model.named_parameters() {
        println!("  {}: {:?}", name, param.size());
    }

    // Move model to device (would be GPU in real scenario)
    let device = Device::Cpu;
    model.to_device(device)?;
    println!("✅ Model moved to device: {:?}", model.device());

    // Create optimizer
    let mut optimizer = SGD::new(model.parameters_mut(), 0.01)?;
    println!("✅ Optimizer created with learning rate: {}", optimizer.learning_rate());

    // Simulate training loop
    println!("\n🏋️  Training Loop:");
    for epoch in 0..5 {
        // Generate fake batch
        let batch_size = 32;
        let input = Tensor::randn(&[batch_size, 784], (Kind::Float, device));
        let target = Tensor::randint(10, &[batch_size], (Kind::Int64, device));

        // Forward pass
        let output = model.forward(&input);
        assert_eq!(output.size(), &[batch_size, 10]);

        // Compute loss (cross entropy)
        let loss = output.cross_entropy_for_logits(&target);
        let loss_value = f64::try_from(loss)?;

        // Backward pass
        optimizer.zero_grad();
        loss.backward();

        // Update parameters
        optimizer.step()?;

        println!("  Epoch {}: Loss = {:.4}", epoch + 1, loss_value);
    }

    // Test inference mode
    model.set_training(false);
    println!("\n🔮 Inference Mode:");
    let test_input = Tensor::randn(&[1, 784], (Kind::Float, device));
    let test_output = model.forward(&test_input);
    let probabilities = test_output.softmax(-1, Kind::Float);
    let prediction = probabilities.argmax(-1, false);

    println!("  Input shape: {:?}", test_input.size());
    println!("  Output shape: {:?}", test_output.size());
    println!("  Predicted class: {}", i64::try_from(prediction)?);

    // Test state dictionary (PyTorch compatibility)
    println!("\n💾 State Dictionary:");
    let state_dict = model.state_dict();
    println!("  State dict contains {} tensors", state_dict.len());

    // Test optimizer state
    let opt_state = optimizer.state_dict();
    println!("  Optimizer state contains {} entries", opt_state.len());

    println!("\n🎉 Phoenix Demo completed successfully!");
    println!("   ✨ PyTorch-like ergonomics achieved in Rust!");

    Ok(())
}

#[cfg(feature = "torch-rs")]
/// Simple Multi-Layer Perceptron using Phoenix components
#[derive(Debug)]
struct SimpleMLP {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
    training: bool,
}

#[cfg(feature = "torch-rs")]
impl SimpleMLP {
    fn new(input_size: i64, hidden_size: i64, output_size: i64) -> Self {
        Self {
            fc1: Linear::new(input_size, hidden_size),
            fc2: Linear::new(hidden_size, hidden_size),
            fc3: Linear::new(hidden_size, output_size),
            training: true,
        }
    }
}

#[cfg(feature = "torch-rs")]
impl tch::nn::Module for SimpleMLP {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let h1 = self.fc1.forward(xs).relu();
        let h2 = self.fc2.forward(&h1).relu();
        self.fc3.forward(&h2)
    }
}

// Use the manual derive macro implementation
#[cfg(feature = "torch-rs")]
tch::impl_phoenix_module!(SimpleMLP { fc1: Linear, fc2: Linear, fc3: Linear });

#[cfg(not(feature = "torch-rs"))]
fn main() {
    println!("Phoenix demo requires the 'torch-rs' feature flag.");
    println!("Run with: cargo run --example phoenix_demo --features torch-rs");
}
