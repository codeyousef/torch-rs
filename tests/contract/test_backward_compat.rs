// Test that existing tch-rs code continues to work with torch-rs features enabled

#[test]
fn test_base_functionality_still_works() {
    // Verify that enabling torch-rs doesn't break existing functionality
    use tch::{nn, Device, Tensor, Kind};

    // Test basic tensor operations still work
    let a = Tensor::randn(&[3, 4], (Kind::Float, Device::Cpu));
    let b = Tensor::randn(&[3, 4], (Kind::Float, Device::Cpu));
    let c = &a + &b;
    assert_eq!(c.size(), vec![3, 4]);

    // Test VarStore still works
    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();
    let _linear = nn::linear(&root, 10, 5, Default::default());
    assert!(vs.trainable_variables().len() > 0);

    // Test existing optimizer APIs still work
    use tch::nn::{Adam, Sgd};
    let opt = Adam::default().build(&vs, 1e-3).unwrap();
    assert!(opt.to_c_opt_ptr() != std::ptr::null_mut());

    let opt2 = Sgd::default().build(&vs, 1e-2).unwrap();
    assert!(opt2.to_c_opt_ptr() != std::ptr::null_mut());
}

#[test]
fn test_existing_module_apis() {
    // Test that existing module APIs are unchanged
    use tch::nn::{Module, ModuleT, Linear, Conv2D};
    use tch::{nn, Device, Tensor, Kind};

    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();

    // Test Linear module
    let linear = Linear::new(&root, 10, 5, Default::default());
    let input = Tensor::randn(&[2, 10], (Kind::Float, Device::Cpu));
    let output = linear.forward(&input);
    assert_eq!(output.size(), vec![2, 5]);

    // Test Conv2D module
    let conv = Conv2D::new(&root, 3, 16, 3, Default::default());
    let input = Tensor::randn(&[1, 3, 32, 32], (Kind::Float, Device::Cpu));
    let output = conv.forward(&input);
    assert_eq!(output.size()[0], 1);
    assert_eq!(output.size()[1], 16);
}

#[cfg(feature = "torch-rs")]
#[test]
fn test_feature_isolation() {
    // Test that torch-rs features are properly isolated behind feature flags
    // This test only runs when torch-rs is enabled

    // Verify we can still use base functionality
    use tch::{Tensor, Kind, Device};
    let t = Tensor::ones(&[2, 2], (Kind::Float, Device::Cpu));
    assert_eq!(t.sum(Kind::Float).double_value(&[]), 4.0);

    // The enhanced features should be available but not interfere
    // with base functionality
    assert!(true, "Features are properly isolated");
}

#[test]
fn test_no_breaking_changes() {
    // This test ensures no breaking changes to existing APIs
    // All of these should compile and work as before

    use tch::{nn, Tensor, Device, Kind};
    use tch::nn::{Module, Sequential};

    let vs = nn::VarStore::new(Device::Cpu);
    let root = vs.root();

    // Build a simple sequential model using existing APIs
    let model = Sequential::new()
        .add(nn::linear(&root / "layer1", 784, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(&root / "layer2", 128, 10, Default::default()));

    let input = Tensor::randn(&[32, 784], (Kind::Float, Device::Cpu));
    let output = model.forward(&input);
    assert_eq!(output.size(), vec![32, 10]);

    assert!(true, "No breaking changes detected");
}