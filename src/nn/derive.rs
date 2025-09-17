//! Derive Macro for Project Phoenix Module System
//!
//! Provides automatic implementation of the PhoenixModule trait

#[cfg(feature = "torch-rs")]
pub use phoenix_derive_impl::*;

// This macro implementation will be enabled when the full procedural macro support is added
#[cfg(feature = "torch-rs")]
mod phoenix_derive_impl {
    // For now, we'll provide the manual implementation template and utilities
    // The actual derive macro will be implemented as a separate proc-macro crate

    /// Utility functions for manual PhoenixModule implementations
    pub mod utils {
        use crate::Tensor;
        use std::collections::HashMap;

        /// Collect parameters from nested fields recursively
        pub fn collect_parameters_recursive<'a, T>(
            fields: &'a [(&str, &T)],
            prefix: &str,
        ) -> Vec<&'a Tensor>
        where
            T: crate::nn::phoenix::PhoenixModule,
        {
            let mut params = Vec::new();
            for (name, field) in fields {
                let field_params = field.parameters();
                params.extend(field_params);
            }
            params
        }

        /// Collect named parameters from nested fields recursively
        pub fn collect_named_parameters_recursive<'a, T>(
            fields: &'a [(&str, &T)],
            prefix: &str,
        ) -> HashMap<String, &'a Tensor>
        where
            T: crate::nn::phoenix::PhoenixModule,
        {
            let mut named_params = HashMap::new();
            for (field_name, field) in fields {
                let field_params = field.named_parameters();
                for (param_name, param) in field_params {
                    let full_name = if prefix.is_empty() {
                        if field_name.is_empty() {
                            param_name
                        } else {
                            format!("{}.{}", field_name, param_name)
                        }
                    } else if field_name.is_empty() {
                        format!("{}.{}", prefix, param_name)
                    } else {
                        format!("{}.{}.{}", prefix, field_name, param_name)
                    };
                    named_params.insert(full_name, param);
                }
            }
            named_params
        }

        /// Apply device movement to nested fields
        pub fn move_fields_to_device<T>(
            fields: &mut [(&str, &mut T)],
            device: crate::Device,
        ) -> Result<(), crate::nn::phoenix::PhoenixModuleError>
        where
            T: crate::nn::phoenix::PhoenixModule,
        {
            for (_, field) in fields {
                field.to_device(device)?;
            }
            Ok(())
        }

        /// Set training mode for nested fields
        pub fn set_training_for_fields<T>(
            fields: &mut [(&str, &mut T)],
            training: bool,
        )
        where
            T: crate::nn::phoenix::PhoenixModule,
        {
            for (_, field) in fields {
                field.set_training(training);
            }
        }

        /// Zero gradients for nested fields
        pub fn zero_grad_for_fields<T>(
            fields: &mut [(&str, &mut T)],
        )
        where
            T: crate::nn::phoenix::PhoenixModule,
        {
            for (_, field) in fields {
                field.zero_grad();
            }
        }

        /// Get consistent device from nested fields
        pub fn get_device_from_fields<T>(
            fields: &[(&str, &T)],
        ) -> Option<crate::Device>
        where
            T: crate::nn::phoenix::PhoenixModule,
        {
            let mut device = None;
            for (_, field) in fields {
                let field_device = field.device();
                match (device, field_device) {
                    (None, Some(d)) => device = Some(d),
                    (Some(d1), Some(d2)) if d1 != d2 => return None, // Mixed devices
                    _ => {}
                }
            }
            device
        }

        /// Check if all fields are in training mode
        pub fn is_training_from_fields<T>(
            fields: &[(&str, &T)],
            default_training: bool,
        ) -> bool
        where
            T: crate::nn::phoenix::PhoenixModule,
        {
            if fields.is_empty() {
                default_training
            } else {
                fields.iter().all(|(_, field)| field.is_training())
            }
        }
    }

    /// Manual implementation macro for PhoenixModule
    ///
    /// This macro generates the boilerplate code for implementing PhoenixModule
    /// until we have a full procedural derive macro.
    #[macro_export]
    macro_rules! impl_phoenix_module {
        (
            $struct_name:ident {
                $( $field_name:ident : $field_type:ty ),* $(,)?
            }
        ) => {
            impl crate::nn::phoenix::PhoenixModule for $struct_name {
                fn parameters(&self) -> Vec<&crate::Tensor> {
                    let mut params = Vec::new();
                    $(
                        // Check if field implements PhoenixModule
                        let field_params = self.$field_name.parameters();
                        params.extend(field_params);
                    )*
                    params
                }

                fn parameters_mut(&mut self) -> Vec<&mut crate::Tensor> {
                    let mut params = Vec::new();
                    $(
                        let field_params = self.$field_name.parameters_mut();
                        params.extend(field_params);
                    )*
                    params
                }

                fn named_parameters(&self) -> std::collections::HashMap<String, &crate::Tensor> {
                    let mut named_params = std::collections::HashMap::new();
                    $(
                        let field_params = self.$field_name.named_parameters();
                        for (param_name, param) in field_params {
                            let full_name = format!("{}.{}", stringify!($field_name), param_name);
                            named_params.insert(full_name, param);
                        }
                    )*
                    named_params
                }

                fn named_parameters_mut(&mut self) -> std::collections::HashMap<String, &mut crate::Tensor> {
                    let mut named_params = std::collections::HashMap::new();
                    $(
                        let field_params = self.$field_name.named_parameters_mut();
                        for (param_name, param) in field_params {
                            let full_name = format!("{}.{}", stringify!($field_name), param_name);
                            named_params.insert(full_name, param);
                        }
                    )*
                    named_params
                }

                fn to_device(&mut self, device: crate::Device) -> Result<(), crate::nn::phoenix::PhoenixModuleError> {
                    $(
                        self.$field_name.to_device(device)?;
                    )*
                    Ok(())
                }

                fn set_training(&mut self, training: bool) {
                    $(
                        self.$field_name.set_training(training);
                    )*
                }

                fn is_training(&self) -> bool {
                    // If no fields, default to training
                    let field_states = vec![$( self.$field_name.is_training() ),*];
                    if field_states.is_empty() {
                        true
                    } else {
                        field_states.iter().all(|&state| state)
                    }
                }

                fn zero_grad(&mut self) {
                    $(
                        self.$field_name.zero_grad();
                    )*
                }

                fn device(&self) -> Option<crate::Device> {
                    let devices: Vec<_> = vec![$( self.$field_name.device() ),*]
                        .into_iter()
                        .flatten()
                        .collect();

                    if devices.is_empty() {
                        None
                    } else {
                        let first = devices[0];
                        if devices.iter().all(|&d| d == first) {
                            Some(first)
                        } else {
                            None // Mixed devices
                        }
                    }
                }
            }
        };
    }

    /// Helper macro for implementing PhoenixModule for leaf modules (with direct Tensor fields)
    #[macro_export]
    macro_rules! impl_phoenix_module_leaf {
        (
            $struct_name:ident {
                tensors: { $( $tensor_field:ident ),* $(,)? },
                training_field: $training_field:ident,
                $( others: { $( $other_field:ident ),* $(,)? } )?
            }
        ) => {
            impl crate::nn::phoenix::PhoenixModule for $struct_name {
                fn parameters(&self) -> Vec<&crate::Tensor> {
                    let mut params = Vec::new();
                    $(
                        if self.$tensor_field.requires_grad() {
                            params.push(&self.$tensor_field);
                        }
                    )*
                    params
                }

                fn parameters_mut(&mut self) -> Vec<&mut crate::Tensor> {
                    let mut params = Vec::new();
                    $(
                        if self.$tensor_field.requires_grad() {
                            params.push(&mut self.$tensor_field);
                        }
                    )*
                    params
                }

                fn named_parameters(&self) -> std::collections::HashMap<String, &crate::Tensor> {
                    let mut named_params = std::collections::HashMap::new();
                    $(
                        if self.$tensor_field.requires_grad() {
                            named_params.insert(stringify!($tensor_field).to_string(), &self.$tensor_field);
                        }
                    )*
                    named_params
                }

                fn named_parameters_mut(&mut self) -> std::collections::HashMap<String, &mut crate::Tensor> {
                    let mut named_params = std::collections::HashMap::new();
                    $(
                        if self.$tensor_field.requires_grad() {
                            named_params.insert(stringify!($tensor_field).to_string(), &mut self.$tensor_field);
                        }
                    )*
                    named_params
                }

                fn to_device(&mut self, device: crate::Device) -> Result<(), crate::nn::phoenix::PhoenixModuleError> {
                    $(
                        self.$tensor_field = self.$tensor_field.to_device(device);
                    )*
                    Ok(())
                }

                fn set_training(&mut self, training: bool) {
                    self.$training_field = training;
                }

                fn is_training(&self) -> bool {
                    self.$training_field
                }

                fn zero_grad(&mut self) {
                    $(
                        if self.$tensor_field.requires_grad() {
                            if let Some(grad) = self.$tensor_field.grad() {
                                let _ = grad.zero_();
                            }
                        }
                    )*
                }

                fn device(&self) -> Option<crate::Device> {
                    let devices: Vec<_> = vec![$( self.$tensor_field.device() ),*];
                    if devices.is_empty() {
                        None
                    } else {
                        let first = devices[0];
                        if devices.iter().all(|&d| d == first) {
                            Some(first)
                        } else {
                            None
                        }
                    }
                }
            }
        };
    }

    // TODO: Full procedural derive macro implementation
    // This would be implemented as a separate proc-macro crate with:
    //
    // #[proc_macro_derive(PhoenixModule)]
    // pub fn derive_phoenix_module(input: TokenStream) -> TokenStream {
    //     // Parse the input struct
    //     // Analyze fields for PhoenixModule or Tensor types
    //     // Generate appropriate implementation
    // }

    #[cfg(test)]
    mod tests {
        use super::*;

        // Example usage of the manual implementation macro
        struct TestLinear {
            weight: crate::Tensor,
            bias: crate::Tensor,
            training: bool,
        }

        impl std::fmt::Debug for TestLinear {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("TestLinear")
                    .field("weight_size", &self.weight.size())
                    .field("bias_size", &self.bias.size())
                    .field("training", &self.training)
                    .finish()
            }
        }

        impl crate::nn::Module for TestLinear {
            fn forward(&self, xs: &crate::Tensor) -> crate::Tensor {
                xs.matmul(&self.weight.tr()) + &self.bias
            }
        }

        // Use the leaf macro for this simple case
        impl_phoenix_module_leaf!(TestLinear {
            tensors: { weight, bias },
            training_field: training,
        });

        struct TestMLP {
            fc1: TestLinear,
            fc2: TestLinear,
            training: bool,
        }

        impl std::fmt::Debug for TestMLP {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("TestMLP")
                    .field("fc1", &self.fc1)
                    .field("fc2", &self.fc2)
                    .field("training", &self.training)
                    .finish()
            }
        }

        impl crate::nn::Module for TestMLP {
            fn forward(&self, xs: &crate::Tensor) -> crate::Tensor {
                let h1 = self.fc1.forward(xs).relu();
                self.fc2.forward(&h1)
            }
        }

        // Use the composite macro for nested modules
        impl_phoenix_module!(TestMLP {
            fc1: TestLinear,
            fc2: TestLinear,
        });

        #[test]
        fn test_manual_derive_macro() {
            use crate::{Kind, Device, Tensor};

            let weight1 = Tensor::randn(&[128, 784], (Kind::Float, Device::Cpu)).set_requires_grad(true);
            let bias1 = Tensor::randn(&[128], (Kind::Float, Device::Cpu)).set_requires_grad(true);
            let fc1 = TestLinear { weight: weight1, bias: bias1, training: true };

            let weight2 = Tensor::randn(&[10, 128], (Kind::Float, Device::Cpu)).set_requires_grad(true);
            let bias2 = Tensor::randn(&[10], (Kind::Float, Device::Cpu)).set_requires_grad(true);
            let fc2 = TestLinear { weight: weight2, bias: bias2, training: true };

            let model = TestMLP { fc1, fc2, training: true };

            // Test parameter collection
            let params = model.parameters();
            assert_eq!(params.len(), 4); // 2 weights + 2 biases

            // Test named parameters
            let named_params = model.named_parameters();
            assert_eq!(named_params.len(), 4);
            assert!(named_params.contains_key("fc1.weight"));
            assert!(named_params.contains_key("fc1.bias"));
            assert!(named_params.contains_key("fc2.weight"));
            assert!(named_params.contains_key("fc2.bias"));

            // Test device consistency
            assert_eq!(model.device(), Some(Device::Cpu));

            // Test training mode
            assert!(model.is_training());
        }
    }
}

// Placeholder for future procedural macro crate
#[cfg(feature = "torch-rs")]
pub mod proc_macro_placeholder {
    //! Placeholder for future procedural macro implementation
    //!
    //! This will eventually be moved to a separate `tch-derive` crate
    //! that provides:
    //!
    //! ```rust
    //! #[derive(PhoenixModule)]
    //! struct MyModel {
    //!     layer1: Linear,
    //!     layer2: Linear,
    //! }
    //! ```

    /// Future derive macro signature
    ///
    /// ```rust,ignore
    /// #[proc_macro_derive(PhoenixModule)]
    /// pub fn derive_phoenix_module(input: TokenStream) -> TokenStream {
    ///     // Implementation will:
    ///     // 1. Parse struct fields
    ///     // 2. Identify PhoenixModule and Tensor fields
    ///     // 3. Generate recursive parameter collection
    ///     // 4. Generate device movement logic
    ///     // 5. Generate training mode propagation
    ///     // 6. Handle both composite and leaf modules
    /// }
    /// ```
    pub fn future_derive_macro_info() -> &'static str {
        "Future derive macro will be implemented in separate tch-derive crate"
    }
}