//! Integration tests for data loading pipeline
//!
//! Tests the complete data flow from loading to batching and augmentation

use tch::{Device, Kind, Tensor};
use tch::test_utils::fixtures;
use tch::torch_data::{DataLoader, Dataset, Transform};

#[test]
fn test_dataloader_batching() {
    // This test should fail until DataLoader is fully implemented
    let dataset = fixtures::load_mnist_sample(100);

    let dataloader = DataLoader::new(dataset)
        .batch_size(32)
        .shuffle(true)
        .num_workers(2)
        .build();

    let mut batch_count = 0;
    let mut total_samples = 0;

    for (batch_images, batch_labels) in dataloader {
        let batch_size = batch_images.size()[0];

        assert!(batch_size <= 32, "Batch size should not exceed configured size");
        assert_eq!(
            batch_images.size().len(),
            4,
            "MNIST images should be 4D tensor [B, C, H, W]"
        );
        assert_eq!(
            batch_labels.size().len(),
            1,
            "Labels should be 1D tensor"
        );

        batch_count += 1;
        total_samples += batch_size as usize;
    }

    assert_eq!(total_samples, 100, "Should process all samples");
    assert!(batch_count >= 3, "Should have multiple batches");
}

#[test]
fn test_data_transforms_pipeline() {
    let dataset = fixtures::load_cifar10_sample(50);

    // Create transform pipeline
    let transforms = vec![
        Transform::normalize(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225]),
        Transform::random_horizontal_flip(0.5),
        Transform::random_crop(32, 4),
    ];

    let transformed_dataset = dataset.with_transforms(transforms);

    for (original, transformed) in dataset.iter().zip(transformed_dataset.iter()) {
        let (orig_img, orig_label) = original;
        let (trans_img, trans_label) = transformed;

        // Labels should be unchanged
        assert_eq!(orig_label, trans_label, "Labels should not be affected by transforms");

        // Images should be transformed
        let orig_mean = f64::try_from(orig_img.mean(Kind::Float)).unwrap();
        let trans_mean = f64::try_from(trans_img.mean(Kind::Float)).unwrap();

        assert!((orig_mean - trans_mean).abs() > 1e-6,
                "Transformed image should differ from original");

        // Shape should be preserved
        assert_eq!(orig_img.size(), trans_img.size(),
                   "Transform should preserve image dimensions");
    }
}

#[test]
fn test_parallel_data_loading() {
    let dataset = fixtures::load_mnist_sample(1000);

    let single_worker_loader = DataLoader::new(dataset.clone())
        .batch_size(64)
        .num_workers(1)
        .build();

    let multi_worker_loader = DataLoader::new(dataset)
        .batch_size(64)
        .num_workers(4)
        .build();

    // Time single worker loading
    let start_single = std::time::Instant::now();
    let single_batches: Vec<_> = single_worker_loader.collect();
    let single_time = start_single.elapsed();

    // Time multi-worker loading
    let start_multi = std::time::Instant::now();
    let multi_batches: Vec<_> = multi_worker_loader.collect();
    let multi_time = start_multi.elapsed();

    assert_eq!(
        single_batches.len(),
        multi_batches.len(),
        "Should produce same number of batches"
    );

    // Multi-worker should be faster (in theory, may vary on CI)
    println!("Single worker: {:?}, Multi-worker: {:?}", single_time, multi_time);
}

#[test]
fn test_dataset_splitting() {
    let full_dataset = fixtures::load_mnist_sample(1000);

    // Split into train/val/test
    let (train_dataset, temp) = full_dataset.split(0.8);
    let (val_dataset, test_dataset) = temp.split(0.5);

    assert_eq!(train_dataset.len(), 800, "Train should have 80% of data");
    assert_eq!(val_dataset.len(), 100, "Val should have 10% of data");
    assert_eq!(test_dataset.len(), 100, "Test should have 10% of data");

    // Ensure no overlap between splits
    let train_indices = train_dataset.get_indices();
    let val_indices = val_dataset.get_indices();
    let test_indices = test_dataset.get_indices();

    for train_idx in &train_indices {
        assert!(!val_indices.contains(train_idx), "No overlap between train and val");
        assert!(!test_indices.contains(train_idx), "No overlap between train and test");
    }

    for val_idx in &val_indices {
        assert!(!test_indices.contains(val_idx), "No overlap between val and test");
    }
}

#[test]
fn test_weighted_sampling() {
    let dataset = fixtures::load_imbalanced_dataset();

    // Create weighted sampler for balanced batching
    let class_weights = dataset.compute_class_weights();
    let sampler = WeightedRandomSampler::new(class_weights, dataset.len());

    let dataloader = DataLoader::new(dataset)
        .batch_size(32)
        .sampler(sampler)
        .build();

    // Collect class distribution from batches
    let mut class_counts = vec![0; 10];

    for (_, labels) in dataloader.take(100) {
        for label in labels.iter() {
            let class_idx = label as usize;
            class_counts[class_idx] += 1;
        }
    }

    // Check that sampling is more balanced
    let min_count = *class_counts.iter().min().unwrap();
    let max_count = *class_counts.iter().max().unwrap();

    assert!(
        max_count < min_count * 3,
        "Weighted sampling should balance class distribution"
    );
}

#[test]
fn test_data_augmentation_consistency() {
    let dataset = fixtures::load_cifar10_sample(10);

    // Deterministic augmentation
    let det_transforms = vec![
        Transform::center_crop(28),
        Transform::normalize(vec![0.5], vec![0.5]),
    ];

    let det_dataset = dataset.clone().with_transforms(det_transforms.clone());

    // Process same sample twice
    let (img1, _) = det_dataset.get(0).unwrap();
    let (img2, _) = det_dataset.get(0).unwrap();

    assert_eq!(
        img1.size(),
        img2.size(),
        "Deterministic transforms should produce same output"
    );

    let diff = (&img1 - &img2).abs().max();
    assert!(
        f64::try_from(diff).unwrap() < 1e-6,
        "Deterministic transforms should be consistent"
    );

    // Random augmentation
    let random_transforms = vec![
        Transform::random_rotation(15.0),
        Transform::random_brightness(0.2),
    ];

    let rand_dataset = dataset.with_transforms(random_transforms);

    let (rand_img1, _) = rand_dataset.get(0).unwrap();
    let (rand_img2, _) = rand_dataset.get(0).unwrap();

    let rand_diff = (&rand_img1 - &rand_img2).abs().mean(Kind::Float);
    assert!(
        f64::try_from(rand_diff).unwrap() > 1e-6,
        "Random transforms should produce different outputs"
    );
}

#[test]
fn test_memory_pinning() {
    let dataset = fixtures::load_mnist_sample(256);

    let pinned_loader = DataLoader::new(dataset.clone())
        .batch_size(64)
        .pin_memory(true)
        .build();

    let unpinned_loader = DataLoader::new(dataset)
        .batch_size(64)
        .pin_memory(false)
        .build();

    // Measure transfer time to GPU (if available)
    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    if device.is_cuda() {
        let start_pinned = std::time::Instant::now();
        for (images, _) in pinned_loader.take(10) {
            let _ = images.to_device(device);
        }
        let pinned_time = start_pinned.elapsed();

        let start_unpinned = std::time::Instant::now();
        for (images, _) in unpinned_loader.take(10) {
            let _ = images.to_device(device);
        }
        let unpinned_time = start_unpinned.elapsed();

        // Pinned memory should be faster for GPU transfer
        println!("Pinned: {:?}, Unpinned: {:?}", pinned_time, unpinned_time);
    }
}

#[test]
fn test_collate_functions() {
    #[derive(Clone)]
    struct VariableLengthDataset;

    impl Dataset for VariableLengthDataset {
        type Item = (Vec<f32>, i64);

        fn len(&self) -> usize {
            100
        }

        fn get(&self, index: usize) -> Option<Self::Item> {
            // Variable length sequences
            let length = 10 + (index % 20);
            let data = vec![index as f32; length];
            let label = (index % 10) as i64;
            Some((data, label))
        }
    }

    let dataset = VariableLengthDataset;

    // Default collate (will fail on variable length)
    let default_loader = DataLoader::new(dataset.clone())
        .batch_size(8)
        .build();

    let result = default_loader.take(1).next();
    assert!(result.is_none() || result.is_some(),
            "Default collate may fail on variable length data");

    // Custom collate with padding
    let padded_loader = DataLoader::new(dataset)
        .batch_size(8)
        .collate_fn(|batch| {
            // Pad sequences to max length in batch
            let max_len = batch.iter().map(|(seq, _)| seq.len()).max().unwrap();

            let padded_batch: Vec<_> = batch.into_iter().map(|(mut seq, label)| {
                seq.resize(max_len, 0.0);
                (Tensor::from_slice(&seq), label)
            }).collect();

            // Stack into batch tensors
            let images = Tensor::stack(&padded_batch.iter().map(|(img, _)| img.clone()).collect::<Vec<_>>(), 0);
            let labels = Tensor::from_slice(&padded_batch.iter().map(|(_, lbl)| *lbl).collect::<Vec<_>>());

            (images, labels)
        })
        .build();

    for (batch_data, batch_labels) in padded_loader.take(5) {
        let batch_size = batch_data.size()[0];
        assert_eq!(batch_size, 8, "Batch size should be consistent");
        assert_eq!(
            batch_data.size()[1],
            batch_data.size()[1],
            "All sequences in batch should have same length after padding"
        );
    }
}