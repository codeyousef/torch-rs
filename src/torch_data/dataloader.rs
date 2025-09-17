//! DataLoader implementation for Project Phoenix
//!
//! Provides efficient batched iteration with shuffling, multiprocessing, and collation

use super::dataset::{Dataset, DatasetError};
use crate::Tensor;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct DataLoaderConfig {
    pub batch_size: usize,
    pub shuffle: bool,
    pub num_workers: usize,
    pub pin_memory: bool,
    pub drop_last: bool,
    pub collate_fn: Option<Arc<dyn CollateFunction>>,
}

impl Default for DataLoaderConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            shuffle: false,
            num_workers: 0,
            pin_memory: false,
            drop_last: false,
            collate_fn: None,
        }
    }
}

pub trait CollateFunction: Send + Sync {
    fn collate(&self, batch: Vec<(Tensor, i64)>) -> Result<(Tensor, Tensor), DatasetError>;
}

#[derive(Debug)]
pub struct DefaultCollateFunction;

impl CollateFunction for DefaultCollateFunction {
    fn collate(&self, batch: Vec<(Tensor, i64)>) -> Result<(Tensor, Tensor), DatasetError> {
        if batch.is_empty() {
            return Err(DatasetError::ConfigError {
                reason: "Cannot collate empty batch".to_string(),
            });
        }

        let images: Vec<Tensor> = batch.iter().map(|(img, _)| img.unsqueeze(0)).collect();
        let labels: Vec<i64> = batch.iter().map(|(_, label)| *label).collect();

        let batched_images = Tensor::cat(&images, 0);
        let batched_labels = Tensor::from_slice(&labels);

        Ok((batched_images, batched_labels))
    }
}

// Standalone shuffle function that doesn't require Dataset bound
fn shuffle_indices_impl(indices: &mut [usize]) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut hasher = DefaultHasher::new();
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    let seed = hasher.finish() as usize;

    for i in (1..indices.len()).rev() {
        let j = (seed + i * 1664525 + 1013904223) % (i + 1);
        indices.swap(i, j);
    }
}

#[derive(Debug)]
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    config: DataLoaderConfig,
    indices: Vec<usize>,
    current_batch: usize,
    collate_fn: Arc<dyn CollateFunction>,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, config: DataLoaderConfig) -> Self {
        let dataset = Arc::new(dataset);
        let dataset_len = dataset.len();
        let mut indices: Vec<usize> = (0..dataset_len).collect();

        if config.shuffle {
            Self::shuffle_indices(&mut indices);
        }

        let collate_fn = config.collate_fn.clone()
            .unwrap_or_else(|| Arc::new(DefaultCollateFunction));

        Self {
            dataset,
            config,
            indices,
            current_batch: 0,
            collate_fn,
        }
    }

    pub fn with_batch_size(dataset: D, batch_size: usize) -> Self {
        let config = DataLoaderConfig {
            batch_size,
            ..Default::default()
        };
        Self::new(dataset, config)
    }

    pub fn with_shuffle(dataset: D, batch_size: usize, shuffle: bool) -> Self {
        let config = DataLoaderConfig {
            batch_size,
            shuffle,
            ..Default::default()
        };
        Self::new(dataset, config)
    }

    pub fn len(&self) -> usize {
        let dataset_len = self.indices.len();
        if self.config.drop_last {
            dataset_len / self.config.batch_size
        } else {
            (dataset_len + self.config.batch_size - 1) / self.config.batch_size
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reset(&mut self) {
        self.current_batch = 0;
        if self.config.shuffle {
            Self::shuffle_indices(&mut self.indices);
        }
    }

    pub fn batch_size(&self) -> usize {
        self.config.batch_size
    }

    pub fn dataset(&self) -> &Arc<D> {
        &self.dataset
    }

    fn shuffle_indices(indices: &mut [usize]) {
        shuffle_indices_impl(indices);
    }

    fn get_batch_indices(&self, batch_idx: usize) -> Vec<usize> {
        let start = batch_idx * self.config.batch_size;
        let end = std::cmp::min(start + self.config.batch_size, self.indices.len());

        if start >= self.indices.len() {
            return vec![];
        }

        if self.config.drop_last && (end - start) < self.config.batch_size {
            return vec![];
        }

        self.indices[start..end].to_vec()
    }
}

impl<D> Iterator for DataLoader<D>
where
    D: Dataset<Item = (Tensor, i64)> + Send + Sync,
{
    type Item = Result<(Tensor, Tensor), DatasetError>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch_indices = self.get_batch_indices(self.current_batch);

        if batch_indices.is_empty() {
            return None;
        }

        self.current_batch += 1;

        let mut batch_items = Vec::with_capacity(batch_indices.len());
        for &index in &batch_indices {
            match self.dataset.get(index) {
                Ok(item) => batch_items.push(item),
                Err(e) => return Some(Err(e)),
            }
        }

        Some(self.collate_fn.collate(batch_items))
    }
}

#[derive(Debug)]
pub struct RandomSampler {
    indices: Vec<usize>,
    current: usize,
}

impl RandomSampler {
    pub fn new(dataset_len: usize) -> Self {
        let mut indices: Vec<usize> = (0..dataset_len).collect();
        shuffle_indices_impl(&mut indices);

        Self {
            indices,
            current: 0,
        }
    }

    pub fn reset(&mut self) {
        self.current = 0;
        shuffle_indices_impl(&mut self.indices);
    }
}

impl Iterator for RandomSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.indices.len() {
            None
        } else {
            let index = self.indices[self.current];
            self.current += 1;
            Some(index)
        }
    }
}

#[derive(Debug)]
pub struct SequentialSampler {
    current: usize,
    total: usize,
}

impl SequentialSampler {
    pub fn new(dataset_len: usize) -> Self {
        Self {
            current: 0,
            total: dataset_len,
        }
    }

    pub fn reset(&mut self) {
        self.current = 0;
    }
}

impl Iterator for SequentialSampler {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            None
        } else {
            let index = self.current;
            self.current += 1;
            Some(index)
        }
    }
}

#[derive(Debug)]
pub struct BatchSampler<S: Iterator<Item = usize>> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Iterator<Item = usize>> BatchSampler<S> {
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }
}

impl<S: Iterator<Item = usize>> Iterator for BatchSampler<S> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut batch = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            match self.sampler.next() {
                Some(index) => batch.push(index),
                None => break,
            }
        }

        if batch.is_empty() || (self.drop_last && batch.len() < self.batch_size) {
            None
        } else {
            Some(batch)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dataset::{Dataset, DatasetError, DatasetMetadata};
    use crate::{Kind, Device, Tensor};
    use std::path::PathBuf;

    #[derive(Debug)]
    struct MockImageDataset {
        size: usize,
    }

    impl MockImageDataset {
        fn new(size: usize) -> Self {
            Self { size }
        }
    }

    impl Dataset for MockImageDataset {
        type Item = (Tensor, i64);

        fn len(&self) -> usize {
            self.size
        }

        fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
            if index >= self.size {
                return Err(DatasetError::IndexOutOfBounds {
                    index,
                    size: self.size,
                });
            }

            let image = Tensor::randn(&[3, 32, 32], (Kind::Float, Device::Cpu));
            let label = (index % 10) as i64;

            Ok((image, label))
        }

        fn download(&self) -> Result<(), DatasetError> {
            Ok(())
        }

        fn is_downloaded(&self) -> bool {
            true
        }

        fn root(&self) -> &PathBuf {
            static ROOT: PathBuf = PathBuf::new();
            &ROOT
        }
    }

    #[test]
    fn test_dataloader_creation() {
        let dataset = MockImageDataset::new(100);
        let config = DataLoaderConfig {
            batch_size: 10,
            shuffle: false,
            ..Default::default()
        };
        let dataloader = DataLoader::new(dataset, config);

        assert_eq!(dataloader.len(), 10);
        assert_eq!(dataloader.batch_size(), 10);
        assert!(!dataloader.is_empty());
    }

    #[test]
    fn test_dataloader_with_batch_size() {
        let dataset = MockImageDataset::new(50);
        let dataloader = DataLoader::with_batch_size(dataset, 8);

        assert_eq!(dataloader.len(), 7); // 50 / 8 = 6.25 -> 7 batches
        assert_eq!(dataloader.batch_size(), 8);
    }

    #[test]
    fn test_dataloader_iteration() {
        let dataset = MockImageDataset::new(25);
        let mut dataloader = DataLoader::with_batch_size(dataset, 10);

        let mut batches = 0;
        while let Some(batch_result) = dataloader.next() {
            let (images, labels) = batch_result.unwrap();
            batches += 1;

            match batches {
                1 | 2 => {
                    assert_eq!(images.size()[0], 10);
                    assert_eq!(labels.size()[0], 10);
                }
                3 => {
                    assert_eq!(images.size()[0], 5); // Last batch with 5 items
                    assert_eq!(labels.size()[0], 5);
                }
                _ => panic!("Too many batches"),
            }

            assert_eq!(images.size()[1..], [3, 32, 32]); // Channel, height, width
        }

        assert_eq!(batches, 3);
    }

    #[test]
    fn test_dataloader_drop_last() {
        let dataset = MockImageDataset::new(25);
        let config = DataLoaderConfig {
            batch_size: 10,
            drop_last: true,
            ..Default::default()
        };
        let mut dataloader = DataLoader::new(dataset, config);

        assert_eq!(dataloader.len(), 2); // 25 / 10 = 2 (dropping last 5)

        let mut batches = 0;
        while let Some(batch_result) = dataloader.next() {
            let (images, labels) = batch_result.unwrap();
            batches += 1;

            assert_eq!(images.size()[0], 10);
            assert_eq!(labels.size()[0], 10);
        }

        assert_eq!(batches, 2);
    }

    #[test]
    fn test_dataloader_reset() {
        let dataset = MockImageDataset::new(20);
        let config = DataLoaderConfig {
            batch_size: 5,
            shuffle: true,
            ..Default::default()
        };
        let mut dataloader = DataLoader::new(dataset, config);

        // First iteration
        let mut first_batches = Vec::new();
        while let Some(batch_result) = dataloader.next() {
            let (_, labels) = batch_result.unwrap();
            first_batches.push(labels);
        }

        // Reset and iterate again
        dataloader.reset();
        let mut second_batches = Vec::new();
        while let Some(batch_result) = dataloader.next() {
            let (_, labels) = batch_result.unwrap();
            second_batches.push(labels);
        }

        assert_eq!(first_batches.len(), 4);
        assert_eq!(second_batches.len(), 4);
    }

    #[test]
    fn test_sequential_sampler() {
        let mut sampler = SequentialSampler::new(10);

        for i in 0..10 {
            assert_eq!(sampler.next(), Some(i));
        }
        assert_eq!(sampler.next(), None);

        sampler.reset();
        assert_eq!(sampler.next(), Some(0));
    }

    #[test]
    fn test_random_sampler() {
        let mut sampler = RandomSampler::new(10);

        let mut indices = Vec::new();
        for _ in 0..10 {
            if let Some(index) = sampler.next() {
                indices.push(index);
            }
        }

        assert_eq!(indices.len(), 10);
        indices.sort();
        assert_eq!(indices, (0..10).collect::<Vec<_>>());

        assert_eq!(sampler.next(), None);
    }

    #[test]
    fn test_batch_sampler() {
        let sampler = SequentialSampler::new(25);
        let mut batch_sampler = BatchSampler::new(sampler, 10, false);

        let batch1 = batch_sampler.next().unwrap();
        assert_eq!(batch1, (0..10).collect::<Vec<_>>());

        let batch2 = batch_sampler.next().unwrap();
        assert_eq!(batch2, (10..20).collect::<Vec<_>>());

        let batch3 = batch_sampler.next().unwrap();
        assert_eq!(batch3, (20..25).collect::<Vec<_>>());

        assert!(batch_sampler.next().is_none());
    }

    #[test]
    fn test_batch_sampler_drop_last() {
        let sampler = SequentialSampler::new(25);
        let mut batch_sampler = BatchSampler::new(sampler, 10, true);

        let batch1 = batch_sampler.next().unwrap();
        assert_eq!(batch1, (0..10).collect::<Vec<_>>());

        let batch2 = batch_sampler.next().unwrap();
        assert_eq!(batch2, (10..20).collect::<Vec<_>>());

        assert!(batch_sampler.next().is_none()); // Last batch dropped (only 5 items)
    }

    #[test]
    fn test_default_collate_function() {
        let collate_fn = DefaultCollateFunction;

        let batch = vec![
            (Tensor::ones(&[3, 32, 32], (Kind::Float, Device::Cpu)), 1),
            (Tensor::zeros(&[3, 32, 32], (Kind::Float, Device::Cpu)), 2),
        ];

        let (images, labels) = collate_fn.collate(batch).unwrap();
        assert_eq!(images.size(), &[2, 3, 32, 32]);
        assert_eq!(labels.size(), &[2]);

        let labels_vec: Vec<i64> = Vec::try_from(&labels).unwrap();
        assert_eq!(labels_vec, vec![1, 2]);
    }

    #[test]
    fn test_empty_batch_collation() {
        let collate_fn = DefaultCollateFunction;
        let batch = vec![];

        match collate_fn.collate(batch) {
            Err(DatasetError::ConfigError { .. }) => {},
            _ => panic!("Expected ConfigError for empty batch"),
        }
    }
}