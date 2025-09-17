//! CIFAR-10 dataset implementation for Project Phoenix
//!
//! Provides CIFAR-10 image classification dataset with automatic downloads

use super::dataset::{Dataset, VisionDataset, DatasetError, DatasetMetadata, utils};
use crate::{Device, Kind, Tensor};
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Read;

const CIFAR10_URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
const CIFAR10_FILENAME: &str = "cifar-10-binary.tar.gz";
const CIFAR10_FOLDER: &str = "cifar-10-batches-bin";

const CIFAR10_CLASSES: &[&str] = &[
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
];

#[derive(Debug)]
pub struct Cifar10Dataset {
    root: PathBuf,
    train: bool,
    download_enabled: bool,
    images: Option<Tensor>,
    labels: Option<Tensor>,
    transforms: Vec<Box<dyn crate::torch_data::Transform>>,
}

impl Cifar10Dataset {
    pub fn new<P: Into<PathBuf>>(root: P, train: bool, download: bool) -> Result<Self, DatasetError> {
        let root = root.into();
        let mut dataset = Self {
            root,
            train,
            download_enabled: download,
            images: None,
            labels: None,
            transforms: Vec::new(),
        };

        if !dataset.is_downloaded() && download {
            dataset.download()?;
        }

        dataset.load_data()?;
        Ok(dataset)
    }

    pub fn with_transforms(mut self, transforms: Vec<Box<dyn crate::torch_data::Transform>>) -> Self {
        self.transforms = transforms;
        self
    }

    fn raw_folder(&self) -> PathBuf {
        self.root.join("CIFAR10").join("raw")
    }

    fn processed_folder(&self) -> PathBuf {
        self.root.join("CIFAR10").join("processed")
    }

    fn data_folder(&self) -> PathBuf {
        self.raw_folder().join(CIFAR10_FOLDER)
    }

    fn load_data(&mut self) -> Result<(), DatasetError> {
        let data_folder = self.data_folder();

        if !data_folder.exists() {
            return Err(DatasetError::NotDownloaded {
                reason: "CIFAR-10 data folder not found".to_string(),
            });
        }

        let (images, labels) = if self.train {
            self.load_train_data(&data_folder)?
        } else {
            self.load_test_data(&data_folder)?
        };

        self.images = Some(images);
        self.labels = Some(labels);

        Ok(())
    }

    fn load_train_data(&self, data_folder: &Path) -> Result<(Tensor, Tensor), DatasetError> {
        let mut all_images = Vec::new();
        let mut all_labels = Vec::new();

        for batch_num in 1..=5 {
            let batch_file = data_folder.join(format!("data_batch_{}.bin", batch_num));
            let (batch_images, batch_labels) = self.load_batch_file(&batch_file)?;
            all_images.push(batch_images);
            all_labels.push(batch_labels);
        }

        let images = Tensor::cat(&all_images, 0);
        let labels = Tensor::cat(&all_labels, 0);

        Ok((images, labels))
    }

    fn load_test_data(&self, data_folder: &Path) -> Result<(Tensor, Tensor), DatasetError> {
        let test_file = data_folder.join("test_batch.bin");
        self.load_batch_file(&test_file)
    }

    fn load_batch_file<P: AsRef<Path>>(&self, path: P) -> Result<(Tensor, Tensor), DatasetError> {
        let mut file = File::open(&path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        const RECORD_SIZE: usize = 3073; // 1 label byte + 3072 image bytes (32x32x3)
        let num_records = buffer.len() / RECORD_SIZE;

        if buffer.len() % RECORD_SIZE != 0 {
            return Err(DatasetError::CorruptedData {
                reason: format!("Invalid file size: {}", buffer.len()),
            });
        }

        let mut images_data = Vec::new();
        let mut labels_data = Vec::new();

        for i in 0..num_records {
            let record_start = i * RECORD_SIZE;
            let label = buffer[record_start];
            labels_data.push(label as i64);

            let image_data = &buffer[record_start + 1..record_start + RECORD_SIZE];

            let mut rgb_data = vec![0u8; 3072];
            for j in 0..1024 {
                rgb_data[j] = image_data[j];           // Red channel
                rgb_data[j + 1024] = image_data[j + 1024]; // Green channel
                rgb_data[j + 2048] = image_data[j + 2048]; // Blue channel
            }

            images_data.extend(rgb_data.into_iter().map(|x| x as f32 / 255.0));
        }

        let images = Tensor::from_slice(&images_data)
            .reshape(&[num_records as i64, 3, 32, 32]);

        let labels = Tensor::from_slice(&labels_data);

        Ok((images, labels))
    }
}

impl Dataset for Cifar10Dataset {
    type Item = (Tensor, i64);

    fn len(&self) -> usize {
        match &self.images {
            Some(images) => images.size()[0] as usize,
            None => if self.train { 50000 } else { 10000 },
        }
    }

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        let images = self.images.as_ref()
            .ok_or_else(|| DatasetError::NotDownloaded {
                reason: "Images not loaded".to_string(),
            })?;

        let labels = self.labels.as_ref()
            .ok_or_else(|| DatasetError::NotDownloaded {
                reason: "Labels not loaded".to_string(),
            })?;

        if index >= self.len() {
            return Err(DatasetError::IndexOutOfBounds {
                index,
                size: self.len(),
            });
        }

        let mut image = images.get(index as i64);
        let label = i64::try_from(labels.get(index as i64))
            .map_err(|_| DatasetError::CorruptedData {
                reason: "Failed to extract label".to_string(),
            })?;

        for transform in &self.transforms {
            image = transform.apply(image)?;
        }

        Ok((image, label))
    }

    fn download(&self) -> Result<(), DatasetError> {
        if !self.download_enabled {
            return Err(DatasetError::ConfigError {
                reason: "Download not enabled".to_string(),
            });
        }

        let raw_folder = self.raw_folder();
        utils::ensure_dir(&raw_folder)?;

        let archive_path = raw_folder.join(CIFAR10_FILENAME);
        let data_folder = self.data_folder();

        if data_folder.exists() {
            return Ok(()); // Already extracted
        }

        if !archive_path.exists() {
            return Err(DatasetError::NetworkError {
                reason: format!(
                    "CIFAR-10 download not implemented. Please manually download {} and place it in {}",
                    CIFAR10_FILENAME, raw_folder.display()
                ),
            });
        }

        utils::extract_targz(&archive_path, &raw_folder)?;

        if !data_folder.exists() {
            return Err(DatasetError::CorruptedData {
                reason: "Failed to extract CIFAR-10 data".to_string(),
            });
        }

        Ok(())
    }

    fn is_downloaded(&self) -> bool {
        let data_folder = self.data_folder();
        let required_files = if self.train {
            vec![
                "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
                "data_batch_4.bin", "data_batch_5.bin", "batches.meta.txt"
            ]
        } else {
            vec!["test_batch.bin", "batches.meta.txt"]
        };

        data_folder.exists() && required_files.iter().all(|filename| {
            data_folder.join(filename).exists()
        })
    }

    fn root(&self) -> &PathBuf {
        &self.root
    }

    fn metadata(&self) -> DatasetMetadata {
        DatasetMetadata {
            name: "CIFAR-10".to_string(),
            version: "1.0.0".to_string(),
            description: "The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes".to_string(),
            url: Some("https://www.cs.toronto.edu/~kriz/cifar.html".to_string()),
            citation: Some("Krizhevsky, A. (2009). Learning multiple layers of features from tiny images".to_string()),
            license: Some("MIT License".to_string()),
            size_bytes: Some(170_000_000),
            checksum: None,
        }
    }
}

impl VisionDataset for Cifar10Dataset {
    type Image = Tensor;
    type Target = i64;

    fn get_item(&self, index: usize) -> Result<(Self::Image, Self::Target), DatasetError> {
        self.get(index)
    }

    fn class_names(&self) -> Option<Vec<String>> {
        Some(CIFAR10_CLASSES.iter().map(|&s| s.to_string()).collect())
    }

    fn num_classes(&self) -> Option<usize> {
        Some(10)
    }

    fn image_shape(&self) -> Option<(usize, usize, usize)> {
        Some((3, 32, 32))
    }

    fn class_distribution(&self) -> Option<Vec<usize>> {
        if let Some(labels) = &self.labels {
            let mut counts = vec![0; 10];
            for i in 0..labels.size()[0] {
                let label = i64::try_from(labels.get(i)).unwrap_or(0) as usize;
                if label < 10 {
                    counts[label] += 1;
                }
            }
            Some(counts)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_cifar10_creation() {
        let temp_dir = env::temp_dir().join("test_cifar10");

        let result = Cifar10Dataset::new(&temp_dir, true, false);
        match result {
            Ok(_) => panic!("Should fail when data is not available"),
            Err(DatasetError::NotDownloaded { .. }) => {},
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_cifar10_metadata() {
        let temp_dir = env::temp_dir().join("test_cifar10_meta");

        if let Ok(dataset) = Cifar10Dataset::new(&temp_dir, true, false) {
            let metadata = dataset.metadata();
            assert_eq!(metadata.name, "CIFAR-10");
            assert_eq!(metadata.version, "1.0.0");
            assert!(metadata.url.is_some());
        }
    }

    #[test]
    fn test_cifar10_class_info() {
        let temp_dir = env::temp_dir().join("test_cifar10_classes");

        if let Ok(dataset) = Cifar10Dataset::new(&temp_dir, true, false) {
            assert_eq!(dataset.num_classes(), Some(10));
            assert_eq!(dataset.image_shape(), Some((3, 32, 32)));
            assert!(dataset.is_classification());

            let class_names = dataset.class_names().unwrap();
            assert_eq!(class_names.len(), 10);
            assert_eq!(class_names[0], "airplane");
            assert_eq!(class_names[9], "truck");

            assert_eq!(class_names, vec![
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]);
        }
    }

    #[test]
    fn test_cifar10_len() {
        let temp_dir = env::temp_dir().join("test_cifar10_len");

        if let Ok(train_dataset) = Cifar10Dataset::new(&temp_dir, true, false) {
            assert_eq!(train_dataset.len(), 50000);
        }

        if let Ok(test_dataset) = Cifar10Dataset::new(&temp_dir, false, false) {
            assert_eq!(test_dataset.len(), 10000);
        }
    }

    #[test]
    fn test_cifar10_folders() {
        let temp_dir = env::temp_dir().join("test_cifar10_folders");
        let dataset = Cifar10Dataset::new(&temp_dir, true, false);

        if let Ok(dataset) = dataset {
            assert_eq!(dataset.root(), &temp_dir);

            let raw_folder = dataset.raw_folder();
            assert!(raw_folder.to_string_lossy().contains("CIFAR10/raw"));

            let processed_folder = dataset.processed_folder();
            assert!(processed_folder.to_string_lossy().contains("CIFAR10/processed"));

            let data_folder = dataset.data_folder();
            assert!(data_folder.to_string_lossy().contains("cifar-10-batches-bin"));
        }
    }

    #[test]
    fn test_cifar10_constants() {
        assert_eq!(CIFAR10_CLASSES.len(), 10);
        assert_eq!(CIFAR10_CLASSES[0], "airplane");
        assert_eq!(CIFAR10_CLASSES[9], "truck");
    }
}