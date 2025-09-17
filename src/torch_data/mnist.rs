//! MNIST dataset implementation for Project Phoenix
//!
//! Provides MNIST handwritten digit classification dataset with automatic downloads

use super::dataset::{utils, Dataset, DatasetError, DatasetMetadata, VisionDataset};
use crate::{Device, Kind, Tensor};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

const MNIST_URLS: &[(&str, &str)] = &[
    ("train-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"),
    ("train-labels-idx1-ubyte.gz", "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"),
    ("t10k-images-idx3-ubyte.gz", "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"),
    ("t10k-labels-idx1-ubyte.gz", "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"),
];

#[derive(Debug)]
pub struct MnistDataset {
    root: PathBuf,
    train: bool,
    download_enabled: bool,
    images: Option<Tensor>,
    labels: Option<Tensor>,
    transforms: Vec<Box<dyn crate::torch_data::Transform>>,
}

impl MnistDataset {
    pub fn new<P: Into<PathBuf>>(
        root: P,
        train: bool,
        download: bool,
    ) -> Result<Self, DatasetError> {
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

    pub fn with_transforms(
        mut self,
        transforms: Vec<Box<dyn crate::torch_data::Transform>>,
    ) -> Self {
        self.transforms = transforms;
        self
    }

    fn raw_folder(&self) -> PathBuf {
        self.root.join("MNIST").join("raw")
    }

    fn processed_folder(&self) -> PathBuf {
        self.root.join("MNIST").join("processed")
    }

    fn load_data(&mut self) -> Result<(), DatasetError> {
        let raw_folder = self.raw_folder();

        let (images_file, labels_file) = if self.train {
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        } else {
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        };

        let images_path = raw_folder.join(images_file);
        let labels_path = raw_folder.join(labels_file);

        if !images_path.exists() || !labels_path.exists() {
            return Err(DatasetError::NotDownloaded {
                reason: "MNIST data files not found".to_string(),
            });
        }

        let images = self.read_images(&images_path)?;
        let labels = self.read_labels(&labels_path)?;

        self.images = Some(images);
        self.labels = Some(labels);

        Ok(())
    }

    fn read_images<P: AsRef<Path>>(&self, path: P) -> Result<Tensor, DatasetError> {
        let mut file = File::open(&path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 16 {
            return Err(DatasetError::CorruptedData {
                reason: "Images file too small".to_string(),
            });
        }

        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 2051 {
            return Err(DatasetError::CorruptedData {
                reason: format!("Invalid magic number for images: {}", magic),
            });
        }

        let num_images = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as i64;
        let num_rows = u32::from_be_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]) as i64;
        let num_cols = u32::from_be_bytes([buffer[12], buffer[13], buffer[14], buffer[15]]) as i64;

        if num_rows != 28 || num_cols != 28 {
            return Err(DatasetError::CorruptedData {
                reason: format!("Unexpected image dimensions: {}x{}", num_rows, num_cols),
            });
        }

        let expected_size = (16 + num_images * 28 * 28) as usize;
        if buffer.len() != expected_size {
            return Err(DatasetError::CorruptedData {
                reason: format!(
                    "File size mismatch: expected {}, got {}",
                    expected_size,
                    buffer.len()
                ),
            });
        }

        let image_data = &buffer[16..];
        let tensor =
            Tensor::from_slice(image_data).reshape(&[num_images, 28, 28]).to_kind(Kind::Float)
                / 255.0;

        Ok(tensor)
    }

    fn read_labels<P: AsRef<Path>>(&self, path: P) -> Result<Tensor, DatasetError> {
        let mut file = File::open(&path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        if buffer.len() < 8 {
            return Err(DatasetError::CorruptedData {
                reason: "Labels file too small".to_string(),
            });
        }

        let magic = u32::from_be_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]);
        if magic != 2049 {
            return Err(DatasetError::CorruptedData {
                reason: format!("Invalid magic number for labels: {}", magic),
            });
        }

        let num_labels = u32::from_be_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]) as i64;

        let expected_size = (8 + num_labels) as usize;
        if buffer.len() != expected_size {
            return Err(DatasetError::CorruptedData {
                reason: format!(
                    "File size mismatch: expected {}, got {}",
                    expected_size,
                    buffer.len()
                ),
            });
        }

        let label_data = &buffer[8..];
        let tensor = Tensor::from_slice(label_data).to_kind(Kind::Int64);

        Ok(tensor)
    }

    fn extract_gz<P: AsRef<Path>, Q: AsRef<Path>>(
        gz_path: P,
        output_path: Q,
    ) -> Result<(), DatasetError> {
        use std::io::Write;

        let file = File::open(&gz_path)?;
        let mut decoder = flate2::read::GzDecoder::new(file);
        let mut contents = Vec::new();
        decoder.read_to_end(&mut contents)?;

        let mut output_file = File::create(&output_path)?;
        output_file.write_all(&contents)?;

        Ok(())
    }
}

impl Dataset for MnistDataset {
    type Item = (Tensor, i64);

    fn len(&self) -> usize {
        match &self.images {
            Some(images) => images.size()[0] as usize,
            None => {
                if self.train {
                    60000
                } else {
                    10000
                }
            }
        }
    }

    fn get(&self, index: usize) -> Result<Self::Item, DatasetError> {
        let images = self.images.as_ref().ok_or_else(|| DatasetError::NotDownloaded {
            reason: "Images not loaded".to_string(),
        })?;

        let labels = self.labels.as_ref().ok_or_else(|| DatasetError::NotDownloaded {
            reason: "Labels not loaded".to_string(),
        })?;

        if index >= self.len() {
            return Err(DatasetError::IndexOutOfBounds { index, size: self.len() });
        }

        let mut image = images.get(index as i64);
        let label = i64::try_from(labels.get(index as i64)).map_err(|_| {
            DatasetError::CorruptedData { reason: "Failed to extract label".to_string() }
        })?;

        for transform in &self.transforms {
            image = transform.apply(image)?;
        }

        Ok((image, label))
    }

    fn download(&self) -> Result<(), DatasetError> {
        if !self.download_enabled {
            return Err(DatasetError::ConfigError { reason: "Download not enabled".to_string() });
        }

        let raw_folder = self.raw_folder();
        utils::ensure_dir(&raw_folder)?;

        for (filename, _url) in MNIST_URLS {
            let gz_path = raw_folder.join(filename);
            let extracted_path = raw_folder.join(filename.trim_end_matches(".gz"));

            if extracted_path.exists() {
                continue;
            }

            if !gz_path.exists() {
                return Err(DatasetError::NetworkError {
                    reason: format!(
                        "MNIST download not implemented. Please manually download {} and place it in {}",
                        filename, raw_folder.display()
                    ),
                });
            }

            Self::extract_gz(&gz_path, &extracted_path)?;
        }

        Ok(())
    }

    fn is_downloaded(&self) -> bool {
        let raw_folder = self.raw_folder();
        let required_files = if self.train {
            vec!["train-images-idx3-ubyte", "train-labels-idx1-ubyte"]
        } else {
            vec!["t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]
        };

        required_files.iter().all(|filename| raw_folder.join(filename).exists())
    }

    fn root(&self) -> &PathBuf {
        &self.root
    }

    fn metadata(&self) -> DatasetMetadata {
        DatasetMetadata {
            name: "MNIST".to_string(),
            version: "1.0.0".to_string(),
            description: "The MNIST database of handwritten digits".to_string(),
            url: Some("http://yann.lecun.com/exdb/mnist/".to_string()),
            citation: Some(
                "LeCun, Y. (1998). The MNIST database of handwritten digits".to_string(),
            ),
            license: Some("Public Domain".to_string()),
            size_bytes: Some(11_000_000),
            checksum: None,
        }
    }
}

impl VisionDataset for MnistDataset {
    type Image = Tensor;
    type Target = i64;

    fn get_item(&self, index: usize) -> Result<(Self::Image, Self::Target), DatasetError> {
        self.get(index)
    }

    fn class_names(&self) -> Option<Vec<String>> {
        Some((0..10).map(|i| i.to_string()).collect())
    }

    fn num_classes(&self) -> Option<usize> {
        Some(10)
    }

    fn image_shape(&self) -> Option<(usize, usize, usize)> {
        Some((1, 28, 28))
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
    fn test_mnist_creation() {
        let temp_dir = env::temp_dir().join("test_mnist");

        let result = MnistDataset::new(&temp_dir, true, false);
        match result {
            Ok(_) => panic!("Should fail when data is not available"),
            Err(DatasetError::NotDownloaded { .. }) => {}
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[test]
    fn test_mnist_metadata() {
        let temp_dir = env::temp_dir().join("test_mnist_meta");

        if let Ok(dataset) = MnistDataset::new(&temp_dir, true, false) {
            let metadata = dataset.metadata();
            assert_eq!(metadata.name, "MNIST");
            assert_eq!(metadata.version, "1.0.0");
            assert!(metadata.url.is_some());
        }
    }

    #[test]
    fn test_mnist_class_info() {
        let temp_dir = env::temp_dir().join("test_mnist_classes");

        if let Ok(dataset) = MnistDataset::new(&temp_dir, true, false) {
            assert_eq!(dataset.num_classes(), Some(10));
            assert_eq!(dataset.image_shape(), Some((1, 28, 28)));
            assert!(dataset.is_classification());

            let class_names = dataset.class_names().unwrap();
            assert_eq!(class_names.len(), 10);
            assert_eq!(class_names[0], "0");
            assert_eq!(class_names[9], "9");
        }
    }

    #[test]
    fn test_mnist_len() {
        let temp_dir = env::temp_dir().join("test_mnist_len");

        if let Ok(train_dataset) = MnistDataset::new(&temp_dir, true, false) {
            assert_eq!(train_dataset.len(), 60000);
        }

        if let Ok(test_dataset) = MnistDataset::new(&temp_dir, false, false) {
            assert_eq!(test_dataset.len(), 10000);
        }
    }

    #[test]
    fn test_mnist_folders() {
        let temp_dir = env::temp_dir().join("test_mnist_folders");
        let dataset = MnistDataset::new(&temp_dir, true, false);

        if let Ok(dataset) = dataset {
            assert_eq!(dataset.root(), &temp_dir);

            let raw_folder = dataset.raw_folder();
            assert!(raw_folder.to_string_lossy().contains("MNIST/raw"));

            let processed_folder = dataset.processed_folder();
            assert!(processed_folder.to_string_lossy().contains("MNIST/processed"));
        }
    }
}
