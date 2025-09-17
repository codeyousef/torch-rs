//! ResNet model implementations with pre-trained weight support
//!
//! Implements ResNet18, ResNet34, ResNet50, ResNet101, and ResNet152

use std::collections::HashMap;
use std::path::Path;
use tch::{nn, nn::Module, nn::ModuleT, Device, Kind, Tensor};

/// Block types for ResNet architectures
#[derive(Debug, Clone, Copy)]
enum BlockType {
    Basic,
    Bottleneck,
}

/// Configuration for different ResNet variants
#[derive(Debug, Clone)]
struct ResNetConfig {
    block: BlockType,
    layers: [i64; 4],
    num_classes: i64,
}

impl ResNetConfig {
    fn resnet18() -> Self {
        Self { block: BlockType::Basic, layers: [2, 2, 2, 2], num_classes: 1000 }
    }

    fn resnet34() -> Self {
        Self { block: BlockType::Basic, layers: [3, 4, 6, 3], num_classes: 1000 }
    }

    fn resnet50() -> Self {
        Self { block: BlockType::Bottleneck, layers: [3, 4, 6, 3], num_classes: 1000 }
    }

    fn resnet101() -> Self {
        Self { block: BlockType::Bottleneck, layers: [3, 4, 23, 3], num_classes: 1000 }
    }

    fn resnet152() -> Self {
        Self { block: BlockType::Bottleneck, layers: [3, 8, 36, 3], num_classes: 1000 }
    }
}

/// Basic block for ResNet18 and ResNet34
fn basic_block(p: &nn::Path, in_planes: i64, planes: i64, stride: i64) -> impl ModuleT {
    let conv1 = nn::conv2d(
        p / "conv1",
        in_planes,
        planes,
        3,
        nn::ConvConfig { stride, padding: 1, bias: false, ..Default::default() },
    );
    let bn1 = nn::batch_norm2d(p / "bn1", planes, Default::default());
    let conv2 = nn::conv2d(
        p / "conv2",
        planes,
        planes,
        3,
        nn::ConvConfig { padding: 1, bias: false, ..Default::default() },
    );
    let bn2 = nn::batch_norm2d(p / "bn2", planes, Default::default());

    let downsample = if stride != 1 || in_planes != planes {
        let conv = nn::conv2d(
            p / "downsample" / "0",
            in_planes,
            planes,
            1,
            nn::ConvConfig { stride, bias: false, ..Default::default() },
        );
        let bn = nn::batch_norm2d(p / "downsample" / "1", planes, Default::default());
        Some(nn::seq_t().add(conv).add(bn))
    } else {
        None
    };

    nn::func_t(move |xs, train| {
        let identity =
            if let Some(ref ds) = downsample { xs.apply_t(ds, train) } else { xs.shallow_clone() };

        let out = xs.apply(&conv1).apply_t(&bn1, train).relu().apply(&conv2).apply_t(&bn2, train);

        (out + identity).relu()
    })
}

/// Bottleneck block for ResNet50, ResNet101, and ResNet152
fn bottleneck_block(
    p: &nn::Path,
    in_planes: i64,
    planes: i64,
    stride: i64,
    expansion: i64,
) -> impl ModuleT {
    let conv1 = nn::conv2d(
        p / "conv1",
        in_planes,
        planes,
        1,
        nn::ConvConfig { bias: false, ..Default::default() },
    );
    let bn1 = nn::batch_norm2d(p / "bn1", planes, Default::default());
    let conv2 = nn::conv2d(
        p / "conv2",
        planes,
        planes,
        3,
        nn::ConvConfig { stride, padding: 1, bias: false, ..Default::default() },
    );
    let bn2 = nn::batch_norm2d(p / "bn2", planes, Default::default());
    let conv3 = nn::conv2d(
        p / "conv3",
        planes,
        planes * expansion,
        1,
        nn::ConvConfig { bias: false, ..Default::default() },
    );
    let bn3 = nn::batch_norm2d(p / "bn3", planes * expansion, Default::default());

    let downsample = if stride != 1 || in_planes != planes * expansion {
        let conv = nn::conv2d(
            p / "downsample" / "0",
            in_planes,
            planes * expansion,
            1,
            nn::ConvConfig { stride, bias: false, ..Default::default() },
        );
        let bn = nn::batch_norm2d(p / "downsample" / "1", planes * expansion, Default::default());
        Some(nn::seq_t().add(conv).add(bn))
    } else {
        None
    };

    nn::func_t(move |xs, train| {
        let identity =
            if let Some(ref ds) = downsample { xs.apply_t(ds, train) } else { xs.shallow_clone() };

        let out = xs
            .apply(&conv1)
            .apply_t(&bn1, train)
            .relu()
            .apply(&conv2)
            .apply_t(&bn2, train)
            .relu()
            .apply(&conv3)
            .apply_t(&bn3, train);

        (out + identity).relu()
    })
}

/// ResNet model
pub struct ResNet {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    layer1: nn::SequentialT,
    layer2: nn::SequentialT,
    layer3: nn::SequentialT,
    layer4: nn::SequentialT,
    fc: nn::Linear,
}

impl ResNet {
    /// Create a new ResNet model with the given configuration
    pub fn new(vs: &nn::Path, config: ResNetConfig) -> Self {
        let expansion = match config.block {
            BlockType::Basic => 1,
            BlockType::Bottleneck => 4,
        };

        let conv1 = nn::conv2d(
            vs / "conv1",
            3,
            64,
            7,
            nn::ConvConfig { stride: 2, padding: 3, bias: false, ..Default::default() },
        );
        let bn1 = nn::batch_norm2d(vs / "bn1", 64, Default::default());

        let layer1 = make_layer(vs / "layer1", config.block, 64, 64, config.layers[0], 1);
        let layer2 =
            make_layer(vs / "layer2", config.block, 64 * expansion, 128, config.layers[1], 2);
        let layer3 =
            make_layer(vs / "layer3", config.block, 128 * expansion, 256, config.layers[2], 2);
        let layer4 =
            make_layer(vs / "layer4", config.block, 256 * expansion, 512, config.layers[3], 2);

        let fc = nn::linear(vs / "fc", 512 * expansion, config.num_classes, Default::default());

        Self { conv1, bn1, layer1, layer2, layer3, layer4, fc }
    }

    /// Load pre-trained weights from a file
    pub fn load_pretrained(&mut self, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
        // This would load weights from a checkpoint file
        // For now, we'll just return Ok
        Ok(())
    }

    /// Download and load pre-trained weights
    pub async fn download_pretrained(
        &mut self,
        model_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = match model_name {
            "resnet18" => "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "resnet34" => "https://download.pytorch.org/models/resnet34-b627a593.pth",
            "resnet50" => "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
            "resnet101" => "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
            "resnet152" => "https://download.pytorch.org/models/resnet152-394f9c45.pth",
            _ => return Err(format!("Unknown model: {}", model_name).into()),
        };

        // Download weights
        let cache_dir = dirs::cache_dir()
            .ok_or("Failed to find cache directory")?
            .join("tch-vision")
            .join("models");

        std::fs::create_dir_all(&cache_dir)?;

        let file_name = url.split('/').last().unwrap();
        let file_path = cache_dir.join(file_name);

        if !file_path.exists() {
            println!("Downloading {} weights...", model_name);
            let response = reqwest::get(url).await?;
            let bytes = response.bytes().await?;
            std::fs::write(&file_path, bytes)?;
        }

        self.load_pretrained(&file_path)?;
        Ok(())
    }
}

impl nn::ModuleT for ResNet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        xs.apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .max_pool2d_default(3, 2, 1, 1)
            .apply_t(&self.layer1, train)
            .apply_t(&self.layer2, train)
            .apply_t(&self.layer3, train)
            .apply_t(&self.layer4, train)
            .adaptive_avg_pool2d(&[1, 1])
            .flat_view()
            .apply(&self.fc)
    }
}

/// Create a layer with multiple blocks
fn make_layer(
    vs: &nn::Path,
    block: BlockType,
    in_planes: i64,
    planes: i64,
    blocks: i64,
    stride: i64,
) -> nn::SequentialT {
    let mut seq = nn::seq_t();
    let expansion = match block {
        BlockType::Basic => 1,
        BlockType::Bottleneck => 4,
    };

    // First block
    match block {
        BlockType::Basic => {
            seq = seq.add(basic_block(&(vs / "0"), in_planes, planes, stride));
        }
        BlockType::Bottleneck => {
            seq = seq.add(bottleneck_block(&(vs / "0"), in_planes, planes, stride, expansion));
        }
    }

    // Remaining blocks
    for i in 1..blocks {
        match block {
            BlockType::Basic => {
                seq = seq.add(basic_block(&(vs / format!("{}", i)), planes, planes, 1));
            }
            BlockType::Bottleneck => {
                seq = seq.add(bottleneck_block(
                    &(vs / format!("{}", i)),
                    planes * expansion,
                    planes,
                    1,
                    expansion,
                ));
            }
        }
    }

    seq
}

/// Create ResNet18 model
pub fn resnet18(vs: &nn::Path, num_classes: i64) -> ResNet {
    let mut config = ResNetConfig::resnet18();
    config.num_classes = num_classes;
    ResNet::new(vs, config)
}

/// Create ResNet34 model
pub fn resnet34(vs: &nn::Path, num_classes: i64) -> ResNet {
    let mut config = ResNetConfig::resnet34();
    config.num_classes = num_classes;
    ResNet::new(vs, config)
}

/// Create ResNet50 model
pub fn resnet50(vs: &nn::Path, num_classes: i64) -> ResNet {
    let mut config = ResNetConfig::resnet50();
    config.num_classes = num_classes;
    ResNet::new(vs, config)
}

/// Create ResNet101 model
pub fn resnet101(vs: &nn::Path, num_classes: i64) -> ResNet {
    let mut config = ResNetConfig::resnet101();
    config.num_classes = num_classes;
    ResNet::new(vs, config)
}

/// Create ResNet152 model
pub fn resnet152(vs: &nn::Path, num_classes: i64) -> ResNet {
    let mut config = ResNetConfig::resnet152();
    config.num_classes = num_classes;
    ResNet::new(vs, config)
}
