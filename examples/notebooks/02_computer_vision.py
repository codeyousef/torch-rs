#!/usr/bin/env python3
"""
Computer Vision with torch-rs Phoenix
====================================

This notebook demonstrates computer vision capabilities including:
- Image preprocessing and transforms
- Pre-trained model usage (ResNet, VGG, ViT)
- Transfer learning
- Custom vision models
- Training with Lightning-style trainer
"""

import numpy as np
try:
    import torch_rs as trs
    from torch_rs import models
except ImportError:
    print("torch-rs not installed. Please install first.")
    exit(1)

print("📷 Computer Vision with Phoenix")
print("==============================\n")

# %% [markdown]
# ## 1. Image Data Preparation
# 
# Let's start by creating synthetic image data and applying transforms.

print("🇮🇲 Image Data Preparation")
print("-" * 30)

# Create synthetic image data (RGB images)
def create_synthetic_images(num_images=100, height=224, width=224):
    """Create synthetic RGB images"""
    images = np.random.rand(num_images, 3, height, width).astype(np.float32)
    # Create dummy labels (10 classes)
    labels = np.random.randint(0, 10, num_images)
    return images, labels

# Generate dataset
train_images, train_labels = create_synthetic_images(1000)
val_images, val_labels = create_synthetic_images(200)

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Validation images shape: {val_images.shape}")
print()

# Convert to Phoenix tensors
train_images_tensor = trs.from_numpy(train_images)
train_labels_tensor = trs.from_numpy(train_labels.astype(np.int64))

print(f"Tensor device: {train_images_tensor.device}")
print(f"Tensor dtype: {train_images_tensor.dtype}")
print()

# %% [markdown]
# ## 2. Image Transforms
# 
# Phoenix provides common image transformations for data preprocessing.

print("🔄 Image Transforms")
print("-" * 30)

# Note: In the full implementation, these would be available
print("Available transforms (conceptual):")
print("- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])")
print("- RandomHorizontalFlip(p=0.5)")
print("- RandomCrop(224, padding=4)")
print("- CenterCrop(224)")
print("- Resize((256, 256))")
print("- ColorJitter(brightness=0.4, contrast=0.4)")
print()

# Simple normalization example
def normalize_images(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Simple normalization function"""
    mean = np.array(mean).reshape(1, 3, 1, 1)
    std = np.array(std).reshape(1, 3, 1, 1)
    return (images - mean) / std

normalized_images = normalize_images(train_images[:5])
print(f"Normalized images shape: {normalized_images.shape}")
print(f"Original range: [{train_images.min():.3f}, {train_images.max():.3f}]")
print(f"Normalized range: [{normalized_images.min():.3f}, {normalized_images.max():.3f}]")
print()

# %% [markdown]
# ## 3. Pre-trained Models
# 
# Phoenix provides several pre-trained computer vision models.

print("🏭 Pre-trained Models")
print("-" * 30)

print("Available pre-trained models:")
print("\n🏭 ResNet Family:")
print("- ResNet18, ResNet34, ResNet50, ResNet101, ResNet152")
print("- Pre-trained on ImageNet")
print("- Suitable for image classification and feature extraction")

print("\n🖼️ VGG Family:")
print("- VGG11, VGG13, VGG16, VGG19")
print("- With and without batch normalization")
print("- Classic CNN architecture")

print("\n🤖 Vision Transformers:")
print("- ViT-B/16, ViT-B/32 (Base models)")
print("- ViT-L/16, ViT-L/32 (Large models)")
print("- State-of-the-art transformer-based vision")
print()

# Model creation examples (conceptual)
print("Model creation examples:")
print("""
# ResNet50
model = models.resnet50(num_classes=1000)
# model.download_pretrained("resnet50")  # Load ImageNet weights

# Vision Transformer
vit_model = models.vit_base_patch16_224(num_classes=1000)
# vit_model.download_pretrained("vit_b_16")

# VGG16 with batch normalization
vgg_model = models.vgg16(num_classes=1000, batch_norm=True)
""")
print()

# %% [markdown]
# ## 4. Custom Vision Model
# 
# Let's build a custom CNN for our classification task.

print("🔨 Custom Vision Model")
print("-" * 30)

# Create a simple CNN
class SimpleCNN:
    """Simple CNN model for demonstration"""
    
    def __init__(self, num_classes=10):
        self.features = trs.Sequential()
        # Note: In full implementation, these would be Conv2d, BatchNorm2d, etc.
        print("Creating CNN with conceptual layers:")
        print("- Conv2d(3, 64, kernel_size=3, padding=1)")
        print("- BatchNorm2d(64)")
        print("- ReLU()")
        print("- MaxPool2d(2)")
        print("- Conv2d(64, 128, kernel_size=3, padding=1)")
        print("- BatchNorm2d(128)")
        print("- ReLU()")
        print("- MaxPool2d(2)")
        print("- AdaptiveAvgPool2d((7, 7))")
        
        # Classifier (we can implement this part)
        self.classifier = trs.Sequential()
        # Assuming flattened features are 128 * 7 * 7 = 6272
        self.classifier.add(trs.Linear(6272, 512))
        self.classifier.add(trs.Linear(512, 256))
        self.classifier.add(trs.Linear(256, num_classes))
        
        print(f"\nClassifier created with {num_classes} output classes")
    
    def forward(self, x):
        # For demonstration, we'll simulate feature extraction
        # In reality, this would go through conv layers
        batch_size = x.shape[0]
        
        # Simulate feature extraction (flatten and reduce)
        features = trs.randn([batch_size, 6272])
        
        # Apply classifier
        output = self.classifier.forward(features)
        return output

# Create model
model = SimpleCNN(num_classes=10)
print("\nModel created successfully!")
print()

# %% [markdown]
# ## 5. Training Setup
# 
# Set up training components: loss function, optimizer, and metrics.

print("🎓 Training Setup")
print("-" * 30)

# Create optimizer
optimizer = trs.Adam(model.classifier.parameters(), lr=0.001)
print("Optimizer: Adam(lr=0.001)")

# Simple accuracy metric
class AccuracyMetric:
    def __init__(self):
        self.reset()
    
    def update(self, predictions, targets):
        pred_classes = np.argmax(trs.to_numpy(predictions), axis=1)
        target_classes = trs.to_numpy(targets)
        correct = np.sum(pred_classes == target_classes)
        self.total_correct += correct
        self.total_samples += len(targets)
    
    def compute(self):
        if self.total_samples == 0:
            return 0.0
        return self.total_correct / self.total_samples
    
    def reset(self):
        self.total_correct = 0
        self.total_samples = 0

accuracy_metric = AccuracyMetric()
print("Metrics: Accuracy")
print()

# %% [markdown]
# ## 6. Training Loop
# 
# Implement training with validation.

print("🏋️ Training Loop")
print("-" * 30)

def create_batches(images, labels, batch_size=32):
    """Create batches from images and labels"""
    num_samples = len(images)
    indices = np.random.permutation(num_samples)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        
        batch_images = trs.from_numpy(images[batch_indices])
        batch_labels = trs.from_numpy(labels[batch_indices].astype(np.int64))
        
        yield batch_images, batch_labels

def simple_cross_entropy_loss(predictions, targets):
    """Simple cross-entropy loss implementation"""
    # Convert to probabilities
    probs = predictions.softmax(-1)
    
    # Simple loss calculation (for demonstration)
    pred_np = trs.to_numpy(probs)
    target_np = trs.to_numpy(targets)
    
    # Compute negative log likelihood
    loss_value = 0.0
    for i, target in enumerate(target_np):
        loss_value -= np.log(pred_np[i, int(target)] + 1e-8)
    
    return trs.from_numpy(np.array([loss_value / len(target_np)]))

# Training parameters
num_epochs = 3
batch_size = 32

print(f"Training for {num_epochs} epochs with batch size {batch_size}")
print("=" * 50)

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 20)
    
    # Training phase
    epoch_loss = 0.0
    num_batches = 0
    accuracy_metric.reset()
    
    for batch_images, batch_labels in create_batches(train_images, train_labels, batch_size):
        # Forward pass
        predictions = model.forward(batch_images)
        loss = simple_cross_entropy_loss(predictions, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Update metrics
        loss_value = trs.to_numpy(loss)[0]
        epoch_loss += loss_value
        accuracy_metric.update(predictions, batch_labels)
        num_batches += 1
        
        if num_batches % 10 == 0:
            current_acc = accuracy_metric.compute()
            print(f"  Batch {num_batches:3d}: Loss = {loss_value:.4f}, Acc = {current_acc:.3f}")
    
    # Epoch summary
    avg_loss = epoch_loss / num_batches
    train_acc = accuracy_metric.compute()
    print(f"\n  Training   - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.3f}")
    
    # Validation phase
    accuracy_metric.reset()
    val_loss = 0.0
    val_batches = 0
    
    for batch_images, batch_labels in create_batches(val_images, val_labels, batch_size):
        # Forward pass only (no gradients)
        predictions = model.forward(batch_images)
        loss = simple_cross_entropy_loss(predictions, batch_labels)
        
        # Update metrics
        val_loss += trs.to_numpy(loss)[0]
        accuracy_metric.update(predictions, batch_labels)
        val_batches += 1
    
    val_avg_loss = val_loss / val_batches
    val_acc = accuracy_metric.compute()
    print(f"  Validation - Loss: {val_avg_loss:.4f}, Accuracy: {val_acc:.3f}")

print("\n🎉 Training completed!")
print()

# %% [markdown]
# ## 7. Transfer Learning Example
# 
# Demonstrate transfer learning with pre-trained models.

print("🔄 Transfer Learning")
print("-" * 30)

print("Transfer learning workflow:")
print("""
1. Load pre-trained model:
   model = models.resnet50(num_classes=1000)
   model.download_pretrained("resnet50")

2. Freeze backbone:
   for param in model.features.parameters():
       param.requires_grad = False

3. Replace classifier:
   model.classifier = trs.Linear(2048, num_custom_classes)

4. Train only the classifier:
   optimizer = trs.Adam(model.classifier.parameters(), lr=0.001)

5. Fine-tune (optional):
   # Unfreeze some layers
   for param in model.features[-2:].parameters():
       param.requires_grad = True
   
   # Lower learning rate
   optimizer = trs.Adam(model.parameters(), lr=0.0001)
""")
print()

# %% [markdown]
# ## 8. Model Evaluation and Inference
# 
# Evaluate the trained model and run inference.

print("📊 Model Evaluation")
print("-" * 30)

# Test on a batch
test_batch_images, test_batch_labels = next(create_batches(val_images, val_labels, 10))

# Inference
predictions = model.forward(test_batch_images)
probs = predictions.softmax(-1)

# Get predicted classes
pred_classes = np.argmax(trs.to_numpy(probs), axis=1)
true_classes = trs.to_numpy(test_batch_labels)

print("Sample predictions:")
for i in range(min(5, len(pred_classes))):
    confidence = trs.to_numpy(probs)[i, pred_classes[i]]
    print(f"  Image {i}: Predicted = {pred_classes[i]}, "
          f"True = {true_classes[i]}, Confidence = {confidence:.3f}")
print()

# %% [markdown]
# ## 9. Performance Tips
# 
# Best practices for computer vision with Phoenix.

print("⚡ Performance Tips")
print("-" * 30)

print("🚀 Optimization strategies:")
print()
print("1. 💾 Memory:")
print("   - Use appropriate batch sizes")
print("   - Enable mixed precision training (FP16)")
print("   - Clear gradients regularly")
print()
print("2. 📱 Data Loading:")
print("   - Use efficient data loaders")
print("   - Preprocess data offline when possible")
print("   - Use data augmentation wisely")
print()
print("3. 🎯 Model Architecture:")
print("   - Start with pre-trained models")
print("   - Use appropriate model size for your data")
print("   - Consider mobile-optimized architectures")
print()
print("4. 🖥️ Hardware:")
print("   - Use GPU for training (model.to('cuda'))")
print("   - Batch operations for efficiency")
print("   - Monitor GPU memory usage")
print()

# %% [markdown]
# ## 10. Next Steps
# 
# Where to go from here.

print("🚀 Next Steps")
print("-" * 30)

print("Explore more advanced topics:")
print()
print("🔍 Object Detection:")
print("- YOLO, R-CNN models")
print("- Bounding box regression")
print("- Multi-class object detection")
print()
print("🗺️ Semantic Segmentation:")
print("- U-Net, DeepLab models")
print("- Pixel-level classification")
print("- Medical image segmentation")
print()
print("🎨 Generative Models:")
print("- GANs for image generation")
print("- VAEs for latent representations")
print("- Style transfer applications")
print()
print("📋 Recommended resources:")
print("- torch-rs documentation: TORCH_RS_GUIDE.md")
print("- Computer vision examples in examples/")
print("- Pre-trained model zoo")
print("- Community tutorials and papers")

print("\n🎉 Computer Vision Tutorial Complete!")
print("You've learned the fundamentals of CV with Phoenix.")
print("Happy coding! 🚀")